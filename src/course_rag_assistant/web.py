"""Local material uploads, cited question answering, and original-file access."""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import uuid
import zipfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .answer import _CITATION_RE, _parse_citations, _sentences, answer, generator_from_env
from .ingest import Chunk, ingest_dir
from .rerank import ChatReranker
from .retrieval import HybridRetriever

ASSETS = Path(__file__).with_name("web_assets")
MAX_UPLOAD = 25 * 1024 * 1024
SUPPORTED = {".pdf", ".pptx", ".md"}


@dataclass
class Document:
    id: str
    name: str
    source: str
    size: int
    chunks: list[Chunk]

    def public(self):
        return {"id": self.id, "name": self.name, "size": self.size,
                "type": Path(self.source).suffix[1:], "chunks": len(self.chunks),
                "pages": max((c.page for c in self.chunks), default=0)}


class Library:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.documents: dict[str, Document] = {}
        self.unreadable = 0
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", folder.name):
                continue
            try:
                data = json.loads((folder / "document.json").read_text(encoding="utf-8"))
                document = Document(**{**data, "chunks": [Chunk(**c) for c in data["chunks"]]})
                if (document.id != folder.name or Path(document.source).name != document.source
                        or not (folder / document.source).is_file()
                        or any(c.doc != document.source for c in document.chunks)):
                    raise ValueError("Invalid stored document")
                self.documents[document.id] = document
            except (OSError, ValueError, KeyError, TypeError):
                self.unreadable += 1

    def add(self, filename: str, content: bytes) -> Document:
        # Uploaded names are display metadata, never paths. Unique citation
        # names prevent same-named documents from sharing evidence by accident.
        name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(415, "请上传 PDF、PPTX 或 Markdown（.md）文件。")
        if not content or len(content) > MAX_UPLOAD:
            raise HTTPException(413, "文件不能为空，单个文件不能超过 25 MB。")
        if len(name) > 180 or any(ord(c) < 32 for c in name):
            raise HTTPException(400, "文件名过长或包含不可见字符，请重命名后上传。")
        with self.lock:
            if len(self.documents) >= 50:
                raise HTTPException(409, "本地资料库最多保存 50 份资料。")
            document_id = uuid.uuid4().hex
            stem = re.sub(r"[^\w\-\u4e00-\u9fff]", "_", Path(name).stem)[:60] or "material"
            source = f"{stem}__{document_id}{suffix}"
            with TemporaryDirectory(prefix=".upload-", dir=self.root) as temporary:
                stage = Path(temporary)
                path = stage / source
                path.write_bytes(content)
                try:
                    if suffix == ".pdf":
                        # Validate from memory first: failed file-based PDF
                        # opens can retain a Windows file handle in a traceback.
                        import pymupdf

                        with pymupdf.open(stream=content, filetype="pdf") as pdf:
                            if pdf.needs_pass or len(pdf) > 1500:
                                raise ValueError("Encrypted or oversized PDF")
                    if suffix == ".pptx":
                        with zipfile.ZipFile(path) as archive:
                            if sum(i.file_size for i in archive.infolist()) > 80 * 1024 * 1024:
                                raise ValueError("Expanded presentation is too large")
                    chunks = ingest_dir(stage)
                except Exception as exc:
                    raise HTTPException(422, "无法读取这份文件。请确认文件未损坏、未加密，且格式正确。") from exc
                if not chunks or not any(c.text.strip() for c in chunks):
                    raise HTTPException(422, "没有读到文字。扫描版 PDF 请先转为可复制文字的版本。")
                if len(chunks) > 10000 or sum(len(c.text) for c in chunks) > 8_000_000:
                    raise HTTPException(413, "这份资料内容过多，请按章节拆分后上传。")
                document = Document(document_id, name, source, len(content), chunks)
                (stage / "document.json").write_text(json.dumps(asdict(document), ensure_ascii=False), encoding="utf-8")
                stage.rename(self.root / document_id)
            self.documents[document_id] = document
            return document

    def selected(self, ids: list[str]) -> list[Document]:
        with self.lock:
            if not ids:
                raise HTTPException(400, "请先选择至少一份资料。")
            if any(doc_id not in self.documents for doc_id in ids):
                raise HTTPException(404, "所选资料已不可用，请刷新资料列表。")
            return [self.documents[doc_id] for doc_id in dict.fromkeys(ids)]

    @lru_cache(maxsize=8)
    def retriever(self, ids: tuple[str, ...]) -> HybridRetriever:
        return HybridRetriever([chunk for doc in self.selected(list(ids)) for chunk in doc.chunks])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[str] = Field(min_length=1, max_length=50)
    use_model: bool = False


class ObservedGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.failed = False

    def generate(self, prompt):
        try:
            return self.generator.generate(prompt)
        except Exception:
            self.failed = True
            raise


def create_app(storage: Path | None = None, *, generator_factory=generator_from_env) -> FastAPI:
    app = FastAPI(title="课内 · 资料问答")
    library = Library(storage or Path(os.getenv("COURSE_RAG_WEB_DATA", "data/web")))
    app.state.library = library
    token = secrets.token_urlsafe(32)
    model_lock = threading.Lock()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        # No CORS is enabled. The custom header blocks cross-site form uploads
        # and model requests against this local service.
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            same_origin = not origin or urlsplit(origin).netloc == request.headers.get("host")
            if not same_origin or not secrets.compare_digest(request.headers.get("x-workspace-token", ""), token):
                return JSONResponse({"detail": "页面连接已失效，请刷新后重试。"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home():
        return FileResponse(ASSETS / "index.html")

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")

    @app.get("/api/state")
    def state():
        try:
            configured = generator_factory() is not None
        except (ValueError, TypeError):
            configured = False
        with library.lock:
            return {"token": token, "documents": [d.public() for d in library.documents.values()],
                    "model_configured": configured, "unreadable_documents": library.unreadable,
                    "max_upload_bytes": MAX_UPLOAD}

    @app.post("/api/documents", status_code=201)
    async def upload(request: Request, filename: str):
        if Path(filename).suffix.lower() not in SUPPORTED:
            raise HTTPException(415, "请上传 PDF、PPTX 或 Markdown（.md）文件。")
        content = bytearray()
        async for part in request.stream():
            content.extend(part)
            if len(content) > MAX_UPLOAD:
                raise HTTPException(413, "单个文件不能超过 25 MB。")
        document = await run_in_threadpool(library.add, filename, bytes(content))
        return document.public()

    @app.post("/api/demo", status_code=201)
    def demo():
        with library.lock:
            sample = ingest_dir(ASSETS / "sample")
            for document in library.documents.values():
                expected = [Chunk(c.text, document.source, c.page, c.section, c.concepts) for c in sample]
                if document.name == "机器学习入门示例.md" and document.chunks == expected:
                    return document.public()
            return library.add("机器学习入门示例.md", (ASSETS / "sample" / "example.md").read_bytes()).public()

    @app.get("/api/documents/{document_id}/file")
    def original(document_id: str):
        document = library.selected([document_id])[0]
        pdf = document.source.endswith(".pdf")
        return FileResponse(library.root / document.id / document.source,
                            media_type="application/pdf" if pdf else "application/octet-stream",
                            filename=document.name, content_disposition_type="inline" if pdf else "attachment")

    @app.post("/api/ask")
    def ask(payload: AskRequest):
        question = payload.question.strip()
        if not question:
            raise HTTPException(400, "请输入你想问的问题。")
        documents = library.selected(payload.document_ids)
        base = library.retriever(tuple(sorted(d.id for d in documents)))
        try:
            generator = generator_factory() if payload.use_model else None
        except (ValueError, TypeError) as exc:
            raise HTTPException(503, "模型配置无法使用，请关闭模型辅助后重试。") from exc
        if payload.use_model and generator is None:
            raise HTTPException(409, "尚未配置模型，请关闭模型辅助后重试。")
        if not model_lock.acquire(blocking=False):
            raise HTTPException(409, "正在处理上一个问题，请稍后再试。")
        try:
            trace = ObservedGenerator(generator) if generator else None
            # Uploaded materials have their own lexical index. Never attach the
            # existing textbook vectors to an unrelated uploaded document set.
            retriever = HybridRetriever(base.chunks, reranker=ChatReranker(generator)) if generator else base
            result = answer(question, retriever.search(question, 8), trace)
        finally:
            model_lock.release()
        by_source = {d.source: d for d in documents}
        sources, source_ids, claims = [], {}, []
        for line in _sentences(result.text):
            ids = []
            for source_name, location in _parse_citations(line):
                document = by_source.get(source_name)
                hits = [h for h in result.hits if h.chunk.doc == source_name and
                        (f"p.{h.chunk.page}" == location if location.startswith("p.") else
                         f"section={h.chunk.section or 'document'}" == location)]
                if not document or not hits:
                    continue
                key = (source_name, location)
                if key not in source_ids:
                    sid = str(len(sources) + 1)
                    source_ids[key] = sid
                    page = hits[0].chunk.page
                    sources.append({"id": sid, "document_id": document.id, "name": document.name,
                                    "page": page, "section": hits[0].chunk.section,
                                    "text": "\n\n".join(dict.fromkeys(h.chunk.text for h in hits)),
                                    "url": f"/api/documents/{document.id}/file" + (f"#page={page}" if page else ""),
                                    "type": Path(document.source).suffix[1:]})
                if source_ids[key] not in ids:
                    ids.append(source_ids[key])
            claims.append({"text": _CITATION_RE.sub("", line).strip(), "source_ids": ids})
        mode = "source" if not trace else "fallback" if trace.failed else "model"
        return {"question": question, "claims": claims, "sources": sources, "mode": mode,
                "abstained": result.abstention_reason is not None,
                "reason": result.abstention_reason, "document_names": [d.name for d in documents]}

    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="启动课程资料问答网页（仅本机访问）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", type=Path, default=Path("data/web"))
    args = parser.parse_args()
    uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
