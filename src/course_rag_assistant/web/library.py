"""Local document storage, validation and per-library retrieval caching."""
from __future__ import annotations

import json
import re
import threading
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from ..ingest import Chunk, ingest_dir
from ..retrieval import HybridRetriever
from .constants import MAX_UPLOAD, SUPPORTED


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
        self._retrievers: dict[tuple[str, ...], HybridRetriever] = {}
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
            raise HTTPException(413, f"文件不能为空，单个文件不能超过 {MAX_UPLOAD // (1024 * 1024)} MB。")
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

    def retriever(self, ids: tuple[str, ...]) -> HybridRetriever:
        with self.lock:
            if ids not in self._retrievers:
                self._retrievers[ids] = HybridRetriever([chunk for doc in self.selected(list(ids)) for chunk in doc.chunks])
                if len(self._retrievers) > 8:
                    self._retrievers.pop(next(iter(self._retrievers)))
            return self._retrievers[ids]
