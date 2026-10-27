"""Document, state and question-answering HTTP routes."""
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ..ingest import Chunk, ingest_dir
from .answers import ask_materials
from .constants import MAX_UPLOAD, STATIC, SUPPORTED, TEMPLATES
from .schemas import AskRequest


def create_router(library, generator_factory, token):
    router = APIRouter()
    model_lock = threading.Lock()
    @router.get("/")
    def home():
        return FileResponse(TEMPLATES / "index.html")


    @router.get("/api/state")
    def state():
        try:
            configured = generator_factory() is not None
        except (ValueError, TypeError):
            configured = False
        with library.lock:
            return {"token": token, "documents": [d.public() for d in library.documents.values()],
                    "model_configured": configured, "unreadable_documents": library.unreadable,
                    "max_upload_bytes": MAX_UPLOAD}

    @router.post("/api/documents", status_code=201)
    async def upload(request: Request, filename: str):
        if Path(filename).suffix.lower() not in SUPPORTED:
            raise HTTPException(415, "请上传 PDF、PPTX 或 Markdown（.md）文件。")
        content = bytearray()
        async for part in request.stream():
            content.extend(part)
            if len(content) > MAX_UPLOAD:
                raise HTTPException(413, f"单个文件不能超过 {MAX_UPLOAD // (1024 * 1024)} MB。")
        document = await run_in_threadpool(library.add, filename, bytes(content))
        return document.public()

    @router.post("/api/demo", status_code=201)
    def demo():
        with library.lock:
            sample = ingest_dir(STATIC / "sample")
            for document in library.documents.values():
                expected = [Chunk(c.text, document.source, c.page, c.section, c.concepts) for c in sample]
                if document.name == "机器学习入门示例.md" and document.chunks == expected:
                    return document.public()
            return library.add("机器学习入门示例.md", (STATIC / "sample" / "example.md").read_bytes()).public()

    @router.get("/api/documents/{document_id}/file")
    def original(document_id: str):
        document = library.selected([document_id])[0]
        pdf = document.source.endswith(".pdf")
        return FileResponse(library.root / document.id / document.source,
                            media_type="application/pdf" if pdf else "application/octet-stream",
                            filename=document.name, content_disposition_type="inline" if pdf else "attachment")


    @router.post("/api/ask")
    def ask(payload: AskRequest):
        return ask_materials(payload, library, generator_factory, model_lock)

    # Keep pre-refactor asset URLs working for cached pages and local clients.
    @router.get("/assets/app.js", include_in_schema=False)
    def legacy_script():
        return FileResponse(STATIC / "js" / "app.js")

    @router.get("/assets/style.css", include_in_schema=False)
    def legacy_style():
        return FileResponse(STATIC / "css" / "style.css")

    return router
