"""Application composition and local server entry point."""
from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..answer import generator_from_env
from ..config import Settings, get_settings
from .compare_routes import create_compare_router
from .constants import STATIC
from .library import Library
from .quiz_routes import create_quiz_router
from .routes import create_router
from .security import install_security


def create_app(storage: Path | None = None, *, generator_factory=None,
               settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()
    if generator_factory is None:
        def generator_factory():
            return generator_from_env(config)
    app = FastAPI(title="课内 · 资料问答")
    library = Library(storage if storage is not None else config.web_data)
    app.state.library = library
    token = secrets.token_urlsafe(32)
    install_security(app, token)
    app.include_router(create_router(library, generator_factory, token))
    app.include_router(create_quiz_router(library))
    app.include_router(create_compare_router(library))
    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")
    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="启动课程资料问答网页（仅本机访问）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port)
