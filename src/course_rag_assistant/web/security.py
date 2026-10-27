"""Local request checks and browser security headers."""
import secrets
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware


def install_security(app: FastAPI, token: str):
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
