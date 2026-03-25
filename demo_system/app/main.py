import logging
import os
import time
import uuid
from collections import defaultdict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.api.routes_predict import router as predict_router
from app.core.config import LOG_LEVEL, MAX_CONTENT_LENGTH
from app.core.errors import AppError
from app.core.logging import configure_logging, request_id_ctx

configure_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))


class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_CONTENT_LENGTH:
            return JSONResponse(status_code=413, content={"error": "上传文件过大"})
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于 IP 的简易内存限流（每分钟 N 次请求）。"""

    def __init__(self, app, rpm: int = 60) -> None:
        super().__init__(app)
        self.rpm = rpm
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = [t for t in self._hits[client_ip] if now - t < 60]
        window.append(now)
        self._hits[client_ip] = window
        if len(window) > self.rpm:
            return JSONResponse(status_code=429, content={"error": "请求过于频繁，请稍后再试"})
        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info("%s %s status=%s cost_ms=%s", request.method, request.url.path, getattr(response, "status_code", "-"), elapsed_ms)
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = request_id
        return response


app = FastAPI(title="Retinal Vessel Segmentation Demo", version="1.2.0")
app.add_middleware(UploadSizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware, rpm=RATE_LIMIT_RPM)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.message,
            "code": exc.code,
            "request_id": request.state.request_id,
        },
    )


@app.exception_handler(Exception)
async def fallback_error_handler(request: Request, exc: Exception):
    logger.exception("未处理异常")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "code": "INTERNAL_ERROR",
            "request_id": request.state.request_id,
        },
    )
