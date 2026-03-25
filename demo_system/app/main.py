import logging
import time
import uuid

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


class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_CONTENT_LENGTH:
            return JSONResponse(status_code=413, content={"error": "上传文件过大"})
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


app = FastAPI(title="Retinal Vessel Segmentation Demo", version="1.1.0")
app.add_middleware(UploadSizeLimitMiddleware)
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
