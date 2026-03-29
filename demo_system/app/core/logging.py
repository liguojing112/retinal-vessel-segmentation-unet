"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

日志工具模块。

提供 request_id 上下文注入与统一日志格式配置能力。"""


import logging
import time
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | req=%(request_id)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)


def now_ms() -> int:
    return int(time.time() * 1000)
