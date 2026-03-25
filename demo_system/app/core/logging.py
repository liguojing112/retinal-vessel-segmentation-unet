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
