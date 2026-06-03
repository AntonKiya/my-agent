import json
import logging
import sys
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from agent_service.config import AppSettings
from agent_service.observability.logfire_integration import configure_logfire
from agent_service.observability.tracing import get_trace_id

_RESERVED_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        trace_id = get_trace_id()
        if trace_id is not None:
            payload["trace_id"] = trace_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = value

        exception_text = _exception_text(self, record)
        if exception_text is not None:
            payload["exception"] = exception_text

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: AppSettings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level)

    logging.getLogger("agent_service").setLevel(settings.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def configure_observability(settings: AppSettings) -> None:
    configure_logging(settings)
    configure_logfire(settings)


def _exception_text(formatter: logging.Formatter, record: logging.LogRecord) -> str | None:
    exc_info = _exception_info(record.exc_info)
    if exc_info is not None:
        return formatter.formatException(exc_info)
    if isinstance(record.exc_text, str) and record.exc_text:
        return record.exc_text
    return None


def _exception_info(
    exc_info: object,
) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
    if isinstance(exc_info, BaseException):
        return (type(exc_info), exc_info, exc_info.__traceback__)
    if not isinstance(exc_info, tuple) or len(exc_info) != 3:
        return None

    exc_type, exc, traceback = exc_info
    if not isinstance(exc_type, type) or not issubclass(exc_type, BaseException):
        return None
    if not isinstance(exc, BaseException):
        return None
    if traceback is not None and not isinstance(traceback, TracebackType):
        return None

    return (exc_type, exc, traceback)
