import logging
from time import perf_counter
from typing import Any

EventFields = dict[str, Any]


def start_timer() -> float:
    return perf_counter()


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    event: str,
    **fields: object,
) -> None:
    logger.log(level, message, extra=_extra(event=event, fields=fields))


def log_exception(
    logger: logging.Logger,
    message: str,
    *,
    event: str,
    **fields: object,
) -> None:
    logger.exception(message, extra=_extra(event=event, fields=fields))


def _extra(*, event: str, fields: EventFields) -> EventFields:
    return {
        "event": event,
        **{key: value for key, value in fields.items() if value is not None},
    }
