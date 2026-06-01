import logging
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager, nullcontext
from importlib import import_module
from time import perf_counter
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.propagate import extract, inject

EventFields = dict[str, Any]
TRACE_CONTEXT_METADATA_KEY = "otel_trace_context"
_business_spans_enabled = False


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


def enable_business_spans() -> None:
    global _business_spans_enabled
    _business_spans_enabled = True


def disable_business_spans_for_tests() -> None:
    global _business_spans_enabled
    _business_spans_enabled = False


def business_span(message: str, *, event: str, **fields: object) -> Any:
    if not _business_spans_enabled:
        return nullcontext()
    try:
        logfire = import_module("logfire")
    except Exception:
        return nullcontext()
    return logfire.span(
        message,
        _span_name=event,
        event=event,
        **_safe_fields(fields),
    )


def capture_trace_context() -> dict[str, str]:
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


@contextmanager
def attached_trace_context(metadata: Mapping[str, object]) -> Iterator[None]:
    carrier = metadata.get(TRACE_CONTEXT_METADATA_KEY)
    if not isinstance(carrier, Mapping):
        yield
        return

    trace_carrier = {
        str(key): str(value)
        for key, value in carrier.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    if not trace_carrier:
        yield
        return

    token = otel_context.attach(extract(trace_carrier))
    try:
        yield
    finally:
        otel_context.detach(token)


def store_current_trace_context(metadata: MutableMapping[str, object]) -> None:
    carrier = capture_trace_context()
    if carrier:
        metadata[TRACE_CONTEXT_METADATA_KEY] = carrier


def _extra(*, event: str, fields: EventFields) -> EventFields:
    return {
        "event": event,
        **_safe_fields(fields),
    }


def _safe_fields(fields: Mapping[str, object]) -> EventFields:
    return {
        key: value
        for key, value in fields.items()
        if value is not None and key != TRACE_CONTEXT_METADATA_KEY
    }
