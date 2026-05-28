from contextvars import ContextVar, Token
from uuid import uuid4

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def create_trace_id() -> str:
    return uuid4().hex


def get_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(trace_id: str) -> Token[str | None]:
    return _trace_id.set(trace_id)


def reset_trace_id(token: Token[str | None]) -> None:
    _trace_id.reset(token)
