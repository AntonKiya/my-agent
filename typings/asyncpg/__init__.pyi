from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any

class UniqueViolationError(Exception): ...


class Connection:
    async def execute(self, query: str, *args: object) -> str: ...

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]: ...

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None: ...

    def transaction(self) -> AbstractAsyncContextManager[object]: ...

    async def close(self) -> None: ...


class PoolAcquireContext:
    async def __aenter__(self) -> Connection: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class Pool:
    def acquire(self, *, timeout: float | None = None) -> PoolAcquireContext: ...

    async def close(self) -> None: ...


async def connect(
    *,
    dsn: str | None = None,
    command_timeout: float | None = None,
    **kwargs: Any,
) -> Connection: ...


def create_pool(
    *,
    dsn: str | None = None,
    min_size: int = 10,
    max_size: int = 10,
    command_timeout: float | None = None,
    **kwargs: Any,
) -> Pool: ...
