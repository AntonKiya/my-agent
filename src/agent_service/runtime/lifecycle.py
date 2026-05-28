import asyncio
import contextlib
import logging
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskSupervisor:
    # Owns background task cancellation and shutdown timing only.
    # Business retries, dead letters, and delivery semantics belong to workers/services.
    shutdown_timeout_seconds: float
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _stopping: bool = field(default=False, init=False, repr=False)

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def create_task(
        self,
        coro: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> asyncio.Task[None]:
        if self._stopping:
            raise RuntimeError("Cannot create background tasks while supervisor is stopping")

        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    async def stop(self) -> None:
        self._stopping = True
        tasks = tuple(self._tasks)
        if not tasks:
            return

        for task in tasks:
            if not task.done():
                task.cancel()

        _done, pending = await asyncio.wait(tasks, timeout=self.shutdown_timeout_seconds)
        if pending:
            logger.warning(
                "Background tasks did not stop before timeout",
                extra={
                    "event": "background_tasks_shutdown_timeout",
                    "pending_task_count": len(pending),
                    "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
                },
            )

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._consume_task_result(task)

    def _consume_task_result(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return

        with contextlib.suppress(asyncio.InvalidStateError):
            exception = task.exception()
            if exception is not None:
                logger.exception(
                    "Background task failed",
                    extra={
                        "event": "background_task_failed",
                        "task_name": task.get_name(),
                    },
                    exc_info=(type(exception), exception, exception.__traceback__),
                )
