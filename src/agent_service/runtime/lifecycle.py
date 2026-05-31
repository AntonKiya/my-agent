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
    _task_groups: dict[asyncio.Task[None], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _stopping: bool = field(default=False, init=False, repr=False)

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def task_count_for_group(self, group: str) -> int:
        return sum(1 for task in self._tasks if self._task_groups.get(task) == group)

    def create_task(
        self,
        coro: Coroutine[Any, Any, None],
        *,
        name: str,
        group: str = "default",
    ) -> asyncio.Task[None]:
        if self._stopping:
            raise RuntimeError("Cannot create background tasks while supervisor is stopping")

        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        self._task_groups[task] = group
        task.add_done_callback(self._on_task_done)
        return task

    async def stop(self, *, group: str | None = None) -> None:
        self._stopping = True
        tasks = self._tasks_for_group(group)
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
                    "task_group": group,
                    "pending_task_count": len(pending),
                    "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
                },
            )

    def _tasks_for_group(self, group: str | None) -> tuple[asyncio.Task[None], ...]:
        if group is None:
            return tuple(self._tasks)
        return tuple(task for task in self._tasks if self._task_groups.get(task) == group)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._task_groups.pop(task, None)
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
