from dataclasses import dataclass, field

from agent_service.config import AppSettings
from agent_service.runtime.lifecycle import TaskSupervisor


@dataclass(slots=True)
class AppContainer:
    # Central assembly point for infrastructure dependencies and their lifecycle.
    # It should not contain message processing, agent logic, or channel behavior.
    settings: AppSettings
    task_supervisor: TaskSupervisor = field(init=False)
    _started: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.task_supervisor = TaskSupervisor(
            shutdown_timeout_seconds=self.settings.graceful_shutdown_timeout_seconds,
        )

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        # The container owns infrastructure lifecycle, not business processing.
        self._started = True

    async def stop(self) -> None:
        # Future queues, DB/Redis clients, and workers should be stopped from here.
        await self.task_supervisor.stop()
        self._started = False
