from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

QueueEventT = TypeVar("QueueEventT")


class QueueStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int = Field(ge=0)
    maxsize: int = Field(ge=0)
    is_empty: bool
    is_full: bool


@runtime_checkable
class EventQueue(Protocol[QueueEventT]):
    @property
    def stats(self) -> QueueStats:
        """Return backend-neutral queue pressure details."""
        ...

    async def publish(self, event: QueueEventT) -> None:
        """Publish an event without exposing the queue backend to callers."""
        ...

    async def consume(self) -> QueueEventT:
        """Consume the next available event from the queue backend."""
        ...

    async def acknowledge(self) -> None:
        """Mark the most recently consumed event as processed."""
        ...

    async def join(self) -> None:
        """Wait until all published events have been acknowledged."""
        ...
