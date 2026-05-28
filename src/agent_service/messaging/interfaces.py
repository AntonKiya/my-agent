from typing import Protocol, TypeVar, runtime_checkable

from agent_service.channels.models import InboundEvent, OutboundEvent

QueueEventT = TypeVar("QueueEventT")


@runtime_checkable
class EventQueue(Protocol[QueueEventT]):
    async def publish(self, event: QueueEventT) -> None:
        """Publish an event without exposing the queue backend to callers."""

    async def consume(self) -> QueueEventT:
        """Consume the next available event from the queue backend."""


@runtime_checkable
class InboundQueue(EventQueue[InboundEvent], Protocol):
    pass


@runtime_checkable
class OutboundQueue(EventQueue[OutboundEvent], Protocol):
    pass
