import asyncio
from dataclasses import dataclass, field

from agent_service.channels.models import InboundEvent
from agent_service.memory.models import ConversationCompactionJob
from agent_service.messaging.base import EventQueue, QueueStats
from agent_service.messaging.interfaces import (
    CompactionQueue,
    InboundQueue,
)
from agent_service.outbound import OutboundEvent, OutboundQueue


@dataclass(slots=True)
class AsyncioEventQueue[QueueEventT](EventQueue[QueueEventT]):
    maxsize: int = 0
    _queue: asyncio.Queue[QueueEventT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.maxsize < 0:
            raise ValueError("Queue maxsize must be greater than or equal to zero")
        self._queue = asyncio.Queue(maxsize=self.maxsize)

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    @property
    def stats(self) -> QueueStats:
        return QueueStats(
            size=self.size,
            maxsize=self.maxsize,
            is_empty=self.is_empty,
            is_full=self.is_full,
        )

    async def publish(self, event: QueueEventT) -> None:
        await self._queue.put(event)

    async def consume(self) -> QueueEventT:
        return await self._queue.get()

    async def acknowledge(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


class AsyncioInboundQueue(AsyncioEventQueue[InboundEvent], InboundQueue):
    pass


class AsyncioOutboundQueue(AsyncioEventQueue[OutboundEvent], OutboundQueue):
    pass


class AsyncioCompactionQueue(AsyncioEventQueue[ConversationCompactionJob], CompactionQueue):
    pass
