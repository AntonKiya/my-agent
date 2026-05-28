import asyncio
from dataclasses import dataclass, field

from agent_service.channels.models import InboundEvent, OutboundEvent
from agent_service.messaging.interfaces import EventQueue, InboundQueue, OutboundQueue


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

    async def publish(self, event: QueueEventT) -> None:
        await self._queue.put(event)

    async def consume(self) -> QueueEventT:
        return await self._queue.get()


class AsyncioInboundQueue(AsyncioEventQueue[InboundEvent], InboundQueue):
    pass


class AsyncioOutboundQueue(AsyncioEventQueue[OutboundEvent], OutboundQueue):
    pass
