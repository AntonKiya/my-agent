import asyncio
from dataclasses import dataclass, field
from uuid import UUID

from agent_service.feedback.interfaces import FeedbackStateStore
from agent_service.feedback.models import PendingFeedback


@dataclass(slots=True)
class InMemoryFeedbackStateStore(FeedbackStateStore):
    _pending: dict[UUID, PendingFeedback] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get_pending(self, *, conversation_id: UUID) -> PendingFeedback | None:
        async with self._lock:
            return self._pending.get(conversation_id)

    async def set_pending(self, *, pending: PendingFeedback) -> None:
        async with self._lock:
            self._pending[pending.conversation_id] = pending

    async def clear_pending(self, *, conversation_id: UUID) -> None:
        async with self._lock:
            self._pending.pop(conversation_id, None)
