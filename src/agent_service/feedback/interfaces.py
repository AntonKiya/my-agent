from typing import Protocol, runtime_checkable
from uuid import UUID

from agent_service.feedback.models import FeedbackEntry, PendingFeedback


@runtime_checkable
class FeedbackStore(Protocol):
    async def create(self, *, feedback: FeedbackEntry) -> FeedbackEntry:
        """Persist user feedback."""
        ...


@runtime_checkable
class FeedbackStateStore(Protocol):
    async def get_pending(self, *, conversation_id: UUID) -> PendingFeedback | None:
        """Return pending feedback state for a conversation, if any."""
        ...

    async def set_pending(self, *, pending: PendingFeedback) -> None:
        """Mark the next text message in the conversation as feedback."""
        ...

    async def clear_pending(self, *, conversation_id: UUID) -> None:
        """Clear pending feedback state for a conversation."""
        ...
