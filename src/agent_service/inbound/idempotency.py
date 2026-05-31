from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from agent_service.channels.models import InboundEvent, InboundEventStatus


@dataclass(frozen=True, slots=True)
class InboundIdempotencyClaim:
    claimed: bool
    event_id: UUID
    existing_event_id: UUID | None = None
    existing_status: InboundEventStatus | None = None


class InboundIdempotencyStore(Protocol):
    async def claim(self, event: InboundEvent) -> InboundIdempotencyClaim:
        """Atomically claim one transport-derived inbound event before queue publish."""
        ...

    async def release_claim(self, *, event_id: UUID) -> None:
        """Release a queued claim when the event was not published to the queue."""
        ...

    async def mark_status(
        self,
        *,
        event_id: UUID,
        status: InboundEventStatus,
        failure_reason: str | None = None,
    ) -> None:
        """Persist the latest processing status for an already claimed event."""
        ...
