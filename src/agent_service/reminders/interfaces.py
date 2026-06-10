from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from agent_service.reminders.models import (
    NotificationOutboxItem,
    Reminder,
    ReminderEventStatus,
    ReminderStatus,
)


class ReminderStore(Protocol):
    async def create(self, reminder: Reminder) -> Reminder:
        """Persist a reminder."""
        ...

    async def list_for_user(
        self,
        *,
        user_id: UUID,
        statuses: Sequence[ReminderStatus] | None = None,
        limit: int = 20,
    ) -> tuple[Reminder, ...]:
        """List reminders owned by a user."""
        ...

    async def mark_deleted(self, *, reminder_id: UUID, user_id: UUID) -> bool:
        """Cancel a user's reminder."""
        ...

    async def count_active_for_user(self, *, user_id: UUID) -> int:
        """Count active reminders owned by a user."""
        ...

    async def count_created_for_source_event(self, *, source_inbound_event_id: UUID) -> int:
        """Count reminders created from a single inbound event."""
        ...

    async def process_due_reminders(
        self,
        *,
        now: datetime,
        batch_size: int,
    ) -> int:
        """Queue due reminders and advance their next_fire_at_utc values."""
        ...


class NotificationOutboxStore(Protocol):
    async def claim_due(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
        limit: int,
    ) -> tuple[NotificationOutboxItem, ...]:
        """Lease due outbox rows for delivery."""
        ...

    async def mark_sent(
        self,
        *,
        item_id: UUID,
        reminder_event_id: UUID | None,
        sent_at: datetime,
    ) -> None:
        """Mark an outbox item as delivered."""
        ...

    async def mark_retry(
        self,
        *,
        item_id: UUID,
        available_at: datetime,
        error: str | None = None,
    ) -> None:
        """Release an outbox item for a later retry."""
        ...

    async def mark_dead_letter(
        self,
        *,
        item_id: UUID,
        reminder_event_id: UUID | None,
        error: str | None = None,
    ) -> None:
        """Mark an outbox item as permanently failed."""
        ...

    async def mark_reminder_event_status(
        self,
        *,
        reminder_event_id: UUID,
        status: ReminderEventStatus,
        error: str | None = None,
        sent_at: datetime | None = None,
    ) -> None:
        """Update the reminder event linked to an outbox item."""
        ...
