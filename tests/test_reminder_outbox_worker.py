from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from agent_service.conversations import AsyncioConversationLockManager
from agent_service.delivery import DeliveryAdapter, DeliveryResult, DeliveryStatus
from agent_service.outbound import OutboundEvent
from agent_service.reminders import NotificationOutboxItem
from agent_service.reminders.postgres import CLAIM_OUTBOX_SQL
from agent_service.reminders.worker import NotificationOutboxDeliveryWorker


@dataclass(slots=True)
class FakeOutboxStore:
    items: list[NotificationOutboxItem] = field(default_factory=list)
    sent: list[UUID] = field(default_factory=list)
    retries: list[UUID] = field(default_factory=list)
    dead_letters: list[UUID] = field(default_factory=list)

    async def claim_due(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
        limit: int,
    ) -> tuple[NotificationOutboxItem, ...]:
        del worker_id, now, lease_seconds
        claimed = tuple(self.items[:limit])
        self.items = self.items[limit:]
        return claimed

    async def mark_sent(
        self,
        *,
        item_id: UUID,
        reminder_event_id: UUID | None,
        sent_at: datetime,
    ) -> None:
        del reminder_event_id, sent_at
        self.sent.append(item_id)

    async def mark_retry(
        self,
        *,
        item_id: UUID,
        available_at: datetime,
        error: str | None = None,
    ) -> None:
        del available_at, error
        self.retries.append(item_id)

    async def mark_dead_letter(
        self,
        *,
        item_id: UUID,
        reminder_event_id: UUID | None,
        error: str | None = None,
    ) -> None:
        del reminder_event_id, error
        self.dead_letters.append(item_id)

    async def mark_reminder_event_status(
        self,
        *,
        reminder_event_id: UUID,
        status: object,
        error: str | None = None,
        sent_at: datetime | None = None,
    ) -> None:
        del reminder_event_id, status, error, sent_at


@dataclass(slots=True)
class FakeAdapterRegistry:
    adapter: "FakeAdapter"

    def get(self, channel: str) -> DeliveryAdapter:
        assert channel == "telegram"
        return cast(DeliveryAdapter, self.adapter)


@dataclass(slots=True)
class FakeAdapter:
    channel: str = "telegram"
    status: DeliveryStatus = DeliveryStatus.SENT
    events: list[OutboundEvent] = field(default_factory=list)

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        self.events.append(event)
        return DeliveryResult(
            event_id=event.event_id,
            channel=event.channel,
            status=self.status,
            error_code=None if self.status is DeliveryStatus.SENT else "failed",
        )


def outbox_item() -> NotificationOutboxItem:
    return NotificationOutboxItem(
        id=uuid4(),
        user_id=uuid4(),
        source_conversation_id=uuid4(),
        reminder_event_id=uuid4(),
        channel="telegram",
        external_chat_id="123",
        thread_id="456",
        payload={"text": "Напоминание: выпить витамины", "metadata": {"source": "reminder"}},
        available_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


async def test_notification_outbox_worker_delivers_and_marks_sent() -> None:
    item = outbox_item()
    store = FakeOutboxStore(items=[item])
    adapter = FakeAdapter()
    worker = NotificationOutboxDeliveryWorker(
        outbox_store=store,
        channel_adapters=FakeAdapterRegistry(adapter),
        lock_manager=AsyncioConversationLockManager(),
    )

    processed = await worker.process_once()

    assert processed == 1
    assert store.sent == [item.id]
    assert adapter.events[0].external_chat_id == "123"
    assert adapter.events[0].thread_id == "456"
    assert adapter.events[0].text == "Напоминание: выпить витамины"


async def test_notification_outbox_worker_retries_retryable_failures() -> None:
    item = outbox_item()
    store = FakeOutboxStore(items=[item])
    adapter = FakeAdapter(status=DeliveryStatus.FAILED_RETRYABLE)
    worker = NotificationOutboxDeliveryWorker(
        outbox_store=store,
        channel_adapters=FakeAdapterRegistry(adapter),
        lock_manager=AsyncioConversationLockManager(),
    )

    processed = await worker.process_once()

    assert processed == 1
    assert store.retries == [item.id]
    assert store.dead_letters == []


def test_outbox_claim_sql_reclaims_expired_sending_leases() -> None:
    assert "status = 'pending'" in CLAIM_OUTBOX_SQL
    assert "status = 'sending'" in CLAIM_OUTBOX_SQL
    assert "locked_until <= $3" in CLAIM_OUTBOX_SQL
