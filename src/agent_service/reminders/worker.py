import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agent_service.conversations import ConversationLockManager
from agent_service.delivery import (
    DeliveryAdapterRegistry,
    DeliveryResult,
    DeliveryRetryPolicy,
    DeliveryStatus,
)
from agent_service.observability.events import elapsed_ms, log_event, start_timer
from agent_service.outbound import OutboundEvent
from agent_service.reminders.interfaces import NotificationOutboxStore, ReminderStore
from agent_service.reminders.models import NotificationOutboxItem

logger = logging.getLogger(__name__)
SleepCallable = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class ReminderWorker:
    reminder_store: ReminderStore
    poll_interval_seconds: float = 30.0
    batch_size: int = 500
    error_backoff_seconds: float = 1.0
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    async def run_forever(self) -> None:
        while True:
            try:
                processed = await self.process_once()
                if processed == 0:
                    await self.sleep(self.poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Reminder worker iteration failed",
                    extra={"event": "reminder_worker_iteration_failed"},
                )
                await self.sleep(self.error_backoff_seconds)

    async def process_once(self) -> int:
        started_at = start_timer()
        processed = await self.reminder_store.process_due_reminders(
            now=datetime.now(UTC),
            batch_size=self.batch_size,
        )
        if processed:
            log_event(
                logger,
                logging.INFO,
                "Due reminders processed",
                event="reminders_due_processed",
                processed_count=processed,
                duration_ms=elapsed_ms(started_at),
            )
        return processed


@dataclass(slots=True)
class NotificationOutboxDeliveryWorker:
    outbox_store: NotificationOutboxStore
    channel_adapters: DeliveryAdapterRegistry
    lock_manager: ConversationLockManager
    retry_policy: DeliveryRetryPolicy = field(default_factory=DeliveryRetryPolicy)
    worker_id: str = field(default_factory=lambda: f"notification-outbox-{uuid4()}")
    poll_interval_seconds: float = 1.0
    lease_seconds: float = 60.0
    batch_size: int = 100
    error_backoff_seconds: float = 1.0
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    async def run_forever(self) -> None:
        while True:
            try:
                processed = await self.process_once()
                if processed == 0:
                    await self.sleep(self.poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Notification outbox worker iteration failed",
                    extra={"event": "notification_outbox_worker_iteration_failed"},
                )
                await self.sleep(self.error_backoff_seconds)

    async def process_once(self) -> int:
        items = await self.outbox_store.claim_due(
            worker_id=self.worker_id,
            now=datetime.now(UTC),
            lease_seconds=self.lease_seconds,
            limit=self.batch_size,
        )
        for item in items:
            await self._process_item(item)
        return len(items)

    async def _process_item(self, item: NotificationOutboxItem) -> None:
        outbox_item = item
        event = _outbound_event_from_item(outbox_item)
        started_at = start_timer()
        try:
            adapter = self.channel_adapters.get(event.channel)
            async with self.lock_manager.acquire(event.conversation_id):
                result = await adapter.send(event)
        except Exception as exc:
            attempt_number = outbox_item.retry_count + 1
            await self._retry_or_dead_letter(
                outbox_item,
                attempt_number=attempt_number,
                error=str(exc) or type(exc).__name__,
                retry_after_seconds=None,
            )
            return

        log_event(
            logger,
            logging.INFO if result.status is DeliveryStatus.SENT else logging.WARNING,
            "Notification outbox delivery attempt completed",
            event="notification_outbox_delivery_attempt_completed",
            outbox_item_id=str(outbox_item.id),
            reminder_event_id=(
                str(outbox_item.reminder_event_id)
                if outbox_item.reminder_event_id is not None
                else None
            ),
            channel=event.channel,
            status=result.status.value,
            duration_ms=elapsed_ms(started_at),
            error_code=result.error_code,
        )
        if result.status is DeliveryStatus.SENT:
            await self.outbox_store.mark_sent(
                item_id=outbox_item.id,
                reminder_event_id=outbox_item.reminder_event_id,
                sent_at=datetime.now(UTC),
            )
            return
        if result.status is DeliveryStatus.DEAD_LETTER:
            await self.outbox_store.mark_dead_letter(
                item_id=outbox_item.id,
                reminder_event_id=outbox_item.reminder_event_id,
                error=result.error_message or result.error_code,
            )
            return

        await self._retry_or_dead_letter(
            outbox_item,
            attempt_number=outbox_item.retry_count + 1,
            error=result.error_message or result.error_code,
            retry_after_seconds=result.retry_after_seconds,
        )

    async def _retry_or_dead_letter(
        self,
        item: NotificationOutboxItem,
        *,
        attempt_number: int,
        error: str | None,
        retry_after_seconds: float | None,
    ) -> None:
        if attempt_number >= self.retry_policy.max_attempts:
            await self.outbox_store.mark_dead_letter(
                item_id=item.id,
                reminder_event_id=item.reminder_event_id,
                error=error or "delivery retry attempts exhausted",
            )
            return
        delay_seconds = (
            retry_after_seconds
            if retry_after_seconds is not None
            else self.retry_policy.delay_for_attempt(
                attempt_number,
                DeliveryResult(
                    event_id=item.id,
                    channel=item.channel,
                    status=DeliveryStatus.FAILED_RETRYABLE,
                ),
            )
        )
        await self.outbox_store.mark_retry(
            item_id=item.id,
            available_at=datetime.now(UTC) + timedelta(seconds=delay_seconds),
            error=error,
        )


def _outbound_event_from_item(item: NotificationOutboxItem) -> OutboundEvent:
    payload = item.payload
    text = payload.get("text")
    metadata = payload.get("metadata")
    return OutboundEvent(
        event_id=item.id,
        channel=item.channel,
        user_id=item.user_id,
        conversation_id=item.source_conversation_id,
        external_chat_id=item.external_chat_id,
        text=text if isinstance(text, str) else "",
        thread_id=item.thread_id,
        metadata=metadata if isinstance(metadata, dict) else {},
    )
