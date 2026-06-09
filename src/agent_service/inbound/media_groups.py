import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import uuid4

from agent_service.channels.models import InboundEvent, MessageType
from agent_service.inbound.idempotency import InboundIdempotencyStore
from agent_service.messaging.interfaces import InboundQueue
from agent_service.observability.events import (
    elapsed_ms,
    log_event,
    start_timer,
    store_current_trace_context,
)

logger = logging.getLogger(__name__)


class MediaGroupBufferError(RuntimeError):
    """Raised when an inbound media group cannot be buffered reliably."""


class MediaGroupAddStatus(StrEnum):
    BUFFERED = "buffered"
    DUPLICATE = "duplicate"
    FINALIZED = "finalized"


@dataclass(frozen=True, slots=True)
class MediaGroupAddResult:
    status: MediaGroupAddStatus
    group_key: str
    item_count: int | None = None


@dataclass(frozen=True, slots=True)
class LeasedMediaGroup:
    group_key: str
    lock_key: str
    lock_token: str
    events: tuple[InboundEvent, ...]
    aggregate_event: InboundEvent


@runtime_checkable
class RedisMediaGroupClient(Protocol):
    async def get(self, name: str) -> bytes | str | None:
        """Return a Redis string value."""
        ...

    async def set(
        self,
        name: str,
        value: str,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> object:
        """Set a Redis string value."""
        ...

    async def delete(self, *names: str) -> object:
        """Delete Redis keys."""
        ...

    def scan_iter(
        self,
        match: str | None = None,
        count: int | None = None,
    ) -> AsyncIterator[bytes | str]:
        """Iterate Redis keys matching a pattern."""
        ...


class RedisInboundMediaGroupAggregator:
    def __init__(
        self,
        client: RedisMediaGroupClient,
        *,
        debounce_seconds: float = 2.0,
        ttl_seconds: int = 60,
        lock_ttl_seconds: float = 10.0,
    ) -> None:
        if debounce_seconds <= 0:
            raise ValueError("Media group debounce must be greater than zero")
        if ttl_seconds <= 0:
            raise ValueError("Media group TTL must be greater than zero")
        if lock_ttl_seconds <= 0:
            raise ValueError("Media group lock TTL must be greater than zero")
        self._client = client
        self._debounce_seconds = debounce_seconds
        self._ttl_seconds = ttl_seconds
        self._lock_ttl_ms = int(lock_ttl_seconds * 1000)

    async def add(self, event: InboundEvent) -> MediaGroupAddResult:
        group_key = media_group_key(event)
        if group_key is None:
            raise MediaGroupBufferError("Inbound event does not belong to a media group")

        finalized_key = self._finalized_key(group_key)
        if await self._client.get(finalized_key) is not None:
            return MediaGroupAddResult(status=MediaGroupAddStatus.FINALIZED, group_key=group_key)

        lock_key = self._lock_key(group_key, purpose="add")
        lock_token = await self._acquire_lock(lock_key)
        if lock_token is None:
            raise MediaGroupBufferError("Timed out acquiring media group buffer lock")

        started_at = start_timer()
        try:
            key = self._group_key(group_key)
            payload = await self._read_group(key) or {
                "due_at": 0.0,
                "events": [],
            }
            events = _payload_events(payload)
            if any(item.idempotency_key == event.idempotency_key for item in events):
                return MediaGroupAddResult(
                    status=MediaGroupAddStatus.DUPLICATE,
                    group_key=group_key,
                    item_count=len(events),
                )

            events.append(event)
            payload = {
                "due_at": time.time() + self._debounce_seconds,
                "events": [item.model_dump(mode="json") for item in events],
            }
            await self._client.set(key, json.dumps(payload), ex=self._ttl_seconds)
            log_event(
                logger,
                logging.INFO,
                "Inbound media group item buffered",
                event="inbound_media_group_item_buffered",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                group_key=group_key,
                media_group_id=event.channel_metadata.get("media_group_id"),
                item_count=len(events),
                duration_ms=elapsed_ms(started_at),
            )
            return MediaGroupAddResult(
                status=MediaGroupAddStatus.BUFFERED,
                group_key=group_key,
                item_count=len(events),
            )
        finally:
            await self._client.delete(lock_key)

    async def due_group_keys(self) -> list[str]:
        pattern = f"{self._group_key('*')}"
        now = time.time()
        due_keys: list[str] = []
        async for raw_key in self._client.scan_iter(match=pattern, count=100):
            redis_key = _decode_redis_value(raw_key)
            payload = await self._read_group(redis_key)
            if payload is None:
                continue
            due_at = payload.get("due_at")
            if isinstance(due_at, int | float) and due_at <= now:
                due_keys.append(redis_key.removeprefix("inbound:media_group:group:"))
        return due_keys

    async def lease_due_group(self, group_key: str) -> LeasedMediaGroup | None:
        lock_key = self._lock_key(group_key, purpose="flush")
        lock_token = await self._acquire_lock(lock_key, wait_seconds=0)
        if lock_token is None:
            return None

        try:
            payload = await self._read_group(self._group_key(group_key))
            if payload is None:
                await self._client.delete(lock_key)
                return None
            due_at = payload.get("due_at")
            if not isinstance(due_at, int | float) or due_at > time.time():
                await self._client.delete(lock_key)
                return None
            events = tuple(_sort_events(_payload_events(payload)))
            if not events:
                await self._client.delete(self._group_key(group_key), lock_key)
                return None
            return LeasedMediaGroup(
                group_key=group_key,
                lock_key=lock_key,
                lock_token=lock_token,
                events=events,
                aggregate_event=_aggregate_events(events),
            )
        except BaseException:
            await self._client.delete(lock_key)
            raise

    async def mark_published(self, group: LeasedMediaGroup) -> None:
        await self._client.set(
            self._finalized_key(group.group_key),
            "1",
            ex=self._ttl_seconds,
        )
        await self._client.delete(self._group_key(group.group_key), group.lock_key)

    async def reschedule(self, group: LeasedMediaGroup, *, delay_seconds: float) -> None:
        payload = {
            "due_at": time.time() + max(delay_seconds, self._debounce_seconds),
            "events": [event.model_dump(mode="json") for event in group.events],
        }
        await self._client.set(
            self._group_key(group.group_key),
            json.dumps(payload),
            ex=self._ttl_seconds,
        )
        await self._client.delete(group.lock_key)

    async def release(self, group: LeasedMediaGroup) -> None:
        await self._client.delete(group.lock_key)

    async def _acquire_lock(
        self,
        lock_key: str,
        *,
        wait_seconds: float = 0.5,
    ) -> str | None:
        deadline = time.monotonic() + wait_seconds
        token = uuid4().hex
        while True:
            result = await self._client.set(
                lock_key,
                token,
                px=self._lock_ttl_ms,
                nx=True,
            )
            if result:
                return token
            if wait_seconds <= 0 or time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.01)

    async def _read_group(self, redis_key: str) -> dict[str, object] | None:
        raw = await self._client.get(redis_key)
        if raw is None:
            return None
        try:
            payload = json.loads(_decode_redis_value(raw))
        except json.JSONDecodeError as exc:
            raise MediaGroupBufferError("Media group buffer payload is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise MediaGroupBufferError("Media group buffer payload must be an object")
        return payload

    @staticmethod
    def _group_key(group_key: str) -> str:
        return f"inbound:media_group:group:{group_key}"

    @staticmethod
    def _finalized_key(group_key: str) -> str:
        return f"inbound:media_group:finalized:{group_key}"

    @staticmethod
    def _lock_key(group_key: str, *, purpose: str) -> str:
        return f"inbound:media_group:lock:{purpose}:{group_key}"


def media_group_key(event: InboundEvent) -> str | None:
    media_group_id = event.channel_metadata.get("media_group_id")
    if not isinstance(media_group_id, str) or not media_group_id:
        return None
    if event.user_id is None:
        raise MediaGroupBufferError("Media group event must be user-resolved before buffering")
    return ":".join(
        [
            event.channel,
            str(event.user_id),
            event.external_chat_id,
            media_group_id,
        ]
    )


def _payload_events(payload: dict[str, object]) -> list[InboundEvent]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise MediaGroupBufferError("Media group buffer payload must include events list")
    return [InboundEvent.model_validate(item) for item in raw_events]


def _aggregate_events(events: tuple[InboundEvent, ...]) -> InboundEvent:
    first = events[0]
    text = next((event.text for event in events if event.text), None)
    media_group_id = first.channel_metadata.get("media_group_id")
    attachments = [attachment for event in events for attachment in event.attachments]
    source_event_ids = [str(event.event_id) for event in events]
    source_update_ids = [
        event.external_update_id for event in events if event.external_update_id is not None
    ]
    source_message_ids = [
        event.external_message_id for event in events if event.external_message_id is not None
    ]
    return first.model_copy(
        deep=True,
        update={
            "event_id": uuid4(),
            "external_message_id": source_message_ids[0] if source_message_ids else None,
            "external_update_id": source_update_ids[0] if source_update_ids else None,
            "idempotency_key": (
                f"{first.channel}:{first.external_chat_id}:media_group:{media_group_id}"
            ),
            "message_type": MessageType.MIXED if text else MessageType.MEDIA,
            "text": text,
            "attachments": attachments,
            "channel_metadata": {
                **first.channel_metadata,
                "media_group_aggregated": True,
                "source_event_ids": source_event_ids,
                "source_update_ids": source_update_ids,
                "source_message_ids": source_message_ids,
            },
            "metadata": {
                **first.metadata,
                "media_group": {
                    "aggregated": True,
                    "source_event_ids": source_event_ids,
                    "source_update_ids": source_update_ids,
                    "source_message_ids": source_message_ids,
                },
            },
            "trace_id": first.trace_id,
            "received_at": min(event.received_at for event in events),
        },
    )


def _sort_events(events: list[InboundEvent]) -> list[InboundEvent]:
    return sorted(events, key=_event_order_key)


def _event_order_key(event: InboundEvent) -> tuple[int, int, str]:
    message_id = _optional_int(event.external_message_id)
    update_id = _optional_int(event.external_update_id)
    return (
        message_id if message_id is not None else 2**63 - 1,
        update_id if update_id is not None else 2**63 - 1,
        str(event.event_id),
    )


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _decode_redis_value(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class InboundMediaGroupFlushWorker:
    def __init__(
        self,
        *,
        aggregator: RedisInboundMediaGroupAggregator,
        inbound_queue: InboundQueue,
        idempotency_store: InboundIdempotencyStore | None = None,
        publish_timeout_seconds: float | None = None,
        flush_interval_seconds: float = 0.5,
        error_backoff_seconds: float = 0.5,
    ) -> None:
        if publish_timeout_seconds is not None and publish_timeout_seconds <= 0:
            raise ValueError("Media group publish timeout must be greater than zero")
        if flush_interval_seconds <= 0:
            raise ValueError("Media group flush interval must be greater than zero")
        if error_backoff_seconds < 0:
            raise ValueError("Media group error backoff must be greater than or equal to zero")
        self.aggregator = aggregator
        self.inbound_queue = inbound_queue
        self.idempotency_store = idempotency_store
        self.publish_timeout_seconds = publish_timeout_seconds
        self.flush_interval_seconds = flush_interval_seconds
        self.error_backoff_seconds = error_backoff_seconds

    async def run_forever(self) -> None:
        while True:
            try:
                await self.flush_once()
                await asyncio.sleep(self.flush_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    logger,
                    logging.ERROR,
                    "Inbound media group flush worker failed",
                    event="inbound_media_group_flush_worker_failed",
                    error_type=type(exc).__name__,
                )
                await asyncio.sleep(self.error_backoff_seconds)

    async def flush_once(self) -> int:
        published_count = 0
        for group_key in await self.aggregator.due_group_keys():
            group = await self.aggregator.lease_due_group(group_key)
            if group is None:
                continue
            if await self._flush_group(group):
                published_count += 1
        return published_count

    async def _flush_group(self, group: LeasedMediaGroup) -> bool:
        started_at = start_timer()
        event = group.aggregate_event
        log_event(
            logger,
            logging.INFO,
            "Inbound media group flush started",
            event="inbound_media_group_flush_started",
            inbound_event_id=str(event.event_id),
            channel=event.channel,
            user_id=str(event.user_id) if event.user_id is not None else None,
            group_key=group.group_key,
            media_group_id=event.channel_metadata.get("media_group_id"),
            item_count=len(group.events),
        )
        claim_acquired = False
        publish_succeeded = False
        try:
            if self.idempotency_store is not None:
                claim = await self.idempotency_store.claim(event)
                if not claim.claimed:
                    await self.aggregator.mark_published(group)
                    log_event(
                        logger,
                        logging.INFO,
                        "Duplicate inbound media group suppressed",
                        event="inbound_media_group_duplicate_suppressed",
                        inbound_event_id=str(event.event_id),
                        existing_inbound_event_id=(
                            str(claim.existing_event_id)
                            if claim.existing_event_id is not None
                            else None
                        ),
                        channel=event.channel,
                        user_id=str(event.user_id) if event.user_id is not None else None,
                        group_key=group.group_key,
                        duration_ms=elapsed_ms(started_at),
                    )
                    return False
                claim_acquired = True

            published = await self._publish_event(event)
            if not published:
                if self.idempotency_store is not None:
                    await self.idempotency_store.release_claim(event_id=event.event_id)
                await self.aggregator.reschedule(
                    group,
                    delay_seconds=self.flush_interval_seconds,
                )
                queue_stats = self.inbound_queue.stats
                log_event(
                    logger,
                    logging.WARNING,
                    "Inbound media group publish timed out",
                    event="inbound_media_group_publish_timed_out",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                    group_key=group.group_key,
                    queue_size=queue_stats.size,
                    queue_maxsize=queue_stats.maxsize,
                    publish_timeout_seconds=self.publish_timeout_seconds,
                    duration_ms=elapsed_ms(started_at),
                )
                return False
            publish_succeeded = True

            await self.aggregator.mark_published(group)
            queue_stats = self.inbound_queue.stats
            log_event(
                logger,
                logging.INFO,
                "Inbound media group published",
                event="inbound_media_group_published",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                group_key=group.group_key,
                media_group_id=event.channel_metadata.get("media_group_id"),
                item_count=len(group.events),
                queue_size=queue_stats.size,
                queue_maxsize=queue_stats.maxsize,
                duration_ms=elapsed_ms(started_at),
            )
            return True
        except BaseException:
            if claim_acquired and not publish_succeeded and self.idempotency_store is not None:
                await self.idempotency_store.release_claim(event_id=event.event_id)
            await self.aggregator.release(group)
            raise

    async def _publish_event(self, event: InboundEvent) -> bool:
        store_current_trace_context(event.metadata)
        if self.publish_timeout_seconds is None:
            await self.inbound_queue.publish(event)
            return True
        try:
            await asyncio.wait_for(
                self.inbound_queue.publish(event),
                timeout=self.publish_timeout_seconds,
            )
        except TimeoutError:
            return False
        return True
