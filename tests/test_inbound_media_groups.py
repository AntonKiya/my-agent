import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from agent_service.channels.models import Attachment, AttachmentType, InboundEvent, MessageType
from agent_service.inbound import InboundIdempotencyClaim
from agent_service.inbound.media_groups import (
    InboundMediaGroupFlushWorker,
    RedisInboundMediaGroupAggregator,
)
from agent_service.messaging.in_memory import AsyncioInboundQueue


@dataclass(slots=True)
class FakeRedis:
    values: dict[str, str] = field(default_factory=dict)

    async def get(self, name: str) -> bytes | str | None:
        return self.values.get(name)

    async def set(
        self,
        name: str,
        value: str,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> object:
        if nx and name in self.values:
            return False
        self.values[name] = value
        return True

    async def delete(self, *names: str) -> object:
        for name in names:
            self.values.pop(name, None)
        return len(names)

    async def scan_iter(
        self,
        match: str | None = None,
        count: int | None = None,
    ) -> AsyncIterator[bytes | str]:
        prefix = (match or "").removesuffix("*")
        for key in list(self.values):
            if not prefix or key.startswith(prefix):
                yield key


@dataclass(slots=True)
class FakeIdempotencyStore:
    claims: list[InboundEvent] = field(default_factory=list)

    async def claim(self, event: InboundEvent) -> InboundIdempotencyClaim:
        self.claims.append(event)
        return InboundIdempotencyClaim(claimed=True, event_id=event.event_id)

    async def release_claim(self, *, event_id: object) -> None:
        raise AssertionError("release_claim should not be called")

    async def mark_status(self, **kwargs: object) -> None:
        raise AssertionError("mark_status should not be called")


def media_group_event(
    *,
    message_id: str,
    text: str | None = None,
    file_id: str,
) -> InboundEvent:
    user_id = uuid4()
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        external_message_id=message_id,
        external_update_id=f"9{message_id}",
        idempotency_key=f"telegram:12345:{message_id}",
        user_id=user_id,
        message_type=MessageType.MIXED if text else MessageType.MEDIA,
        text=text,
        attachments=[
            Attachment(
                attachment_id=file_id,
                attachment_type=AttachmentType.IMAGE,
                external_id=file_id,
                content_type="image/jpeg",
            )
        ],
        channel_metadata={"media_group_id": "album-1"},
    )


async def test_media_group_flush_publishes_one_aggregate_event_with_all_images() -> None:
    user_id = uuid4()
    first = media_group_event(message_id="42", text="Сравни эти фото", file_id="file-1")
    second = media_group_event(message_id="43", file_id="file-2")
    first.user_id = user_id
    second.user_id = user_id
    redis = FakeRedis()
    aggregator = RedisInboundMediaGroupAggregator(
        redis,
        debounce_seconds=0.01,
        ttl_seconds=60,
        lock_ttl_seconds=1,
    )
    queue = AsyncioInboundQueue()
    idempotency_store = FakeIdempotencyStore()
    worker = InboundMediaGroupFlushWorker(
        aggregator=aggregator,
        inbound_queue=queue,
        idempotency_store=idempotency_store,
        publish_timeout_seconds=1,
        flush_interval_seconds=0.01,
    )

    await aggregator.add(second)
    await aggregator.add(first)
    await asyncio.sleep(0.02)
    await worker.flush_once()
    published = await queue.consume()

    assert published.message_type is MessageType.MIXED
    assert published.text == "Сравни эти фото"
    assert published.idempotency_key == "telegram:12345:media_group:album-1"
    assert published.channel_metadata["media_group_aggregated"] is True
    assert published.channel_metadata["source_message_ids"] == ["42", "43"]
    assert [attachment.external_id for attachment in published.attachments] == [
        "file-1",
        "file-2",
    ]
    assert idempotency_store.claims == [published]
