import pytest

from agent_service.channels import InboundEvent, InboundEventStatus
from agent_service.inbound import (
    InboundIdempotencyClaim,
    InboundIntakeService,
    InboundIntakeStatus,
    MediaGroupAddResult,
    MediaGroupAddStatus,
)
from agent_service.messaging.in_memory import AsyncioInboundQueue
from agent_service.users import (
    ChannelIdentity,
    User,
    UserResolutionError,
    UserResolutionResult,
    UserResolutionStatus,
    UserWithIdentity,
)


class FakeUserResolver:
    def __init__(self, resolution: UserResolutionResult) -> None:
        self.resolution = resolution
        self.events: list[InboundEvent] = []

    async def resolve(self, event: InboundEvent) -> UserResolutionResult:
        self.events.append(event)
        return self.resolution


class FakeIdempotencyStore:
    def __init__(self, *, claimed: bool = True) -> None:
        self.claimed = claimed
        self.claims: list[InboundEvent] = []
        self.released_event_ids: list[object] = []
        self.statuses: list[tuple[object, InboundEventStatus, str | None]] = []

    async def claim(self, event: InboundEvent) -> InboundIdempotencyClaim:
        self.claims.append(event)
        return InboundIdempotencyClaim(
            claimed=self.claimed,
            event_id=event.event_id,
            existing_event_id=event.event_id if not self.claimed else None,
            existing_status=InboundEventStatus.COMPLETED if not self.claimed else None,
        )

    async def release_claim(self, *, event_id: object) -> None:
        self.released_event_ids.append(event_id)

    async def mark_status(
        self,
        *,
        event_id: object,
        status: InboundEventStatus,
        failure_reason: str | None = None,
    ) -> None:
        self.statuses.append((event_id, status, failure_reason))


class FakeMediaGroupAggregator:
    def __init__(self) -> None:
        self.events: list[InboundEvent] = []

    async def add(self, event: InboundEvent) -> MediaGroupAddResult:
        self.events.append(event)
        return MediaGroupAddResult(
            status=MediaGroupAddStatus.BUFFERED,
            group_key="telegram:user:chat:album-1",
            item_count=1,
        )


def inbound_event() -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        external_message_id="42",
        idempotency_key="telegram:12345:42",
        text="hello",
    )


def user_with_identity() -> UserWithIdentity:
    user = User()
    return UserWithIdentity(
        user=user,
        identity=ChannelIdentity(
            user_id=user.id,
            channel="telegram",
            external_user_id="67890",
        ),
    )


async def test_inbound_intake_publishes_only_resolved_event_with_user_id() -> None:
    event = inbound_event()
    stored = user_with_identity()
    resolved_event = event.model_copy(update={"user_id": stored.user.id})
    queue = AsyncioInboundQueue()
    resolver = FakeUserResolver(
        UserResolutionResult(
            status=UserResolutionStatus.RESOLVED,
            user=stored.user,
            identity=stored.identity,
            event=resolved_event,
        )
    )
    service = InboundIntakeService(user_resolver=resolver, inbound_queue=queue)

    result = await service.accept(event)

    assert result.status is InboundIntakeStatus.PUBLISHED
    assert result.published
    assert result.user_resolution_status is UserResolutionStatus.RESOLVED
    assert result.queue_size == 1
    assert result.queue_maxsize == 0
    assert resolver.events == [event]
    published_event = await queue.consume()
    assert published_event == resolved_event
    assert published_event.user_id == stored.user.id


async def test_inbound_intake_claims_idempotency_before_publish() -> None:
    event = inbound_event()
    stored = user_with_identity()
    resolved_event = event.model_copy(update={"user_id": stored.user.id})
    queue = AsyncioInboundQueue()
    idempotency_store = FakeIdempotencyStore()
    service = InboundIntakeService(
        user_resolver=FakeUserResolver(
            UserResolutionResult(
                status=UserResolutionStatus.RESOLVED,
                user=stored.user,
                identity=stored.identity,
                event=resolved_event,
            )
        ),
        inbound_queue=queue,
        idempotency_store=idempotency_store,
    )

    result = await service.accept(event)

    assert result.status is InboundIntakeStatus.PUBLISHED
    assert idempotency_store.claims == [resolved_event]
    assert not idempotency_store.released_event_ids
    assert not queue.is_empty


async def test_inbound_intake_buffers_media_group_without_idempotency_claim_or_publish() -> None:
    event = inbound_event()
    stored = user_with_identity()
    resolved_event = event.model_copy(
        update={
            "user_id": stored.user.id,
            "message_type": "mixed",
            "channel_metadata": {"media_group_id": "album-1"},
        }
    )
    queue = AsyncioInboundQueue()
    idempotency_store = FakeIdempotencyStore()
    media_group_aggregator = FakeMediaGroupAggregator()
    service = InboundIntakeService(
        user_resolver=FakeUserResolver(
            UserResolutionResult(
                status=UserResolutionStatus.RESOLVED,
                user=stored.user,
                identity=stored.identity,
                event=resolved_event,
            )
        ),
        inbound_queue=queue,
        idempotency_store=idempotency_store,
        media_group_aggregator=media_group_aggregator,  # type: ignore[arg-type]
    )

    result = await service.accept(event)

    assert result.status is InboundIntakeStatus.BUFFERED
    assert not result.published
    assert idempotency_store.claims == []
    assert media_group_aggregator.events == [resolved_event]
    assert queue.is_empty


async def test_inbound_intake_suppresses_duplicate_claim_without_publish() -> None:
    event = inbound_event()
    stored = user_with_identity()
    resolved_event = event.model_copy(update={"user_id": stored.user.id})
    queue = AsyncioInboundQueue()
    idempotency_store = FakeIdempotencyStore(claimed=False)
    service = InboundIntakeService(
        user_resolver=FakeUserResolver(
            UserResolutionResult(
                status=UserResolutionStatus.RESOLVED,
                user=stored.user,
                identity=stored.identity,
                event=resolved_event,
            )
        ),
        inbound_queue=queue,
        idempotency_store=idempotency_store,
    )

    result = await service.accept(event)

    assert result.status is InboundIntakeStatus.DUPLICATE
    assert not result.published
    assert result.reason == "duplicate inbound event"
    assert idempotency_store.claims == [resolved_event]
    assert queue.is_empty


async def test_inbound_intake_does_not_publish_blocked_user() -> None:
    stored = user_with_identity()
    queue = AsyncioInboundQueue()
    service = InboundIntakeService(
        user_resolver=FakeUserResolver(
            UserResolutionResult(
                status=UserResolutionStatus.BLOCKED,
                user=stored.user,
                identity=stored.identity,
                reason="user is blocked",
            )
        ),
        inbound_queue=queue,
    )

    result = await service.accept(inbound_event())

    assert result.status is InboundIntakeStatus.REJECTED
    assert not result.published
    assert result.user_resolution_status is UserResolutionStatus.BLOCKED
    assert result.reason == "user is blocked"
    assert queue.is_empty


async def test_inbound_intake_rejects_impossible_resolved_result_without_event() -> None:
    stored = user_with_identity()
    queue = AsyncioInboundQueue()
    service = InboundIntakeService(
        user_resolver=FakeUserResolver(
            UserResolutionResult.model_construct(
                status=UserResolutionStatus.RESOLVED,
                user=stored.user,
                identity=stored.identity,
                event=None,
            )
        ),
        inbound_queue=queue,
    )

    with pytest.raises(UserResolutionError):
        await service.accept(inbound_event())

    assert queue.is_empty


async def test_inbound_intake_returns_overloaded_when_publish_times_out() -> None:
    event = inbound_event()
    stored = user_with_identity()
    resolved_event = event.model_copy(update={"user_id": stored.user.id})
    queue = AsyncioInboundQueue(maxsize=1)
    await queue.publish(inbound_event().model_copy(update={"text": "already queued"}))
    service = InboundIntakeService(
        user_resolver=FakeUserResolver(
            UserResolutionResult(
                status=UserResolutionStatus.RESOLVED,
                user=stored.user,
                identity=stored.identity,
                event=resolved_event,
            )
        ),
        inbound_queue=queue,
        publish_timeout_seconds=0.001,
    )

    result = await service.accept(event)

    assert result.status is InboundIntakeStatus.OVERLOADED
    assert not result.published
    assert result.reason == "inbound queue is overloaded"
    assert result.queue_size == 1
    assert result.queue_maxsize == 1
    assert queue.is_full


async def test_inbound_intake_releases_idempotency_claim_when_publish_times_out() -> None:
    event = inbound_event()
    stored = user_with_identity()
    resolved_event = event.model_copy(update={"user_id": stored.user.id})
    queue = AsyncioInboundQueue(maxsize=1)
    await queue.publish(inbound_event().model_copy(update={"text": "already queued"}))
    idempotency_store = FakeIdempotencyStore()
    service = InboundIntakeService(
        user_resolver=FakeUserResolver(
            UserResolutionResult(
                status=UserResolutionStatus.RESOLVED,
                user=stored.user,
                identity=stored.identity,
                event=resolved_event,
            )
        ),
        inbound_queue=queue,
        idempotency_store=idempotency_store,
        publish_timeout_seconds=0.001,
    )

    result = await service.accept(event)

    assert result.status is InboundIntakeStatus.OVERLOADED
    assert idempotency_store.released_event_ids == [resolved_event.event_id]


def test_inbound_intake_rejects_invalid_publish_timeout() -> None:
    with pytest.raises(ValueError):
        InboundIntakeService(
            user_resolver=FakeUserResolver(
                UserResolutionResult(status=UserResolutionStatus.BLOCKED)
            ),
            inbound_queue=AsyncioInboundQueue(),
            publish_timeout_seconds=0,
        )
