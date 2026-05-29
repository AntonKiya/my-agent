import pytest

from agent_service.channels import InboundEvent
from agent_service.inbound import InboundIntakeService, InboundIntakeStatus
from agent_service.messaging import AsyncioInboundQueue
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
    assert resolver.events == [event]
    published_event = await queue.consume()
    assert published_event == resolved_event
    assert published_event.user_id == stored.user.id


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
