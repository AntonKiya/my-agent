from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from agent_service.channels import InboundEvent
from agent_service.users import (
    ChannelIdentity,
    ObservedChannelIdentity,
    User,
    UserResolutionError,
    UserResolutionStatus,
    UserResolver,
    UserStatus,
    UserStore,
    UserWithIdentity,
    observed_channel_identity_from_event,
)


class FakeUserStore:
    def __init__(self, result: UserWithIdentity) -> None:
        self.result = result
        self.observed_identities: list[ObservedChannelIdentity] = []

    async def get_by_channel_identity(self, *, lookup: object) -> UserWithIdentity | None:
        return self.result

    async def get_or_create_active_user_with_identity(
        self,
        *,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        self.observed_identities.append(identity)
        return self.result

    async def update_identity_seen(
        self,
        *,
        user: UserWithIdentity,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        self.observed_identities.append(identity)
        return user


def inbound_event(*, user_id: UUID | None = None) -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        external_message_id="42",
        external_update_id="100",
        idempotency_key="telegram:12345:42",
        text="hello",
        user_id=user_id,
        channel_metadata={
            "username": "handle",
            "first_name": "Anton",
            "chat_type": "private",
            "empty_value": None,
        },
        received_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )


def user_with_identity(
    *,
    status: UserStatus = UserStatus.ACTIVE,
    metadata: dict[str, object] | None = None,
) -> UserWithIdentity:
    user = User(status=status, metadata=metadata or {})
    return UserWithIdentity(
        user=user,
        identity=ChannelIdentity(
            user_id=user.id,
            channel="telegram",
            external_user_id="67890",
            external_chat_id="12345",
            username="handle",
            metadata={"first_name": "Anton"},
        ),
    )


async def test_user_resolver_returns_event_with_internal_user_id_for_active_user() -> None:
    stored = user_with_identity()
    store = FakeUserStore(stored)
    resolver = UserResolver(store)

    result = await resolver.resolve(inbound_event())

    assert isinstance(store, UserStore)
    assert result.status is UserResolutionStatus.RESOLVED
    assert result.user == stored.user
    assert result.identity == stored.identity
    assert result.event is not None
    assert result.event.user_id == stored.user.id
    assert result.event.external_user_id == "67890"
    assert store.observed_identities[0].external_user_id == "67890"


async def test_user_resolver_passes_profile_timezone_to_event_metadata() -> None:
    stored = user_with_identity(metadata={"timezone": "Europe/Moscow"})
    resolver = UserResolver(FakeUserStore(stored))

    result = await resolver.resolve(inbound_event())

    assert result.event is not None
    assert result.event.metadata["user_timezone"] == "Europe/Moscow"


async def test_user_resolver_rejects_blocked_user_without_resolved_event() -> None:
    resolver = UserResolver(FakeUserStore(user_with_identity(status=UserStatus.BLOCKED)))

    result = await resolver.resolve(inbound_event())

    assert result.status is UserResolutionStatus.BLOCKED
    assert result.event is None
    assert result.reason == "user is blocked"


async def test_user_resolver_rejects_pending_user_without_resolved_event() -> None:
    resolver = UserResolver(FakeUserStore(user_with_identity(status=UserStatus.PENDING)))

    result = await resolver.resolve(inbound_event())

    assert result.status is UserResolutionStatus.PENDING
    assert result.event is None
    assert result.reason == "user is pending"


async def test_user_resolver_rejects_event_with_different_existing_user_id() -> None:
    resolver = UserResolver(FakeUserStore(user_with_identity()))

    with pytest.raises(UserResolutionError):
        await resolver.resolve(inbound_event(user_id=uuid4()))


def test_observed_identity_from_event_keeps_username_out_of_metadata() -> None:
    observed = observed_channel_identity_from_event(inbound_event())

    assert observed.channel == "telegram"
    assert observed.external_user_id == "67890"
    assert observed.external_chat_id == "12345"
    assert observed.username == "handle"
    assert observed.metadata == {
        "first_name": "Anton",
        "chat_type": "private",
    }
    assert observed.observed_at == datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
