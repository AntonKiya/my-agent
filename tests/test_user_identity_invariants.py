import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agent_service.channels import InboundEvent
from agent_service.inbound import InboundIntakeService
from agent_service.messaging import AsyncioInboundQueue
from agent_service.users import (
    ChannelIdentity,
    ChannelIdentityLookup,
    ObservedChannelIdentity,
    User,
    UserResolver,
    UserStatus,
    UserStore,
    UserWithIdentity,
)


@dataclass(slots=True)
class InMemoryInvariantUserStore:
    identities: dict[tuple[str, str], UserWithIdentity] = field(default_factory=dict)
    create_count: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get_by_channel_identity(
        self,
        *,
        lookup: ChannelIdentityLookup,
    ) -> UserWithIdentity | None:
        return self.identities.get((lookup.channel, lookup.external_user_id))

    async def get_or_create_active_user_with_identity(
        self,
        *,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        async with self._lock:
            key = (identity.channel, identity.external_user_id)
            existing = self.identities.get(key)
            if existing is not None:
                updated = await self.update_identity_seen(user=existing, identity=identity)
                self.identities[key] = updated
                return updated

            user = User(
                status=UserStatus.ACTIVE,
                created_at=identity.observed_at,
                updated_at=identity.observed_at,
            )
            stored = UserWithIdentity(
                user=user,
                identity=ChannelIdentity(
                    user_id=user.id,
                    channel=identity.channel,
                    external_user_id=identity.external_user_id,
                    external_chat_id=identity.external_chat_id,
                    username=identity.username,
                    metadata=dict(identity.metadata),
                    created_at=identity.observed_at,
                    updated_at=identity.observed_at,
                    last_seen_at=identity.observed_at,
                ),
            )
            self.identities[key] = stored
            self.create_count += 1
            return stored

    async def update_identity_seen(
        self,
        *,
        user: UserWithIdentity,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        return UserWithIdentity(
            user=user.user,
            identity=ChannelIdentity(
                id=user.identity.id,
                user_id=user.identity.user_id,
                channel=user.identity.channel,
                external_user_id=user.identity.external_user_id,
                external_chat_id=identity.external_chat_id,
                username=identity.username,
                metadata=dict(identity.metadata),
                created_at=user.identity.created_at,
                updated_at=identity.observed_at,
                last_seen_at=identity.observed_at,
            ),
        )

    async def seed(
        self,
        *,
        channel: str,
        external_user_id: str,
        status: UserStatus,
    ) -> UserWithIdentity:
        identity = ObservedChannelIdentity(
            channel=channel,
            external_user_id=external_user_id,
            external_chat_id=f"chat-{external_user_id}",
            username="seeded",
            observed_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
        )
        stored = await self.get_or_create_active_user_with_identity(identity=identity)
        stored.user.status = status
        return stored


def inbound_event(
    *,
    channel: str = "telegram",
    external_user_id: str = "67890",
    external_chat_id: str = "12345",
    external_message_id: str = "42",
    username: str = "handle",
    first_name: str = "Anton",
) -> InboundEvent:
    return InboundEvent(
        channel=channel,
        external_user_id=external_user_id,
        external_chat_id=external_chat_id,
        external_message_id=external_message_id,
        idempotency_key=f"{channel}:{external_chat_id}:{external_message_id}",
        text="hello",
        channel_metadata={
            "username": username,
            "first_name": first_name,
            "chat_type": "private",
        },
        received_at=datetime(2026, 5, 29, 12, 30, tzinfo=UTC),
    )


def intake_for(
    store: InMemoryInvariantUserStore,
) -> tuple[InboundIntakeService, AsyncioInboundQueue]:
    queue = AsyncioInboundQueue()
    return InboundIntakeService(
        user_resolver=UserResolver(store),
        inbound_queue=queue,
    ), queue


async def test_new_telegram_user_is_created_active_and_published_with_user_id() -> None:
    store = InMemoryInvariantUserStore()
    intake, queue = intake_for(store)

    result = await intake.accept(inbound_event())

    assert isinstance(store, UserStore)
    assert result.published
    published = await queue.consume()
    stored = store.identities[("telegram", "67890")]
    assert stored.user.status is UserStatus.ACTIVE
    assert published.user_id == stored.user.id
    assert store.create_count == 1


async def test_repeated_external_identity_reuses_user_and_updates_mutable_metadata() -> None:
    store = InMemoryInvariantUserStore()
    intake, queue = intake_for(store)

    await intake.accept(inbound_event(username="old_handle", first_name="Old"))
    first = await queue.consume()
    await intake.accept(
        inbound_event(
            external_message_id="43",
            username="new_handle",
            first_name="New",
        )
    )
    second = await queue.consume()

    stored = store.identities[("telegram", "67890")]
    assert first.user_id == second.user_id == stored.user.id
    assert stored.identity.username == "new_handle"
    assert stored.identity.metadata["first_name"] == "New"
    assert store.create_count == 1


async def test_username_is_not_identity_and_does_not_merge_different_external_users() -> None:
    store = InMemoryInvariantUserStore()
    intake, queue = intake_for(store)

    await intake.accept(inbound_event(external_user_id="1", external_chat_id="10", username="same"))
    first = await queue.consume()
    await intake.accept(inbound_event(external_user_id="2", external_chat_id="20", username="same"))
    second = await queue.consume()

    assert first.user_id != second.user_id
    assert store.create_count == 2
    assert ("telegram", "1") in store.identities
    assert ("telegram", "2") in store.identities


async def test_same_external_user_id_in_different_channels_does_not_mix_users() -> None:
    store = InMemoryInvariantUserStore()
    intake, queue = intake_for(store)

    await intake.accept(inbound_event(channel="telegram", external_user_id="42"))
    telegram_event = await queue.consume()
    await intake.accept(inbound_event(channel="slack", external_user_id="42"))
    slack_event = await queue.consume()

    assert telegram_event.user_id != slack_event.user_id
    assert ("telegram", "42") in store.identities
    assert ("slack", "42") in store.identities
    assert store.create_count == 2


async def test_blocked_user_never_reaches_inbound_queue() -> None:
    store = InMemoryInvariantUserStore()
    await store.seed(
        channel="telegram",
        external_user_id="67890",
        status=UserStatus.BLOCKED,
    )
    intake, queue = intake_for(store)

    result = await intake.accept(inbound_event())

    assert not result.published
    assert queue.is_empty


async def test_concurrent_first_touch_for_same_identity_creates_one_user() -> None:
    store = InMemoryInvariantUserStore()
    intake, queue = intake_for(store)

    results = await asyncio.gather(
        *[
            intake.accept(inbound_event(external_message_id=str(message_id)))
            for message_id in range(50)
        ]
    )

    user_ids = {store.identities[("telegram", "67890")].user.id}
    for _ in results:
        published = await queue.consume()
        if published.user_id is not None:
            user_ids.add(published.user_id)

    assert all(result.published for result in results)
    assert len(user_ids) == 1
    assert store.create_count == 1
    assert queue.is_empty
