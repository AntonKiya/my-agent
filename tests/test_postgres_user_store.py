from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg
import pytest

from agent_service.users import (
    ChannelIdentity,
    ChannelIdentityLookup,
    ObservedChannelIdentity,
    PostgresUserStore,
    User,
    UserResolutionError,
    UserStatus,
    UserStore,
    UserWithIdentity,
)


@dataclass(slots=True)
class FakeTransaction:
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> "FakeTransaction":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self.exited = True


@dataclass(slots=True)
class FakeConnection:
    fetch_results: list[Mapping[str, object] | None] = field(default_factory=list)
    execute_errors: list[BaseException | None] = field(default_factory=list)
    fetch_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    execute_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    transactions: list[FakeTransaction] = field(default_factory=list)

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        self.fetch_calls.append((query, args))
        if not self.fetch_results:
            return None
        return self.fetch_results.pop(0)

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        if self.execute_errors:
            error = self.execute_errors.pop(0)
            if error is not None:
                raise error
        return "OK"

    def transaction(self) -> FakeTransaction:
        transaction = FakeTransaction()
        self.transactions.append(transaction)
        return transaction


@dataclass(slots=True)
class FakeAcquire:
    connection: FakeConnection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


@dataclass(slots=True)
class FakePool:
    connection: FakeConnection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


def user_identity_row(
    *,
    user_id: UUID | None = None,
    identity_id: UUID | None = None,
    channel: str = "telegram",
    external_user_id: str = "67890",
    external_chat_id: str | None = "12345",
    username: str | None = "handle",
    metadata: Mapping[str, object] | str | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    created_at = now or datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
    resolved_user_id = user_id or uuid4()
    return {
        "user_id": resolved_user_id,
        "user_status": "active",
        "user_metadata": {},
        "user_created_at": created_at,
        "user_updated_at": created_at,
        "identity_id": identity_id or uuid4(),
        "identity_user_id": resolved_user_id,
        "identity_channel": channel,
        "identity_external_user_id": external_user_id,
        "identity_external_chat_id": external_chat_id,
        "identity_username": username,
        "identity_metadata": metadata if metadata is not None else {"first_name": "Anton"},
        "identity_created_at": created_at,
        "identity_updated_at": created_at,
        "identity_last_seen_at": created_at,
    }


async def test_postgres_user_store_loads_by_stable_channel_identity_only() -> None:
    connection = FakeConnection(fetch_results=[user_identity_row(username="old_handle")])
    store: UserStore = PostgresUserStore(FakePool(connection))

    result = await store.get_by_channel_identity(
        lookup=ChannelIdentityLookup(channel="telegram", external_user_id="67890"),
    )

    assert isinstance(store, UserStore)
    assert result is not None
    assert result.user.status is UserStatus.ACTIVE
    assert result.identity.external_user_id == "67890"
    assert result.identity.username == "old_handle"
    assert connection.fetch_calls[0][1] == ("telegram", "67890")


async def test_postgres_user_store_creates_active_user_and_identity_atomically() -> None:
    observed_at = datetime(2026, 5, 29, 12, 30, tzinfo=UTC)
    connection = FakeConnection(fetch_results=[None])
    store = PostgresUserStore(FakePool(connection))

    result = await store.get_or_create_active_user_with_identity(
        identity=ObservedChannelIdentity(
            channel="telegram",
            external_user_id="67890",
            external_chat_id="12345",
            username="handle",
            metadata={"first_name": "Anton"},
            observed_at=observed_at,
        ),
    )

    assert result.user.status is UserStatus.ACTIVE
    assert result.user.created_at == observed_at
    assert result.identity.user_id == result.user.id
    assert result.identity.channel == "telegram"
    assert result.identity.external_user_id == "67890"
    assert result.identity.metadata == {"first_name": "Anton"}
    assert len(connection.transactions) == 1
    assert connection.transactions[0].entered
    assert connection.transactions[0].exited
    assert len(connection.execute_calls) == 2
    assert connection.execute_calls[0][1][1] == "active"
    assert connection.execute_calls[1][1][2:6] == (
        "telegram",
        "67890",
        "12345",
        "handle",
    )


async def test_postgres_user_store_updates_seen_identity_without_changing_user() -> None:
    original_seen_at = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
    observed_at = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)
    user = User(id=uuid4(), status=UserStatus.ACTIVE, created_at=original_seen_at)
    stored = UserWithIdentity(
        user=user,
        identity=ChannelIdentity(
            id=uuid4(),
            user_id=user.id,
            channel="telegram",
            external_user_id="67890",
            external_chat_id="old-chat",
            username="old_handle",
            metadata={"first_name": "Old"},
            created_at=original_seen_at,
            updated_at=original_seen_at,
            last_seen_at=original_seen_at,
        ),
    )
    connection = FakeConnection()
    store = PostgresUserStore(FakePool(connection))

    result = await store.update_identity_seen(
        user=stored,
        identity=ObservedChannelIdentity(
            channel="telegram",
            external_user_id="67890",
            external_chat_id="new-chat",
            username="new_handle",
            metadata={"first_name": "New"},
            observed_at=observed_at,
        ),
    )

    assert result.user.id == user.id
    assert result.identity.id == stored.identity.id
    assert result.identity.external_chat_id == "new-chat"
    assert result.identity.username == "new_handle"
    assert result.identity.metadata == {"first_name": "New"}
    assert result.identity.updated_at == observed_at
    assert result.identity.last_seen_at == observed_at
    assert len(connection.execute_calls) == 1


async def test_postgres_user_store_rereads_identity_after_unique_conflict() -> None:
    existing_row = user_identity_row(username="winner")
    connection = FakeConnection(
        fetch_results=[None, existing_row],
        execute_errors=[None, asyncpg.UniqueViolationError("duplicate identity")],
    )
    store = PostgresUserStore(FakePool(connection))

    result = await store.get_or_create_active_user_with_identity(
        identity=ObservedChannelIdentity(
            channel="telegram",
            external_user_id="67890",
            external_chat_id="12345",
            username="handle",
            metadata={"first_name": "Anton"},
        ),
    )

    assert result.identity.username == "handle"
    assert len(connection.fetch_calls) == 2
    assert len(connection.execute_calls) == 3
    assert connection.fetch_calls[1][1] == ("telegram", "67890")


async def test_postgres_user_store_rejects_seen_update_for_mismatched_identity() -> None:
    user = User()
    stored = UserWithIdentity(
        user=user,
        identity=ChannelIdentity(
            user_id=user.id,
            channel="telegram",
            external_user_id="67890",
        ),
    )
    store = PostgresUserStore(FakePool(FakeConnection()))

    with pytest.raises(UserResolutionError):
        await store.update_identity_seen(
            user=stored,
            identity=ObservedChannelIdentity(
                channel="telegram",
                external_user_id="different",
            ),
        )
