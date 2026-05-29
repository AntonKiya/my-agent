from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg
import pytest

from agent_service.conversations import (
    ConversationLookup,
    ConversationOwnershipError,
    ConversationStatus,
    ConversationStore,
    ConversationType,
    ObservedConversation,
    PostgresConversationStore,
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


def conversation_row(
    *,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
    channel: str = "telegram",
    conversation_key: str = "telegram:private:12345",
    external_chat_id: str = "12345",
    conversation_type: str = "private",
    thread_id: str | None = None,
    metadata: Mapping[str, object] | str | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    created_at = now or datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
    return {
        "id": conversation_id or uuid4(),
        "user_id": user_id or uuid4(),
        "channel": channel,
        "conversation_key": conversation_key,
        "external_chat_id": external_chat_id,
        "type": conversation_type,
        "thread_id": thread_id,
        "status": "active",
        "metadata": metadata if metadata is not None else {"source": "private"},
        "created_at": created_at,
        "updated_at": created_at,
    }


def observed_conversation(
    *,
    user_id: UUID | None = None,
    observed_at: datetime | None = None,
) -> ObservedConversation:
    return ObservedConversation(
        user_id=user_id or uuid4(),
        channel="telegram",
        conversation_key="telegram:private:12345",
        external_chat_id="12345",
        type=ConversationType.PRIVATE,
        metadata={"source": "private"},
        observed_at=observed_at or datetime(2026, 5, 29, 12, 30, tzinfo=UTC),
    )


async def test_postgres_conversation_store_loads_by_conversation_key_only() -> None:
    row = conversation_row(metadata='{"source":"private"}')
    connection = FakeConnection(fetch_results=[row])
    store: ConversationStore = PostgresConversationStore(FakePool(connection))

    result = await store.get_by_key(
        lookup=ConversationLookup(conversation_key="telegram:private:12345"),
    )

    assert isinstance(store, ConversationStore)
    assert result is not None
    assert result.status is ConversationStatus.ACTIVE
    assert result.type is ConversationType.PRIVATE
    assert result.metadata == {"source": "private"}
    assert connection.fetch_calls[0][1] == ("telegram:private:12345",)


async def test_postgres_conversation_store_creates_conversation_atomically() -> None:
    observed_at = datetime(2026, 5, 29, 12, 30, tzinfo=UTC)
    observed = observed_conversation(observed_at=observed_at)
    connection = FakeConnection(fetch_results=[None])
    store = PostgresConversationStore(FakePool(connection))

    result = await store.get_or_create_conversation(conversation=observed)

    assert result.user_id == observed.user_id
    assert result.channel == "telegram"
    assert result.conversation_key == "telegram:private:12345"
    assert result.external_chat_id == "12345"
    assert result.status is ConversationStatus.ACTIVE
    assert result.created_at == observed_at
    assert len(connection.transactions) == 1
    assert connection.transactions[0].entered
    assert connection.transactions[0].exited
    assert len(connection.execute_calls) == 1
    assert connection.execute_calls[0][1][1:8] == (
        observed.user_id,
        "telegram",
        "telegram:private:12345",
        "12345",
        "private",
        None,
        "active",
    )


async def test_postgres_conversation_store_updates_seen_conversation_without_changing_id() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    created_at = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
    observed_at = datetime(2026, 5, 29, 12, 5, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            conversation_row(
                conversation_id=conversation_id,
                user_id=user_id,
                now=created_at,
                metadata={"old": True},
            )
        ]
    )
    store = PostgresConversationStore(FakePool(connection))

    result = await store.get_or_create_conversation(
        conversation=observed_conversation(user_id=user_id, observed_at=observed_at)
    )

    assert result.id == conversation_id
    assert result.user_id == user_id
    assert result.metadata == {"source": "private"}
    assert result.created_at == created_at
    assert result.updated_at == observed_at
    assert len(connection.execute_calls) == 1


async def test_postgres_conversation_store_rejects_existing_key_for_different_user() -> None:
    existing_user_id = uuid4()
    observed_user_id = uuid4()
    connection = FakeConnection(
        fetch_results=[conversation_row(user_id=existing_user_id)],
    )
    store = PostgresConversationStore(FakePool(connection))

    with pytest.raises(ConversationOwnershipError):
        await store.get_or_create_conversation(
            conversation=observed_conversation(user_id=observed_user_id)
        )


async def test_postgres_conversation_store_rereads_after_unique_conflict() -> None:
    user_id = uuid4()
    existing_row = conversation_row(user_id=user_id)
    connection = FakeConnection(
        fetch_results=[None, existing_row],
        execute_errors=[asyncpg.UniqueViolationError("duplicate conversation")],
    )
    store = PostgresConversationStore(FakePool(connection))

    result = await store.get_or_create_conversation(
        conversation=observed_conversation(user_id=user_id)
    )

    assert result.user_id == user_id
    assert len(connection.fetch_calls) == 2
    assert len(connection.execute_calls) == 2
    assert connection.fetch_calls[1][1] == ("telegram:private:12345",)


def test_conversation_store_protocol_accepts_postgres_store() -> None:
    store = PostgresConversationStore(FakePool(FakeConnection()))

    assert isinstance(store, ConversationStore)
