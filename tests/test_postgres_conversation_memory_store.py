from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from agent_service.channels import Attachment, AttachmentType
from agent_service.memory import (
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationMemoryStore,
    PostgresConversationMemoryStore,
    PostgresMemoryError,
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
    fetchrow_results: list[Mapping[str, object] | None] = field(default_factory=list)
    fetch_results: list[list[Mapping[str, object]]] = field(default_factory=list)
    fetchrow_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    fetch_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    execute_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    transactions: list[FakeTransaction] = field(default_factory=list)

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        self.fetchrow_calls.append((query, args))
        if not self.fetchrow_results:
            return None
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        self.fetch_calls.append((query, args))
        if not self.fetch_results:
            return []
        return self.fetch_results.pop(0)

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
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


def memory_message(
    *,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
    role: ConversationMemoryRole = ConversationMemoryRole.USER,
) -> ConversationMemoryMessage:
    return ConversationMemoryMessage(
        id=uuid4(),
        conversation_id=conversation_id or uuid4(),
        user_id=user_id or uuid4(),
        role=role,
        text="hello",
        attachments=[
            Attachment(
                attachment_type=AttachmentType.DOCUMENT,
                external_id="file-1",
                content_type="application/pdf",
            )
        ],
        inbound_event_id=uuid4(),
        trace_id="trace-1",
        token_count=3,
        metadata={"channel": "telegram"},
        created_at=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
    )


def message_row(
    *,
    message_id: UUID | None = None,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
    sequence: int = 1,
    role: str = "user",
    text: str | None = "hello",
) -> dict[str, object]:
    return {
        "id": message_id or uuid4(),
        "conversation_id": conversation_id or uuid4(),
        "user_id": user_id or uuid4(),
        "sequence": sequence,
        "role": role,
        "text": text,
        "attachments": [],
        "tool_name": None,
        "tool_call_id": None,
        "inbound_event_id": None,
        "outbound_event_id": None,
        "trace_id": "trace-1",
        "token_count": None,
        "metadata": {},
        "created_at": datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
    }


async def test_postgres_memory_store_appends_message_with_transactional_sequence() -> None:
    connection = FakeConnection(fetchrow_results=[{"message_sequence": 7}])
    store: ConversationMemoryStore = PostgresConversationMemoryStore(FakePool(connection))
    message = memory_message()

    stored = await store.append_message(message=message)

    assert isinstance(store, ConversationMemoryStore)
    assert stored.sequence == 7
    assert stored.id == message.id
    assert stored.conversation_id == message.conversation_id
    assert len(connection.transactions) == 1
    assert connection.transactions[0].entered
    assert connection.transactions[0].exited
    assert connection.fetchrow_calls[0][1] == (message.conversation_id, message.user_id)
    assert len(connection.execute_calls) == 1
    execute_args = connection.execute_calls[0][1]
    assert execute_args[0:5] == (
        message.id,
        message.conversation_id,
        message.user_id,
        7,
        "user",
    )
    assert execute_args[7:9] == (None, None)


async def test_postgres_memory_store_appends_tool_call_message() -> None:
    connection = FakeConnection(fetchrow_results=[{"message_sequence": 8}])
    store = PostgresConversationMemoryStore(FakePool(connection))
    message = ConversationMemoryMessage(
        conversation_id=uuid4(),
        user_id=uuid4(),
        role=ConversationMemoryRole.TOOL_CALL,
        tool_name="search",
        tool_call_id="call-1",
        metadata={"args": {"query": "hello"}},
    )

    stored = await store.append_message(message=message)

    execute_args = connection.execute_calls[0][1]
    assert stored.sequence == 8
    assert execute_args[4] == "tool_call"
    assert execute_args[7:9] == ("search", "call-1")


async def test_postgres_memory_store_rejects_append_when_conversation_user_mismatch() -> None:
    connection = FakeConnection(fetchrow_results=[None])
    store = PostgresConversationMemoryStore(FakePool(connection))

    with pytest.raises(PostgresMemoryError):
        await store.append_message(message=memory_message())

    assert connection.execute_calls == []


async def test_postgres_memory_store_lists_recent_messages_in_sequence_order() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    connection = FakeConnection(
        fetch_results=[
            [
                message_row(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    sequence=2,
                    role="assistant",
                    text="answer",
                ),
                message_row(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    sequence=3,
                    role="tool_result",
                    text=None,
                ),
            ]
        ]
    )
    store = PostgresConversationMemoryStore(FakePool(connection))

    messages = await store.list_recent_messages(conversation_id=conversation_id, limit=2)

    assert connection.fetch_calls[0][1] == (conversation_id, 2)
    assert [message.sequence for message in messages] == [2, 3]
    assert messages[0].role is ConversationMemoryRole.ASSISTANT
    assert messages[1].role is ConversationMemoryRole.TOOL_RESULT


async def test_postgres_memory_store_reads_current_message_sequence() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    connection = FakeConnection(fetchrow_results=[{"message_sequence": 12}])
    store = PostgresConversationMemoryStore(FakePool(connection))

    sequence = await store.current_message_sequence(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert sequence == 12
    assert connection.fetchrow_calls[0][1] == (conversation_id, user_id)


async def test_postgres_memory_store_rejects_sequence_read_when_conversation_missing() -> None:
    store = PostgresConversationMemoryStore(FakePool(FakeConnection(fetchrow_results=[None])))

    with pytest.raises(PostgresMemoryError):
        await store.current_message_sequence(conversation_id=uuid4(), user_id=uuid4())


async def test_postgres_memory_store_rejects_invalid_recent_limit() -> None:
    store = PostgresConversationMemoryStore(FakePool(FakeConnection()))

    with pytest.raises(ValueError):
        await store.list_recent_messages(conversation_id=uuid4(), limit=0)
