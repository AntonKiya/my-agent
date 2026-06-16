from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from agent_service.feedback import FeedbackEntry, PendingFeedback, RedisFeedbackStateStore
from agent_service.feedback.postgres import INSERT_FEEDBACK_SQL, PostgresFeedbackStore


@dataclass(slots=True)
class FakeRedis:
    values: dict[str, str] = field(default_factory=dict)
    expiries: dict[str, int | None] = field(default_factory=dict)

    async def get(self, name: str) -> str | None:
        return self.values.get(name)

    async def set(self, name: str, value: str, *, ex: int | None = None) -> object:
        self.values[name] = value
        self.expiries[name] = ex
        return True

    async def delete(self, *names: str) -> object:
        for name in names:
            self.values.pop(name, None)
            self.expiries.pop(name, None)
        return len(names)


@dataclass(slots=True)
class FakeConnection:
    execute_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        return "OK"


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

    def acquire(self) -> AbstractAsyncContextManager[FakeConnection]:
        return FakeAcquire(self.connection)


async def test_redis_feedback_state_store_roundtrip_with_ttl() -> None:
    client = FakeRedis()
    store = RedisFeedbackStateStore(client, ttl_seconds=60)
    pending = PendingFeedback(
        user_id=uuid4(),
        conversation_id=uuid4(),
        channel="telegram",
        request_inbound_event_id=uuid4(),
        requested_at=datetime(2026, 6, 16, 12, 0, tzinfo=UTC),
    )

    await store.set_pending(pending=pending)
    loaded = await store.get_pending(conversation_id=pending.conversation_id)

    assert loaded == pending
    assert client.expiries[store.key_for_conversation(pending.conversation_id)] == 60

    await store.clear_pending(conversation_id=pending.conversation_id)

    assert await store.get_pending(conversation_id=pending.conversation_id) is None


async def test_postgres_feedback_store_inserts_feedback() -> None:
    connection = FakeConnection()
    store = PostgresFeedbackStore(FakePool(connection))
    feedback = FeedbackEntry(
        id=uuid4(),
        user_id=uuid4(),
        conversation_id=uuid4(),
        source_channel="telegram",
        source_external_user_id="67890",
        source_external_chat_id="12345",
        source_thread_id="7",
        source_inbound_event_id=uuid4(),
        request_inbound_event_id=uuid4(),
        text="хочу экспорт истории",
        metadata={"external_message_id": "42"},
        created_at=datetime(2026, 6, 16, 12, 5, tzinfo=UTC),
    )

    created = await store.create(feedback=feedback)

    assert created == feedback
    assert len(connection.execute_calls) == 1
    query, args = connection.execute_calls[0]
    assert query == INSERT_FEEDBACK_SQL
    assert args[0:10] == (
        feedback.id,
        feedback.user_id,
        feedback.conversation_id,
        "telegram",
        "67890",
        "12345",
        "7",
        feedback.source_inbound_event_id,
        feedback.request_inbound_event_id,
        "хочу экспорт истории",
    )
    assert args[10] == '{"external_message_id":"42"}'
    assert args[11] == feedback.created_at
