from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest

from agent_service.memory import (
    ConversationCompactionStore,
    ConversationSummary,
    ConversationSummaryStatus,
    PostgresConversationCompactionStore,
    PostgresMemoryError,
    PostgresPool,
)


@dataclass(slots=True)
class FakeConnection:
    fetchrow_results: list[Mapping[str, object] | None] = field(default_factory=list)
    fetchrow_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        self.fetchrow_calls.append((query, args))
        if not self.fetchrow_results:
            return None
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        return []

    async def execute(self, query: str, *args: object) -> str:
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

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


def conversation_summary(
    *,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
    from_sequence: int = 1,
    to_sequence: int = 5,
) -> ConversationSummary:
    last_message_id = uuid4()
    return ConversationSummary(
        id=uuid4(),
        conversation_id=conversation_id or uuid4(),
        user_id=user_id or uuid4(),
        from_sequence=from_sequence,
        to_sequence=to_sequence,
        previous_summary="previous",
        summary="compressed context",
        compacted_message_ids=[uuid4(), last_message_id],
        last_compacted_message_id=last_message_id,
        input_token_count=120,
        output_token_count=30,
        model="summary-model",
        trace_id="trace-1",
        metadata={"source": "test"},
        created_at=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
    )


def summary_row(summary: ConversationSummary) -> dict[str, object]:
    return {
        "id": summary.id,
        "conversation_id": summary.conversation_id,
        "user_id": summary.user_id,
        "from_sequence": summary.from_sequence,
        "to_sequence": summary.to_sequence,
        "previous_summary": summary.previous_summary,
        "summary": summary.summary,
        "compacted_message_ids": [str(value) for value in summary.compacted_message_ids],
        "last_compacted_message_id": summary.last_compacted_message_id,
        "input_token_count": summary.input_token_count,
        "output_token_count": summary.output_token_count,
        "model": summary.model,
        "status": summary.status.value,
        "trace_id": summary.trace_id,
        "metadata": summary.metadata,
        "created_at": summary.created_at,
    }


async def test_postgres_compaction_store_appends_summary_state() -> None:
    summary = conversation_summary()
    connection = FakeConnection(fetchrow_results=[summary_row(summary)])
    store: ConversationCompactionStore = PostgresConversationCompactionStore(
        cast(PostgresPool, FakePool(connection))
    )

    stored = await store.append_summary(summary=summary)

    assert isinstance(store, ConversationCompactionStore)
    assert stored == summary
    assert len(connection.fetchrow_calls) == 1
    fetchrow_args = connection.fetchrow_calls[0][1]
    assert fetchrow_args[0:5] == (
        summary.id,
        summary.conversation_id,
        summary.user_id,
        summary.from_sequence,
        summary.to_sequence,
    )
    assert fetchrow_args[8] == summary.last_compacted_message_id
    assert fetchrow_args[12] == ConversationSummaryStatus.COMPLETED.value


async def test_postgres_compaction_store_rejects_summary_for_missing_conversation() -> None:
    connection = FakeConnection(fetchrow_results=[None])
    store = PostgresConversationCompactionStore(cast(PostgresPool, FakePool(connection)))

    with pytest.raises(PostgresMemoryError):
        await store.append_summary(summary=conversation_summary())


async def test_postgres_compaction_store_append_is_idempotent_for_completed_sequence() -> None:
    requested = conversation_summary(to_sequence=9)
    existing = conversation_summary(
        conversation_id=requested.conversation_id,
        user_id=requested.user_id,
        from_sequence=3,
        to_sequence=9,
    )
    connection = FakeConnection(fetchrow_results=[summary_row(existing)])
    store = PostgresConversationCompactionStore(cast(PostgresPool, FakePool(connection)))

    stored = await store.append_summary(summary=requested)

    assert stored == existing
    assert stored.id != requested.id
    assert connection.fetchrow_calls[0][1][1:5] == (
        requested.conversation_id,
        requested.user_id,
        requested.from_sequence,
        requested.to_sequence,
    )


async def test_postgres_compaction_store_reads_latest_completed_summary() -> None:
    expected = conversation_summary(from_sequence=3, to_sequence=9)
    connection = FakeConnection(fetchrow_results=[summary_row(expected)])
    store = PostgresConversationCompactionStore(cast(PostgresPool, FakePool(connection)))

    loaded = await store.get_latest_completed_summary(
        conversation_id=expected.conversation_id,
        user_id=expected.user_id,
    )

    assert loaded == expected
    assert connection.fetchrow_calls[0][1] == (expected.conversation_id, expected.user_id)


async def test_postgres_compaction_store_reads_completed_summary_by_sequence() -> None:
    expected = conversation_summary(from_sequence=3, to_sequence=9)
    connection = FakeConnection(fetchrow_results=[summary_row(expected)])
    store = PostgresConversationCompactionStore(cast(PostgresPool, FakePool(connection)))

    loaded = await store.get_completed_summary_by_sequence(
        conversation_id=expected.conversation_id,
        user_id=expected.user_id,
        to_sequence=expected.to_sequence,
    )

    assert loaded == expected
    assert connection.fetchrow_calls[0][1] == (
        expected.conversation_id,
        expected.user_id,
        expected.to_sequence,
    )


async def test_postgres_compaction_store_returns_none_when_summary_missing() -> None:
    store = PostgresConversationCompactionStore(
        cast(PostgresPool, FakePool(FakeConnection(fetchrow_results=[None])))
    )

    loaded = await store.get_latest_completed_summary(
        conversation_id=uuid4(),
        user_id=uuid4(),
    )

    assert loaded is None


async def test_postgres_compaction_store_rejects_invalid_summary_sequence_lookup() -> None:
    store = PostgresConversationCompactionStore(
        cast(PostgresPool, FakePool(FakeConnection()))
    )

    with pytest.raises(ValueError):
        await store.get_completed_summary_by_sequence(
            conversation_id=uuid4(),
            user_id=uuid4(),
            to_sequence=0,
        )
