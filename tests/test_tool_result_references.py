from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from agent_service.tool_refs import (
    InMemoryToolResultReferenceStore,
    PostgresToolResultReferenceStore,
    ToolResultReference,
)


class FakeAcquire(AbstractAsyncContextManager[object]):
    def __init__(self, connection: object) -> None:
        self.connection = connection

    async def __aenter__(self) -> object:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        return None


@dataclass
class FakeConnection:
    fetchrow_result: Mapping[str, object] | None = None
    execute_calls: list[tuple[str, tuple[object, ...]]] | None = None
    fetchrow_calls: list[tuple[str, tuple[object, ...]]] | None = None

    async def execute(self, query: str, *args: object) -> str:
        if self.execute_calls is None:
            self.execute_calls = []
        self.execute_calls.append((query, args))
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        if self.fetchrow_calls is None:
            self.fetchrow_calls = []
        self.fetchrow_calls.append((query, args))
        return self.fetchrow_result


@dataclass
class FakePool:
    connection: FakeConnection

    def acquire(self) -> AbstractAsyncContextManager[Any]:
        return FakeAcquire(self.connection)


def _reference(
    *,
    selection_id: str = "sel_ref0001",
    expires_at: datetime | None = None,
) -> ToolResultReference:
    return ToolResultReference(
        selection_id=selection_id,
        provider="demo",
        source_tool_name="demo_search",
        user_id=uuid4(),
        conversation_id=uuid4(),
        item_kind="offer",
        item_index=0,
        label="Offer",
        display_snapshot={"selection_id": selection_id, "name": "Offer"},
        ref_payload={"checkout_ref": {"opaque": "hash"}},
        expires_at=expires_at or datetime(2026, 7, 5, 12, 0, tzinfo=UTC),
        created_at=datetime(2026, 7, 5, 10, 0, tzinfo=UTC),
    )


async def test_in_memory_tool_result_reference_store_enforces_owner_and_expiry() -> None:
    reference = _reference()
    store = InMemoryToolResultReferenceStore(
        now_provider=lambda: datetime(2026, 7, 5, 11, 0, tzinfo=UTC)
    )

    await store.create(reference=reference)

    assert (
        await store.get(
            selection_id=reference.selection_id,
            user_id=reference.user_id,
            conversation_id=reference.conversation_id,
            provider="demo",
        )
        == reference
    )
    assert (
        await store.get(
            selection_id=reference.selection_id,
            user_id=uuid4(),
            conversation_id=reference.conversation_id,
            provider="demo",
        )
        is None
    )

    expired_store = InMemoryToolResultReferenceStore(
        now_provider=lambda: datetime(2026, 7, 5, 13, 0, tzinfo=UTC)
    )
    await expired_store.create(reference=reference)
    assert (
        await expired_store.get(
            selection_id=reference.selection_id,
            user_id=reference.user_id,
            conversation_id=reference.conversation_id,
            provider="demo",
        )
        is None
    )


async def test_postgres_tool_result_reference_store_writes_json_payloads() -> None:
    connection = FakeConnection()
    store = PostgresToolResultReferenceStore(FakePool(connection))
    reference = _reference()

    await store.create(reference=reference)

    assert connection.execute_calls is not None
    query, args = connection.execute_calls[0]
    assert "INSERT INTO tool_result_references" in query
    assert args[0] == reference.selection_id
    assert args[1] == "demo"
    assert args[8] == '{"selection_id":"sel_ref0001","name":"Offer"}'
    assert args[9] == '{"checkout_ref":{"opaque":"hash"}}'


async def test_postgres_tool_result_reference_store_reads_owned_unexpired_reference() -> None:
    reference = _reference()
    connection = FakeConnection(
        fetchrow_result={
            "selection_id": reference.selection_id,
            "provider": reference.provider,
            "source_tool_name": reference.source_tool_name,
            "user_id": reference.user_id,
            "conversation_id": reference.conversation_id,
            "item_kind": reference.item_kind,
            "item_index": reference.item_index,
            "label": reference.label,
            "display_snapshot": reference.display_snapshot,
            "ref_payload": reference.ref_payload,
            "expires_at": reference.expires_at,
            "created_at": reference.created_at,
        }
    )
    store = PostgresToolResultReferenceStore(FakePool(connection))

    result = await store.get(
        selection_id=reference.selection_id,
        user_id=reference.user_id,
        conversation_id=reference.conversation_id,
        provider="demo",
    )

    assert result == reference
    assert connection.fetchrow_calls is not None
    _, args = connection.fetchrow_calls[0]
    assert args == (
        reference.selection_id,
        reference.user_id,
        reference.conversation_id,
        "demo",
    )
