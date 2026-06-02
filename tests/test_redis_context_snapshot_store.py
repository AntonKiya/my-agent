from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_service.memory import (
    ConversationContextSnapshot,
    ConversationContextSnapshotStore,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    RedisConversationContextSnapshotStore,
    RedisSnapshotError,
)


@dataclass(slots=True)
class FakeRedisClient:
    values: dict[str, str | bytes] = field(default_factory=dict)
    expiries: dict[str, int | None] = field(default_factory=dict)
    get_calls: list[str] = field(default_factory=list)
    set_calls: list[tuple[str, str, int | None]] = field(default_factory=list)
    delete_calls: list[tuple[str, ...]] = field(default_factory=list)

    async def get(self, name: str) -> bytes | str | None:
        self.get_calls.append(name)
        return self.values.get(name)

    async def set(
        self,
        name: str,
        value: str,
        *,
        ex: int | None = None,
    ) -> object:
        self.values[name] = value
        self.expiries[name] = ex
        self.set_calls.append((name, value, ex))
        return True

    async def delete(self, *names: str) -> object:
        self.delete_calls.append(names)
        for name in names:
            self.values.pop(name, None)
            self.expiries.pop(name, None)
        return len(names)


def snapshot() -> ConversationContextSnapshot:
    conversation_id = uuid4()
    user_id = uuid4()
    message = ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=10,
        role=ConversationMemoryRole.USER,
        text="hello",
        created_at=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
    )
    return ConversationContextSnapshot(
        conversation_id=conversation_id,
        user_id=user_id,
        summary="compressed context",
        recent_messages=[message],
        last_compacted_message_id=message.id,
        last_seen_message_id=message.id,
        last_compacted_sequence=8,
        last_seen_sequence=10,
        version=2,
        token_count=100,
        updated_at=datetime(2026, 5, 30, 12, 1, tzinfo=UTC),
    )


async def test_redis_snapshot_store_saves_and_loads_snapshot_with_ttl() -> None:
    client = FakeRedisClient()
    store: ConversationContextSnapshotStore = RedisConversationContextSnapshotStore(
        client,
        ttl_seconds=60,
    )
    stored_snapshot = snapshot()

    await store.save_snapshot(snapshot=stored_snapshot)
    loaded = await store.get_snapshot(conversation_id=stored_snapshot.conversation_id)

    key = f"conversation_context:{stored_snapshot.conversation_id}"
    assert isinstance(store, ConversationContextSnapshotStore)
    assert client.set_calls[0][0] == key
    assert client.set_calls[0][2] == 60
    assert loaded == stored_snapshot
    assert loaded is not None
    assert loaded.last_compacted_sequence == 8
    assert loaded.last_seen_sequence == 10


async def test_redis_snapshot_store_loads_bytes_payload() -> None:
    client = FakeRedisClient()
    store = RedisConversationContextSnapshotStore(client)
    stored_snapshot = snapshot()
    key = store.key_for_conversation(stored_snapshot.conversation_id)
    client.values[key] = stored_snapshot.model_dump_json().encode("utf-8")

    loaded = await store.get_snapshot(conversation_id=stored_snapshot.conversation_id)

    assert loaded == stored_snapshot


async def test_redis_snapshot_store_returns_none_on_cache_miss() -> None:
    client = FakeRedisClient()
    store = RedisConversationContextSnapshotStore(client)
    conversation_id = uuid4()

    assert await store.get_snapshot(conversation_id=conversation_id) is None
    assert client.get_calls == [f"conversation_context:{conversation_id}"]


async def test_redis_snapshot_store_deletes_snapshot() -> None:
    client = FakeRedisClient()
    store = RedisConversationContextSnapshotStore(client)
    stored_snapshot = snapshot()
    await store.save_snapshot(snapshot=stored_snapshot)

    await store.delete_snapshot(conversation_id=stored_snapshot.conversation_id)

    key = f"conversation_context:{stored_snapshot.conversation_id}"
    assert client.delete_calls == [(key,)]
    assert key not in client.values


async def test_redis_snapshot_store_rejects_conversation_id_mismatch() -> None:
    client = FakeRedisClient()
    store = RedisConversationContextSnapshotStore(client)
    stored_snapshot = snapshot()
    requested_conversation_id = uuid4()
    client.values[store.key_for_conversation(requested_conversation_id)] = (
        stored_snapshot.model_dump_json()
    )

    with pytest.raises(RedisSnapshotError):
        await store.get_snapshot(conversation_id=requested_conversation_id)


def test_redis_snapshot_store_validates_configuration() -> None:
    client = FakeRedisClient()

    with pytest.raises(ValueError):
        RedisConversationContextSnapshotStore(client, ttl_seconds=0)

    with pytest.raises(ValueError):
        RedisConversationContextSnapshotStore(client, key_prefix="")
