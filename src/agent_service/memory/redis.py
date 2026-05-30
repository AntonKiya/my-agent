from typing import Protocol
from uuid import UUID

from agent_service.memory.interfaces import ConversationContextSnapshotStore
from agent_service.memory.models import ConversationContextSnapshot

DEFAULT_CONTEXT_SNAPSHOT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_CONTEXT_SNAPSHOT_KEY_PREFIX = "conversation_context"


class RedisSnapshotError(Exception):
    """Raised when a Redis context snapshot is invalid or unsafe to use."""


class RedisClient(Protocol):
    async def get(self, name: str) -> bytes | str | None:
        """Return a Redis string value."""
        ...

    async def set(
        self,
        name: str,
        value: str,
        *,
        ex: int | None = None,
    ) -> object:
        """Set a Redis string value with optional expiry seconds."""
        ...

    async def delete(self, *names: str) -> object:
        """Delete one or more Redis keys."""
        ...


class RedisConversationContextSnapshotStore(ConversationContextSnapshotStore):
    def __init__(
        self,
        client: RedisClient,
        *,
        ttl_seconds: int = DEFAULT_CONTEXT_SNAPSHOT_TTL_SECONDS,
        key_prefix: str = DEFAULT_CONTEXT_SNAPSHOT_KEY_PREFIX,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("Redis context snapshot ttl_seconds must be greater than zero")
        if not key_prefix:
            raise ValueError("Redis context snapshot key_prefix must not be empty")
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    async def get_snapshot(
        self,
        *,
        conversation_id: UUID,
    ) -> ConversationContextSnapshot | None:
        value = await self._client.get(self._key(conversation_id))
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        snapshot = ConversationContextSnapshot.model_validate_json(value)
        if snapshot.conversation_id != conversation_id:
            raise RedisSnapshotError("Redis context snapshot conversation_id mismatch")
        return snapshot

    async def save_snapshot(
        self,
        *,
        snapshot: ConversationContextSnapshot,
    ) -> None:
        await self._client.set(
            self._key(snapshot.conversation_id),
            snapshot.model_dump_json(),
            ex=self._ttl_seconds,
        )

    async def delete_snapshot(
        self,
        *,
        conversation_id: UUID,
    ) -> None:
        await self._client.delete(self._key(conversation_id))

    def key_for_conversation(self, conversation_id: UUID) -> str:
        return self._key(conversation_id)

    def _key(self, conversation_id: UUID) -> str:
        return f"{self._key_prefix}:{conversation_id}"
