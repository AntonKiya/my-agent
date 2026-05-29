import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

import asyncpg

from agent_service.conversations.errors import ConversationError, ConversationOwnershipError
from agent_service.conversations.interfaces import ConversationStore
from agent_service.conversations.models import (
    Conversation,
    ConversationLookup,
    ConversationMetadata,
    ConversationStatus,
    ConversationType,
    ObservedConversation,
)

CONVERSATION_SELECT = """
SELECT
    id,
    user_id,
    channel,
    conversation_key,
    external_chat_id,
    type,
    thread_id,
    status,
    metadata,
    created_at,
    updated_at
FROM conversations
WHERE conversation_key = $1
"""

INSERT_CONVERSATION_SQL = """
INSERT INTO conversations (
    id,
    user_id,
    channel,
    conversation_key,
    external_chat_id,
    type,
    thread_id,
    status,
    metadata,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11)
"""

UPDATE_CONVERSATION_SEEN_SQL = """
UPDATE conversations
SET
    metadata = $3::jsonb,
    updated_at = $4
WHERE id = $1
  AND user_id = $2
"""


class PostgresConnection(Protocol):
    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        """Fetch one row from Postgres."""
        ...

    async def execute(self, query: str, *args: object) -> str:
        """Execute a SQL command against Postgres."""
        ...

    def transaction(self) -> AbstractAsyncContextManager[object]:
        """Open a transaction on this connection."""
        ...


class PostgresPool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire a Postgres connection from a pool."""
        ...


class PostgresConversationStore(ConversationStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def get_by_key(
        self,
        *,
        lookup: ConversationLookup,
    ) -> Conversation | None:
        async with self._pool.acquire() as connection:
            return await self._fetch_conversation(connection, lookup)

    async def get_or_create_conversation(
        self,
        *,
        conversation: ObservedConversation,
    ) -> Conversation:
        lookup = conversation.lookup()
        async with self._pool.acquire() as connection:
            try:
                async with connection.transaction():
                    existing = await self._fetch_conversation(connection, lookup)
                    if existing is not None:
                        return await self._update_seen_on_connection(
                            connection=connection,
                            stored=existing,
                            observed=conversation,
                        )
                    return await self._create_on_connection(
                        connection=connection,
                        observed=conversation,
                    )
            except asyncpg.UniqueViolationError:
                existing = await self._fetch_conversation(connection, lookup)
                if existing is None:
                    raise
                return await self._update_seen_on_connection(
                    connection=connection,
                    stored=existing,
                    observed=conversation,
                )

    async def _fetch_conversation(
        self,
        connection: PostgresConnection,
        lookup: ConversationLookup,
    ) -> Conversation | None:
        row = await connection.fetchrow(CONVERSATION_SELECT, lookup.conversation_key)
        if row is None:
            return None
        return _conversation_from_row(row)

    async def _create_on_connection(
        self,
        *,
        connection: PostgresConnection,
        observed: ObservedConversation,
    ) -> Conversation:
        now = observed.observed_at
        conversation = Conversation(
            id=uuid4(),
            user_id=observed.user_id,
            channel=observed.channel,
            conversation_key=observed.conversation_key,
            external_chat_id=observed.external_chat_id,
            type=observed.type,
            thread_id=observed.thread_id,
            status=ConversationStatus.ACTIVE,
            metadata=dict(observed.metadata),
            created_at=now,
            updated_at=now,
        )
        await connection.execute(
            INSERT_CONVERSATION_SQL,
            conversation.id,
            conversation.user_id,
            conversation.channel,
            conversation.conversation_key,
            conversation.external_chat_id,
            conversation.type.value,
            conversation.thread_id,
            conversation.status.value,
            _jsonb(conversation.metadata),
            conversation.created_at,
            conversation.updated_at,
        )
        return conversation

    async def _update_seen_on_connection(
        self,
        *,
        connection: PostgresConnection,
        stored: Conversation,
        observed: ObservedConversation,
    ) -> Conversation:
        _validate_existing_conversation(stored=stored, observed=observed)
        await connection.execute(
            UPDATE_CONVERSATION_SEEN_SQL,
            stored.id,
            stored.user_id,
            _jsonb(observed.metadata),
            observed.observed_at,
        )
        return Conversation(
            id=stored.id,
            user_id=stored.user_id,
            channel=stored.channel,
            conversation_key=stored.conversation_key,
            external_chat_id=stored.external_chat_id,
            type=stored.type,
            thread_id=stored.thread_id,
            status=stored.status,
            metadata=dict(observed.metadata),
            created_at=stored.created_at,
            updated_at=observed.observed_at,
        )


def _validate_existing_conversation(
    *,
    stored: Conversation,
    observed: ObservedConversation,
) -> None:
    if stored.user_id != observed.user_id:
        raise ConversationOwnershipError("Conversation key belongs to a different user")
    if stored.channel != observed.channel:
        raise ConversationOwnershipError("Observed conversation channel does not match stored one")
    if stored.conversation_key != observed.conversation_key:
        raise ConversationOwnershipError("Observed conversation key does not match stored one")
    if stored.external_chat_id != observed.external_chat_id:
        raise ConversationOwnershipError("Observed external_chat_id does not match stored one")
    if stored.type != observed.type:
        raise ConversationOwnershipError("Observed conversation type does not match stored one")
    if stored.thread_id != observed.thread_id:
        raise ConversationOwnershipError("Observed thread_id does not match stored one")


def _conversation_from_row(row: Mapping[str, object]) -> Conversation:
    return Conversation(
        id=_uuid(row["id"]),
        user_id=_uuid(row["user_id"]),
        channel=str(row["channel"]),
        conversation_key=str(row["conversation_key"]),
        external_chat_id=str(row["external_chat_id"]),
        type=ConversationType(str(row["type"])),
        thread_id=_optional_str(row["thread_id"]),
        status=ConversationStatus(str(row["status"])),
        metadata=_metadata(row["metadata"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _jsonb(metadata: ConversationMetadata) -> str:
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True)


def _metadata(value: object) -> ConversationMetadata:
    if isinstance(value, str):
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ConversationError("Postgres metadata value must decode to an object")
        return cast(ConversationMetadata, decoded)
    if isinstance(value, dict):
        return cast(ConversationMetadata, dict(value))
    raise ConversationError("Postgres metadata value must be a JSON object")


def _uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise ConversationError("Postgres UUID value has unexpected type")


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise ConversationError("Postgres datetime value has unexpected type")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
