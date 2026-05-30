import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol, cast
from uuid import UUID

from agent_service.channels.models import Attachment
from agent_service.memory.interfaces import ConversationMemoryStore
from agent_service.memory.models import (
    ConversationMemoryMessage,
    ConversationMemoryRole,
    MemoryMetadata,
)

NEXT_MESSAGE_SEQUENCE_SQL = """
UPDATE conversations
SET message_sequence = message_sequence + 1
WHERE id = $1
  AND user_id = $2
RETURNING message_sequence
"""

INSERT_MESSAGE_SQL = """
INSERT INTO conversation_messages (
    id,
    conversation_id,
    user_id,
    sequence,
    role,
    text,
    attachments,
    tool_name,
    tool_call_id,
    inbound_event_id,
    outbound_event_id,
    trace_id,
    token_count,
    metadata,
    created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13, $14::jsonb, $15)
"""

RECENT_MESSAGES_SQL = """
SELECT
    id,
    conversation_id,
    user_id,
    sequence,
    role,
    text,
    attachments,
    tool_name,
    tool_call_id,
    inbound_event_id,
    outbound_event_id,
    trace_id,
    token_count,
    metadata,
    created_at
FROM (
    SELECT *
    FROM conversation_messages
    WHERE conversation_id = $1
    ORDER BY sequence DESC
    LIMIT $2
) AS recent
ORDER BY sequence ASC
"""

CURRENT_MESSAGE_SEQUENCE_SQL = """
SELECT message_sequence
FROM conversations
WHERE id = $1
  AND user_id = $2
"""


class PostgresMemoryError(Exception):
    """Raised when persisted conversation memory cannot be read or written safely."""


class PostgresConnection(Protocol):
    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        """Fetch one row from Postgres."""
        ...

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        """Fetch rows from Postgres."""
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


class PostgresConversationMemoryStore(ConversationMemoryStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def append_message(
        self,
        *,
        message: ConversationMemoryMessage,
    ) -> ConversationMemoryMessage:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                sequence = await self._next_sequence(connection, message)
                stored = message.model_copy(update={"sequence": sequence})
                await connection.execute(
                    INSERT_MESSAGE_SQL,
                    stored.id,
                    stored.conversation_id,
                    stored.user_id,
                    stored.sequence,
                    stored.role.value,
                    stored.text,
                    _attachments_jsonb(stored.attachments),
                    stored.tool_name,
                    stored.tool_call_id,
                    stored.inbound_event_id,
                    stored.outbound_event_id,
                    stored.trace_id,
                    stored.token_count,
                    _jsonb(stored.metadata),
                    stored.created_at,
                )
                return stored

    async def list_recent_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int,
    ) -> list[ConversationMemoryMessage]:
        if limit < 1:
            raise ValueError("Recent message limit must be greater than zero")
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(RECENT_MESSAGES_SQL, conversation_id, limit)
        return [_message_from_row(row) for row in rows]

    async def current_message_sequence(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> int:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                CURRENT_MESSAGE_SEQUENCE_SQL,
                conversation_id,
                user_id,
            )
        if row is None:
            raise PostgresMemoryError("Conversation was not found for sequence read")
        sequence = row.get("message_sequence")
        if isinstance(sequence, int):
            return sequence
        raise PostgresMemoryError("Postgres message_sequence value has unexpected type")

    async def _next_sequence(
        self,
        connection: PostgresConnection,
        message: ConversationMemoryMessage,
    ) -> int:
        row = await connection.fetchrow(
            NEXT_MESSAGE_SEQUENCE_SQL,
            message.conversation_id,
            message.user_id,
        )
        if row is None:
            raise PostgresMemoryError("Conversation was not found for message append")
        sequence = row.get("message_sequence")
        if isinstance(sequence, int):
            return sequence
        raise PostgresMemoryError("Postgres message_sequence value has unexpected type")


def _message_from_row(row: Mapping[str, object]) -> ConversationMemoryMessage:
    return ConversationMemoryMessage(
        id=_uuid(row["id"]),
        conversation_id=_uuid(row["conversation_id"]),
        user_id=_uuid(row["user_id"]),
        sequence=_int(row["sequence"]),
        role=ConversationMemoryRole(str(row["role"])),
        text=_optional_str(row["text"]),
        attachments=_attachments(row["attachments"]),
        tool_name=_optional_str(row["tool_name"]),
        tool_call_id=_optional_str(row["tool_call_id"]),
        inbound_event_id=_optional_uuid(row["inbound_event_id"]),
        outbound_event_id=_optional_uuid(row["outbound_event_id"]),
        trace_id=_optional_str(row["trace_id"]),
        token_count=_optional_int(row["token_count"]),
        metadata=_metadata(row["metadata"]),
        created_at=_datetime(row["created_at"]),
    )


def _attachments_jsonb(attachments: list[Attachment]) -> str:
    return json.dumps(
        [attachment.model_dump(mode="json") for attachment in attachments],
        separators=(",", ":"),
        sort_keys=True,
    )


def _attachments(value: object) -> list[Attachment]:
    if isinstance(value, str):
        decoded = json.loads(value)
    else:
        decoded = value
    if not isinstance(decoded, list):
        raise PostgresMemoryError("Postgres attachments value must be a JSON array")
    return [Attachment.model_validate(item) for item in decoded]


def _jsonb(metadata: MemoryMetadata) -> str:
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True)


def _metadata(value: object) -> MemoryMetadata:
    if isinstance(value, str):
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise PostgresMemoryError("Postgres metadata value must decode to an object")
        return cast(MemoryMetadata, decoded)
    if isinstance(value, dict):
        return cast(MemoryMetadata, dict(value))
    raise PostgresMemoryError("Postgres metadata value must be a JSON object")


def _uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise PostgresMemoryError("Postgres UUID value has unexpected type")


def _optional_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    return _uuid(value)


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise PostgresMemoryError("Postgres datetime value has unexpected type")


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    raise PostgresMemoryError("Postgres integer value has unexpected type")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
