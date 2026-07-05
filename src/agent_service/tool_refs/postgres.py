import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from agent_service.tool_refs.interfaces import ToolResultReferenceStore
from agent_service.tool_refs.models import ToolResultReference

INSERT_TOOL_RESULT_REFERENCE_SQL = """
INSERT INTO tool_result_references (
    selection_id,
    provider,
    source_tool_name,
    user_id,
    conversation_id,
    item_kind,
    item_index,
    label,
    display_snapshot,
    ref_payload,
    expires_at,
    created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11, $12)
"""

GET_TOOL_RESULT_REFERENCE_SQL = """
SELECT
    selection_id,
    provider,
    source_tool_name,
    user_id,
    conversation_id,
    item_kind,
    item_index,
    label,
    display_snapshot,
    ref_payload,
    expires_at,
    created_at
FROM tool_result_references
WHERE selection_id = $1
  AND user_id = $2
  AND conversation_id = $3
  AND ($4::text IS NULL OR provider = $4)
  AND expires_at > now()
LIMIT 1
"""


class ToolResultReferenceStorageError(Exception):
    """Raised when a persisted tool reference row is malformed."""


class PostgresConnection(Protocol):
    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        """Fetch one row from Postgres."""
        ...

    async def execute(self, query: str, *args: object) -> str:
        """Execute a SQL command against Postgres."""
        ...


class PostgresPool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire a Postgres connection from a pool."""
        ...


class PostgresToolResultReferenceStore(ToolResultReferenceStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create(self, *, reference: ToolResultReference) -> ToolResultReference:
        async with self._pool.acquire() as connection:
            await connection.execute(
                INSERT_TOOL_RESULT_REFERENCE_SQL,
                reference.selection_id,
                reference.provider,
                reference.source_tool_name,
                reference.user_id,
                reference.conversation_id,
                reference.item_kind,
                reference.item_index,
                reference.label,
                _jsonb(reference.display_snapshot),
                _jsonb(reference.ref_payload),
                reference.expires_at,
                reference.created_at,
            )
        return reference

    async def get(
        self,
        *,
        selection_id: str,
        user_id: UUID,
        conversation_id: UUID,
        provider: str | None = None,
    ) -> ToolResultReference | None:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                GET_TOOL_RESULT_REFERENCE_SQL,
                selection_id,
                user_id,
                conversation_id,
                provider,
            )
        if row is None:
            return None
        return _reference_from_row(row)


def _reference_from_row(row: Mapping[str, object]) -> ToolResultReference:
    selection_id = row["selection_id"]
    provider = row["provider"]
    source_tool_name = row["source_tool_name"]
    user_id = row["user_id"]
    conversation_id = row["conversation_id"]
    item_kind = row["item_kind"]
    item_index = row["item_index"]
    expires_at = row["expires_at"]
    created_at = row["created_at"]

    if not isinstance(selection_id, str):
        raise ToolResultReferenceStorageError("Tool reference row has invalid selection_id")
    if not isinstance(provider, str):
        raise ToolResultReferenceStorageError("Tool reference row has invalid provider")
    if not isinstance(source_tool_name, str):
        raise ToolResultReferenceStorageError("Tool reference row has invalid source_tool_name")
    if not isinstance(user_id, UUID):
        raise ToolResultReferenceStorageError("Tool reference row has invalid user_id")
    if not isinstance(conversation_id, UUID):
        raise ToolResultReferenceStorageError("Tool reference row has invalid conversation_id")
    if not isinstance(item_kind, str):
        raise ToolResultReferenceStorageError("Tool reference row has invalid item_kind")
    if not isinstance(item_index, int):
        raise ToolResultReferenceStorageError("Tool reference row has invalid item_index")
    if not isinstance(expires_at, datetime):
        raise ToolResultReferenceStorageError("Tool reference row has invalid expires_at")
    if not isinstance(created_at, datetime):
        raise ToolResultReferenceStorageError("Tool reference row has invalid created_at")

    return ToolResultReference(
        selection_id=selection_id,
        provider=provider,
        source_tool_name=source_tool_name,
        user_id=user_id,
        conversation_id=conversation_id,
        item_kind=item_kind,
        item_index=item_index,
        label=_optional_str(row.get("label")),
        display_snapshot=_json_object(row.get("display_snapshot"), "display_snapshot"),
        ref_payload=_json_object(row.get("ref_payload"), "ref_payload"),
        expires_at=expires_at,
        created_at=created_at,
    )


def _optional_str(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ToolResultReferenceStorageError("Tool reference row has invalid optional string value")


def _json_object(value: object, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    raise ToolResultReferenceStorageError(f"Tool reference row has invalid {field_name}")


def _jsonb(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
