import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol
from uuid import UUID

from agent_service.media.interfaces import MediaStorageError
from agent_service.media.models import MediaAsset, MediaAssetType

INSERT_MEDIA_ASSET_SQL = """
INSERT INTO media_assets (
    media_id,
    user_id,
    conversation_id,
    media_type,
    storage_key,
    content_type,
    size_bytes,
    source_channel,
    source_attachment_id,
    source_inbound_event_id,
    metadata,
    created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12)
"""

GET_MEDIA_ASSET_SQL = """
SELECT
    media_id,
    user_id,
    conversation_id,
    media_type,
    storage_key,
    content_type,
    size_bytes,
    source_channel,
    source_attachment_id,
    source_inbound_event_id,
    metadata,
    created_at
FROM media_assets
WHERE media_id = $1
  AND user_id = $2
  AND conversation_id = $3
LIMIT 1
"""


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


class PostgresMediaAssetStore:
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create(self, *, asset: MediaAsset) -> MediaAsset:
        async with self._pool.acquire() as connection:
            await connection.execute(
                INSERT_MEDIA_ASSET_SQL,
                asset.media_id,
                asset.user_id,
                asset.conversation_id,
                asset.media_type.value,
                asset.storage_key,
                asset.content_type,
                asset.size_bytes,
                asset.source_channel,
                asset.source_attachment_id,
                asset.source_inbound_event_id,
                _jsonb(asset.metadata),
                asset.created_at,
            )
        return asset

    async def get(
        self,
        *,
        media_id: str,
        user_id: UUID,
        conversation_id: UUID,
    ) -> MediaAsset | None:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                GET_MEDIA_ASSET_SQL,
                media_id,
                user_id,
                conversation_id,
            )
        if row is None:
            return None
        return _media_asset(row)


def _media_asset(row: Mapping[str, object]) -> MediaAsset:
    media_id = row["media_id"]
    user_id = row["user_id"]
    conversation_id = row["conversation_id"]
    media_type = row["media_type"]
    storage_key = row["storage_key"]
    size_bytes = row["size_bytes"]
    source_channel = row["source_channel"]
    created_at = row["created_at"]
    if not isinstance(media_id, str):
        raise MediaStorageError("Media asset row has invalid media_id")
    if not isinstance(user_id, UUID):
        raise MediaStorageError("Media asset row has invalid user_id")
    if not isinstance(conversation_id, UUID):
        raise MediaStorageError("Media asset row has invalid conversation_id")
    if not isinstance(media_type, str):
        raise MediaStorageError("Media asset row has invalid media_type")
    if not isinstance(storage_key, str):
        raise MediaStorageError("Media asset row has invalid storage_key")
    if not isinstance(size_bytes, int):
        raise MediaStorageError("Media asset row has invalid size_bytes")
    if not isinstance(source_channel, str):
        raise MediaStorageError("Media asset row has invalid source_channel")
    if not isinstance(created_at, datetime):
        raise MediaStorageError("Media asset row has invalid created_at")
    return MediaAsset(
        media_id=media_id,
        user_id=user_id,
        conversation_id=conversation_id,
        media_type=MediaAssetType(media_type),
        storage_key=storage_key,
        content_type=_optional_str(row.get("content_type")),
        size_bytes=size_bytes,
        source_channel=source_channel,
        source_attachment_id=_optional_str(row.get("source_attachment_id")),
        source_inbound_event_id=_optional_uuid(row.get("source_inbound_event_id")),
        metadata=_metadata(row.get("metadata")),
        created_at=created_at,
    )


def _optional_str(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise MediaStorageError("Media asset row has invalid optional string value")


def _optional_uuid(value: object) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    raise MediaStorageError("Media asset row has invalid optional UUID value")


def _metadata(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    raise MediaStorageError("Media asset row has invalid metadata")


def _jsonb(value: object) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))
