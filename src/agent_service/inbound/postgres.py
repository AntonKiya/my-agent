import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

import asyncpg

from agent_service.channels.models import InboundEvent, InboundEventStatus
from agent_service.inbound.idempotency import (
    InboundIdempotencyClaim,
    InboundIdempotencyStore,
)

INSERT_INBOUND_EVENT_PROCESSING_SQL = """
INSERT INTO inbound_event_processing (
    event_id,
    channel,
    idempotency_key,
    external_update_id,
    external_chat_id,
    external_message_id,
    user_id,
    status,
    trace_id,
    metadata,
    first_received_at,
    last_seen_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12, $13)
"""

FETCH_DUPLICATE_INBOUND_EVENT_SQL = """
SELECT event_id, status
FROM inbound_event_processing
WHERE idempotency_key = $1
   OR (
        $3::text IS NOT NULL
        AND channel = $2
        AND external_update_id = $3
   )
ORDER BY first_received_at ASC
LIMIT 1
"""

UPDATE_DUPLICATE_LAST_SEEN_SQL = """
UPDATE inbound_event_processing
SET last_seen_at = $2,
    updated_at = $2
WHERE event_id = $1
"""

RELEASE_INBOUND_EVENT_CLAIM_SQL = """
DELETE FROM inbound_event_processing
WHERE event_id = $1
  AND status = 'queued'
"""

UPDATE_INBOUND_EVENT_STATUS_SQL = """
UPDATE inbound_event_processing
SET status = $2,
    failure_reason = COALESCE($3, failure_reason),
    processing_started_at = CASE
        WHEN $2 = 'processing' AND processing_started_at IS NULL THEN $4
        ELSE processing_started_at
    END,
    completed_at = CASE
        WHEN $2 IN ('completed', 'fallback_sent', 'dead_letter') THEN $4
        ELSE completed_at
    END,
    updated_at = $4
WHERE event_id = $1
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


class PostgresInboundIdempotencyStore(InboundIdempotencyStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def claim(self, event: InboundEvent) -> InboundIdempotencyClaim:
        if event.user_id is None:
            raise ValueError("Inbound event must be resolved before idempotency claim")

        now = datetime.now(UTC)
        async with self._pool.acquire() as connection:
            try:
                await connection.execute(
                    INSERT_INBOUND_EVENT_PROCESSING_SQL,
                    event.event_id,
                    event.channel,
                    event.idempotency_key,
                    event.external_update_id,
                    event.external_chat_id,
                    event.external_message_id,
                    event.user_id,
                    InboundEventStatus.QUEUED.value,
                    event.trace_id,
                    _jsonb(
                        {
                            "message_type": event.message_type.value,
                            "thread_id": event.thread_id,
                            "reply_to_message_id": event.reply_to_message_id,
                        }
                    ),
                    event.received_at,
                    now,
                    now,
                )
            except asyncpg.UniqueViolationError:
                duplicate = await _fetch_duplicate(connection, event)
                if duplicate is None:
                    raise
                await connection.execute(
                    UPDATE_DUPLICATE_LAST_SEEN_SQL,
                    duplicate.existing_event_id,
                    now,
                )
                return duplicate

        return InboundIdempotencyClaim(claimed=True, event_id=event.event_id)

    async def release_claim(self, *, event_id: UUID) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(RELEASE_INBOUND_EVENT_CLAIM_SQL, event_id)

    async def mark_status(
        self,
        *,
        event_id: UUID,
        status: InboundEventStatus,
        failure_reason: str | None = None,
    ) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(
                UPDATE_INBOUND_EVENT_STATUS_SQL,
                event_id,
                status.value,
                failure_reason,
                datetime.now(UTC),
            )


async def _fetch_duplicate(
    connection: PostgresConnection,
    event: InboundEvent,
) -> InboundIdempotencyClaim | None:
    row = await connection.fetchrow(
        FETCH_DUPLICATE_INBOUND_EVENT_SQL,
        event.idempotency_key,
        event.channel,
        event.external_update_id,
    )
    if row is None:
        return None
    existing_event_id = row["event_id"]
    status = row["status"]
    if not isinstance(existing_event_id, UUID) or not isinstance(status, str):
        raise TypeError("Inbound idempotency duplicate row has invalid shape")
    return InboundIdempotencyClaim(
        claimed=False,
        event_id=event.event_id,
        existing_event_id=existing_event_id,
        existing_status=InboundEventStatus(status),
    )


def _jsonb(value: object) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))
