import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from agent_service.feedback.interfaces import FeedbackStore
from agent_service.feedback.models import FeedbackEntry

INSERT_FEEDBACK_SQL = """
INSERT INTO feedback (
    id,
    user_id,
    conversation_id,
    source_channel,
    source_external_user_id,
    source_external_chat_id,
    source_thread_id,
    source_inbound_event_id,
    request_inbound_event_id,
    text,
    metadata,
    created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12)
ON CONFLICT (source_inbound_event_id) DO NOTHING
"""


class PostgresConnection(Protocol):
    async def execute(self, query: str, *args: object) -> str:
        """Execute a SQL command against Postgres."""
        ...


class PostgresPool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire a Postgres connection from a pool."""
        ...


class PostgresFeedbackStore(FeedbackStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create(self, *, feedback: FeedbackEntry) -> FeedbackEntry:
        async with self._pool.acquire() as connection:
            await connection.execute(
                INSERT_FEEDBACK_SQL,
                feedback.id,
                feedback.user_id,
                feedback.conversation_id,
                feedback.source_channel,
                feedback.source_external_user_id,
                feedback.source_external_chat_id,
                feedback.source_thread_id,
                feedback.source_inbound_event_id,
                feedback.request_inbound_event_id,
                feedback.text,
                _jsonb(feedback.metadata),
                feedback.created_at,
            )
        return feedback


def _jsonb(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
