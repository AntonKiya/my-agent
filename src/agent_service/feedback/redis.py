import json
from datetime import datetime
from typing import Protocol
from uuid import UUID

from agent_service.feedback.interfaces import FeedbackStateStore
from agent_service.feedback.models import PendingFeedback

DEFAULT_FEEDBACK_PENDING_TTL_SECONDS = 24 * 60 * 60
DEFAULT_FEEDBACK_PENDING_KEY_PREFIX = "feedback:pending"


class FeedbackStateError(Exception):
    """Raised when pending feedback state is invalid or unsafe to use."""


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


class RedisFeedbackStateStore(FeedbackStateStore):
    def __init__(
        self,
        client: RedisClient,
        *,
        ttl_seconds: int = DEFAULT_FEEDBACK_PENDING_TTL_SECONDS,
        key_prefix: str = DEFAULT_FEEDBACK_PENDING_KEY_PREFIX,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("Redis feedback pending ttl_seconds must be greater than zero")
        if not key_prefix:
            raise ValueError("Redis feedback pending key_prefix must not be empty")
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    async def get_pending(self, *, conversation_id: UUID) -> PendingFeedback | None:
        value = await self._client.get(self._key(conversation_id))
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise FeedbackStateError("Redis pending feedback payload is not an object")
        pending = PendingFeedback(
            user_id=UUID(str(payload["user_id"])),
            conversation_id=UUID(str(payload["conversation_id"])),
            channel=str(payload["channel"]),
            request_inbound_event_id=UUID(str(payload["request_inbound_event_id"])),
            requested_at=datetime.fromisoformat(str(payload["requested_at"])),
        )
        if pending.conversation_id != conversation_id:
            raise FeedbackStateError("Redis pending feedback conversation_id mismatch")
        return pending

    async def set_pending(self, *, pending: PendingFeedback) -> None:
        payload = {
            "user_id": str(pending.user_id),
            "conversation_id": str(pending.conversation_id),
            "channel": pending.channel,
            "request_inbound_event_id": str(pending.request_inbound_event_id),
            "requested_at": pending.requested_at.isoformat(),
        }
        await self._client.set(
            self._key(pending.conversation_id),
            json.dumps(payload, separators=(",", ":")),
            ex=self._ttl_seconds,
        )

    async def clear_pending(self, *, conversation_id: UUID) -> None:
        await self._client.delete(self._key(conversation_id))

    def key_for_conversation(self, conversation_id: UUID) -> str:
        return self._key(conversation_id)

    def _key(self, conversation_id: UUID) -> str:
        return f"{self._key_prefix}:{conversation_id}"
