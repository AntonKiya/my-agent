import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from agent_service.conversations.errors import ConversationLockTimeoutError


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ConversationLockLease:
    conversation_id: UUID
    acquired_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class _ConversationLockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ref_count: int = 0


class AsyncioConversationLockManager:
    def __init__(self) -> None:
        self._entries: dict[UUID, _ConversationLockEntry] = {}
        self._guard = asyncio.Lock()

    @property
    def tracked_lock_count(self) -> int:
        return len(self._entries)

    @asynccontextmanager
    async def acquire(
        self,
        conversation_id: UUID,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ConversationLockLease]:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Conversation lock timeout must be greater than zero")

        entry = await self._retain(conversation_id)
        acquired = False
        try:
            if timeout_seconds is None:
                await entry.lock.acquire()
            else:
                await asyncio.wait_for(entry.lock.acquire(), timeout=timeout_seconds)
            acquired = True
            yield ConversationLockLease(conversation_id=conversation_id)
        except TimeoutError as exc:
            raise ConversationLockTimeoutError(
                f"Timed out acquiring conversation lock for {conversation_id}"
            ) from exc
        finally:
            if acquired:
                entry.lock.release()
            await asyncio.shield(self._release(conversation_id, entry))

    async def _retain(self, conversation_id: UUID) -> _ConversationLockEntry:
        async with self._guard:
            entry = self._entries.get(conversation_id)
            if entry is None:
                entry = _ConversationLockEntry()
                self._entries[conversation_id] = entry
            entry.ref_count += 1
            return entry

    async def _release(
        self,
        conversation_id: UUID,
        entry: _ConversationLockEntry,
    ) -> None:
        async with self._guard:
            entry.ref_count -= 1
            if entry.ref_count < 0:
                raise RuntimeError("Conversation lock reference count became negative")
            if entry.ref_count == 0 and not entry.lock.locked():
                if self._entries.get(conversation_id) is entry:
                    del self._entries[conversation_id]
