from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable
from uuid import UUID

from agent_service.conversations.locks import ConversationLockLease
from agent_service.conversations.models import (
    Conversation,
    ConversationLookup,
    ObservedConversation,
)


@runtime_checkable
class ConversationStore(Protocol):
    async def get_by_key(
        self,
        *,
        lookup: ConversationLookup,
    ) -> Conversation | None:
        """Load a conversation by its stable derived key."""
        ...

    async def get_or_create_conversation(
        self,
        *,
        conversation: ObservedConversation,
    ) -> Conversation:
        """Load or atomically create a conversation without crossing user boundaries."""
        ...


@runtime_checkable
class ConversationLockManager(Protocol):
    def acquire(
        self,
        conversation_id: UUID,
        *,
        timeout_seconds: float | None = None,
    ) -> AbstractAsyncContextManager[ConversationLockLease]:
        """Acquire exclusive processing rights for one conversation."""
        ...
