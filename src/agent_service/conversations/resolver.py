from typing import Protocol, runtime_checkable

from agent_service.channels.models import InboundEvent
from agent_service.conversations.errors import (
    ConversationResolutionError,
    UnsupportedConversationChannelError,
)
from agent_service.conversations.interfaces import ConversationStore
from agent_service.conversations.models import (
    Conversation,
    ConversationMetadata,
    ConversationType,
    ObservedConversation,
)

TELEGRAM_CHANNEL = "telegram"
TELEGRAM_GROUP_CHAT_TYPES = {"group", "supergroup"}


@runtime_checkable
class ConversationResolverProtocol(Protocol):
    async def resolve(self, event: InboundEvent) -> Conversation:
        """Resolve the internal conversation for a user-resolved inbound event."""
        ...


class ConversationResolver(ConversationResolverProtocol):
    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    async def resolve(self, event: InboundEvent) -> Conversation:
        observed = observed_conversation_from_event(event)
        return await self._store.get_or_create_conversation(conversation=observed)


def observed_conversation_from_event(event: InboundEvent) -> ObservedConversation:
    if event.user_id is None:
        raise ConversationResolutionError(
            "Inbound event must have user_id before conversation resolution"
        )
    if event.channel == TELEGRAM_CHANNEL:
        return _telegram_conversation_from_event(event)
    raise UnsupportedConversationChannelError(
        f"Conversation resolution does not support channel {event.channel!r}"
    )


def _telegram_conversation_from_event(event: InboundEvent) -> ObservedConversation:
    if event.user_id is None:
        raise ConversationResolutionError(
            "Inbound event must have user_id before conversation resolution"
        )
    scope = _telegram_scope(event)
    conversation_type = _conversation_type(scope=scope, thread_id=event.thread_id)
    return ObservedConversation(
        user_id=event.user_id,
        channel=event.channel,
        conversation_key=_conversation_key(
            channel=event.channel,
            scope=scope,
            external_chat_id=event.external_chat_id,
            thread_id=event.thread_id,
        ),
        external_chat_id=event.external_chat_id,
        type=conversation_type,
        thread_id=event.thread_id,
        metadata=_conversation_metadata(event, scope),
        observed_at=event.received_at,
    )


def _telegram_scope(event: InboundEvent) -> str:
    chat_type = event.channel_metadata.get("chat_type")
    if chat_type in TELEGRAM_GROUP_CHAT_TYPES:
        return "group"
    return "private"


def _conversation_type(*, scope: str, thread_id: str | None) -> ConversationType:
    if thread_id is not None:
        return ConversationType.THREAD
    if scope == "group":
        return ConversationType.GROUP
    return ConversationType.PRIVATE


def _conversation_key(
    *,
    channel: str,
    scope: str,
    external_chat_id: str,
    thread_id: str | None,
) -> str:
    base_key = f"{channel}:{scope}:{external_chat_id}"
    if thread_id is not None:
        return f"{base_key}:thread:{thread_id}"
    return base_key


def _conversation_metadata(event: InboundEvent, scope: str) -> ConversationMetadata:
    metadata: ConversationMetadata = {"scope": scope}
    chat_type = event.channel_metadata.get("chat_type")
    if isinstance(chat_type, str):
        metadata["chat_type"] = chat_type
    return metadata
