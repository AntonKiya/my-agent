from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agent_service.channels import InboundEvent
from agent_service.conversations import (
    Conversation,
    ConversationLookup,
    ConversationResolutionError,
    ConversationResolver,
    ConversationResolverProtocol,
    ConversationType,
    ObservedConversation,
    UnsupportedConversationChannelError,
    observed_conversation_from_event,
)


@dataclass(slots=True)
class FakeConversationStore:
    conversations: list[ObservedConversation] = field(default_factory=list)
    result: Conversation | None = None

    async def get_or_create_conversation(
        self,
        *,
        conversation: ObservedConversation,
    ) -> Conversation:
        self.conversations.append(conversation)
        if self.result is not None:
            return self.result
        return Conversation(
            user_id=conversation.user_id,
            channel=conversation.channel,
            conversation_key=conversation.conversation_key,
            external_chat_id=conversation.external_chat_id,
            type=conversation.type,
            thread_id=conversation.thread_id,
            metadata=dict(conversation.metadata),
            created_at=conversation.observed_at,
            updated_at=conversation.observed_at,
        )

    async def get_by_key(self, *, lookup: ConversationLookup) -> Conversation | None:
        return None


def inbound_event(
    *,
    channel: str = "telegram",
    resolved: bool = True,
    external_chat_id: str = "12345",
    thread_id: str | None = None,
    chat_type: str = "private",
) -> InboundEvent:
    return InboundEvent(
        channel=channel,
        external_user_id="67890",
        external_chat_id=external_chat_id,
        external_message_id="42",
        external_update_id="100",
        idempotency_key=f"{channel}:{external_chat_id}:42",
        user_id=uuid4() if resolved else None,
        text="hello",
        thread_id=thread_id,
        channel_metadata={"chat_type": chat_type},
        received_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )


def test_observed_conversation_from_telegram_private_event() -> None:
    event = inbound_event()

    observed = observed_conversation_from_event(event)

    assert event.user_id is not None
    assert observed.user_id == event.user_id
    assert observed.channel == "telegram"
    assert observed.conversation_key == "telegram:private:12345"
    assert observed.external_chat_id == "12345"
    assert observed.type is ConversationType.PRIVATE
    assert observed.thread_id is None
    assert observed.metadata == {"scope": "private", "chat_type": "private"}
    assert observed.observed_at == event.received_at


def test_observed_conversation_from_telegram_thread_uses_thread_key() -> None:
    event = inbound_event(thread_id="11", chat_type="supergroup")

    observed = observed_conversation_from_event(event)

    assert observed.conversation_key == "telegram:group:12345:thread:11"
    assert observed.type is ConversationType.THREAD
    assert observed.thread_id == "11"
    assert observed.metadata == {"scope": "group", "chat_type": "supergroup"}


def test_observed_conversation_from_telegram_group_without_thread_is_group() -> None:
    event = inbound_event(chat_type="group")

    observed = observed_conversation_from_event(event)

    assert observed.conversation_key == "telegram:group:12345"
    assert observed.type is ConversationType.GROUP


def test_observed_conversation_requires_resolved_user_id() -> None:
    event = inbound_event(resolved=False)

    with pytest.raises(ConversationResolutionError):
        observed_conversation_from_event(event)


def test_observed_conversation_rejects_unsupported_channel() -> None:
    event = inbound_event(channel="slack")

    with pytest.raises(UnsupportedConversationChannelError):
        observed_conversation_from_event(event)


async def test_conversation_resolver_delegates_to_store() -> None:
    store = FakeConversationStore()
    resolver: ConversationResolverProtocol = ConversationResolver(store)
    event = inbound_event()

    result = await resolver.resolve(event)

    assert isinstance(resolver, ConversationResolverProtocol)
    assert result.conversation_key == "telegram:private:12345"
    assert result.user_id == event.user_id
    assert len(store.conversations) == 1
    assert store.conversations[0].conversation_key == "telegram:private:12345"
