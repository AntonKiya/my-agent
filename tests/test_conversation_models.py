from datetime import UTC
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_service.conversations import (
    Conversation,
    ConversationLookup,
    ConversationStatus,
    ConversationType,
    ObservedConversation,
)


def test_conversation_defaults_to_active_private_with_isolated_metadata() -> None:
    user_id = uuid4()
    first = Conversation(
        user_id=user_id,
        channel="telegram",
        conversation_key="telegram:private:12345",
        external_chat_id="12345",
    )
    second = Conversation(
        user_id=user_id,
        channel="telegram",
        conversation_key="telegram:private:67890",
        external_chat_id="67890",
    )

    first.metadata["topic"] = "support"

    assert first.status is ConversationStatus.ACTIVE
    assert first.type is ConversationType.PRIVATE
    assert first.created_at.tzinfo is UTC
    assert second.metadata == {}


def test_observed_conversation_builds_lookup_from_conversation_key_only() -> None:
    observed = ObservedConversation(
        user_id=uuid4(),
        channel="telegram",
        conversation_key="telegram:private:12345",
        external_chat_id="12345",
        metadata={"source": "private"},
    )

    lookup = observed.lookup()

    assert lookup == ConversationLookup(conversation_key="telegram:private:12345")
    assert "user_id" not in lookup.model_dump()
    assert observed.observed_at.tzinfo is UTC


def test_conversation_model_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Conversation.model_validate(
            {
                "user_id": str(uuid4()),
                "channel": "telegram",
                "conversation_key": "telegram:private:12345",
                "external_chat_id": "12345",
                "telegram_user_id": "67890",
            }
        )


def test_conversation_requires_non_empty_conversation_key() -> None:
    with pytest.raises(ValidationError):
        Conversation(
            user_id=uuid4(),
            channel="telegram",
            conversation_key="",
            external_chat_id="12345",
        )
