from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agent_service.agents import (
    AgentContext,
    AgentContextMessage,
    AgentContextRole,
    AgentResponse,
    PydanticAIRunContext,
)
from agent_service.channels import InboundEvent
from agent_service.conversations import Conversation
from agent_service.memory import (
    ConversationContextSnapshot,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationMemoryService,
    PreparedConversationContext,
)


class FakeConversationMemoryService:
    def __init__(self) -> None:
        self.user_messages: list[ConversationMemoryMessage] = []
        self.assistant_messages: list[ConversationMemoryMessage] = []

    async def record_user_message(
        self,
        *,
        conversation: Conversation,
        event: InboundEvent,
    ) -> ConversationMemoryMessage:
        if event.user_id is None:
            raise ValueError("event must be user-resolved")
        message = ConversationMemoryMessage(
            conversation_id=conversation.id,
            user_id=event.user_id,
            role=ConversationMemoryRole.USER,
            text=event.text,
            attachments=list(event.attachments),
            inbound_event_id=event.event_id,
            trace_id=event.trace_id,
            metadata={"channel": event.channel},
            created_at=event.received_at,
        )
        self.user_messages.append(message)
        return message

    async def prepare_agent_context(
        self,
        *,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> PreparedConversationContext:
        snapshot = ConversationContextSnapshot(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            summary="User prefers concise answers.",
            recent_messages=[latest_user_message],
            last_seen_message_id=latest_user_message.id,
            token_count=42,
            updated_at=datetime(2026, 5, 29, 12, 1, tzinfo=UTC),
        )
        return PreparedConversationContext(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            latest_user_message_id=latest_user_message.id,
            agent_context=AgentContext(
                system_prompt_parts=[snapshot.summary or ""],
                recent_messages=[
                    AgentContextMessage(
                        role=AgentContextRole.USER,
                        text=latest_user_message.text or "",
                        message_id=latest_user_message.id,
                    )
                ],
            ),
            pydantic_ai=PydanticAIRunContext(
                user_prompt=latest_user_message.text,
                message_history=[],
                conversation_id=str(conversation.id),
                instructions=snapshot.summary,
            ),
            snapshot=snapshot,
        )

    async def record_assistant_message(
        self,
        *,
        conversation: Conversation,
        response: AgentResponse,
        trace_id: str | None = None,
        outbound_event_id: UUID | None = None,
    ) -> ConversationMemoryMessage:
        message = ConversationMemoryMessage(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            role=ConversationMemoryRole.ASSISTANT,
            text=response.text,
            outbound_event_id=outbound_event_id,
            trace_id=trace_id or response.trace_id,
            metadata=response.metadata,
        )
        self.assistant_messages.append(message)
        return message


def conversation() -> Conversation:
    user_id = uuid4()
    return Conversation(
        id=uuid4(),
        user_id=user_id,
        channel="telegram",
        conversation_key="telegram:private:12345",
        external_chat_id="12345",
    )


def inbound_event(*, user_id: UUID, text: str = "hello") -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        external_message_id="42",
        idempotency_key="telegram:12345:42",
        user_id=user_id,
        text=text,
        trace_id="trace-1",
        received_at=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )


def test_conversation_memory_message_defaults_are_isolated() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    first = ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        role=ConversationMemoryRole.USER,
        text="first",
    )
    second = ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        role=ConversationMemoryRole.ASSISTANT,
        text="second",
    )

    first.metadata["key"] = "value"

    assert first.created_at.tzinfo is UTC
    assert second.metadata == {}


def test_context_snapshot_models_redis_working_state_shape() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    message = ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        role=ConversationMemoryRole.USER,
        text="hello",
        token_count=3,
    )

    snapshot = ConversationContextSnapshot(
        conversation_id=conversation_id,
        user_id=user_id,
        summary="compressed context",
        recent_messages=[message],
        last_compacted_message_id=message.id,
        last_seen_message_id=message.id,
        version=2,
        token_count=100,
    )

    assert snapshot.summary == "compressed context"
    assert snapshot.recent_messages == [message]
    assert snapshot.last_compacted_message_id == message.id
    assert snapshot.last_seen_message_id == message.id
    assert snapshot.version == 2
    assert snapshot.updated_at.tzinfo is UTC


def test_prepared_context_exposes_agent_and_pydantic_ai_shapes() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    latest_message_id = uuid4()

    prepared = PreparedConversationContext(
        conversation_id=conversation_id,
        user_id=user_id,
        latest_user_message_id=latest_message_id,
        agent_context=AgentContext(system_prompt_parts=["summary"]),
        pydantic_ai=PydanticAIRunContext(
            user_prompt="hello",
            message_history=[
                {
                    "kind": "response",
                    "parts": [{"part_kind": "text", "content": "previous answer"}],
                }
            ],
            conversation_id=str(conversation_id),
            instructions="summary",
        ),
    )

    assert prepared.agent_context.system_prompt_parts == ["summary"]
    assert prepared.pydantic_ai.user_prompt == "hello"
    assert prepared.pydantic_ai.conversation_id == str(conversation_id)
    assert prepared.pydantic_ai.instructions == "summary"


def test_memory_models_reject_unknown_fields_and_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        ConversationMemoryMessage.model_validate(
            {
                "conversation_id": str(uuid4()),
                "user_id": str(uuid4()),
                "role": "user",
                "text": "hello",
                "raw_update": {},
            }
        )

    with pytest.raises(ValidationError):
        ConversationContextSnapshot(
            conversation_id=uuid4(),
            user_id=uuid4(),
            token_count=-1,
        )


async def test_conversation_memory_service_protocol_flow() -> None:
    service = FakeConversationMemoryService()
    resolved_conversation = conversation()
    event = inbound_event(user_id=resolved_conversation.user_id)

    user_message = await service.record_user_message(
        conversation=resolved_conversation,
        event=event,
    )
    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=user_message,
    )
    assistant_message = await service.record_assistant_message(
        conversation=resolved_conversation,
        response=AgentResponse(text="answer", trace_id="trace-1"),
        outbound_event_id=uuid4(),
    )

    assert isinstance(service, ConversationMemoryService)
    assert user_message.role is ConversationMemoryRole.USER
    assert user_message.inbound_event_id == event.event_id
    assert prepared.latest_user_message_id == user_message.id
    assert prepared.pydantic_ai.user_prompt == "hello"
    assert assistant_message.role is ConversationMemoryRole.ASSISTANT
    assert assistant_message.text == "answer"
    assert service.user_messages == [user_message]
    assert service.assistant_messages == [assistant_message]
