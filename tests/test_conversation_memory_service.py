from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from agent_service.agents import AgentResponse, AgentToolInfo, AgentToolStatus, AgentUsage
from agent_service.channels import InboundEvent
from agent_service.conversations import Conversation
from agent_service.memory import (
    ConversationContextSnapshot,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationMemoryServiceError,
    DefaultConversationMemoryService,
)


@dataclass(slots=True)
class FakeMemoryStore:
    messages: dict[UUID, list[ConversationMemoryMessage]] = field(default_factory=dict)
    append_calls: list[ConversationMemoryMessage] = field(default_factory=list)
    recent_calls: list[tuple[UUID, int]] = field(default_factory=list)
    sequence_calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def append_message(
        self,
        *,
        message: ConversationMemoryMessage,
    ) -> ConversationMemoryMessage:
        current = await self.current_message_sequence(
            conversation_id=message.conversation_id,
            user_id=message.user_id,
        )
        stored = message.model_copy(update={"sequence": current + 1})
        self.messages.setdefault(stored.conversation_id, []).append(stored)
        self.append_calls.append(stored)
        return stored

    async def list_recent_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int,
    ) -> list[ConversationMemoryMessage]:
        self.recent_calls.append((conversation_id, limit))
        return self.messages.get(conversation_id, [])[-limit:]

    async def current_message_sequence(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> int:
        self.sequence_calls.append((conversation_id, user_id))
        sequences = [
            message.sequence or 0
            for message in self.messages.get(conversation_id, [])
            if message.user_id == user_id
        ]
        return max(sequences, default=0)


@dataclass(slots=True)
class FakeSnapshotStore:
    snapshots: dict[UUID, ConversationContextSnapshot] = field(default_factory=dict)
    get_calls: list[UUID] = field(default_factory=list)
    save_calls: list[ConversationContextSnapshot] = field(default_factory=list)
    delete_calls: list[UUID] = field(default_factory=list)

    async def get_snapshot(
        self,
        *,
        conversation_id: UUID,
    ) -> ConversationContextSnapshot | None:
        self.get_calls.append(conversation_id)
        return self.snapshots.get(conversation_id)

    async def save_snapshot(
        self,
        *,
        snapshot: ConversationContextSnapshot,
    ) -> None:
        self.snapshots[snapshot.conversation_id] = snapshot
        self.save_calls.append(snapshot)

    async def delete_snapshot(
        self,
        *,
        conversation_id: UUID,
    ) -> None:
        self.snapshots.pop(conversation_id, None)
        self.delete_calls.append(conversation_id)


def conversation() -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=uuid4(),
        channel="telegram",
        conversation_key="telegram:private:12345",
        external_chat_id="12345",
    )


def inbound_event(*, resolved_conversation: Conversation, text: str = "hello") -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id=resolved_conversation.external_chat_id,
        external_message_id="42",
        external_update_id="100",
        idempotency_key="telegram:12345:42",
        user_id=resolved_conversation.user_id,
        text=text,
        trace_id="trace-1",
        received_at=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
    )


def memory_message(
    *,
    resolved_conversation: Conversation,
    role: ConversationMemoryRole,
    sequence: int,
    text: str | None = "hello",
) -> ConversationMemoryMessage:
    return ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=sequence,
        role=role,
        text=text,
        created_at=datetime(2026, 5, 30, 12, sequence, tzinfo=UTC),
    )


async def test_memory_service_records_user_message_and_extends_fresh_snapshot() -> None:
    resolved_conversation = conversation()
    first_message = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=1,
        text="previous",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [first_message]})
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                recent_messages=[first_message],
                last_seen_message_id=first_message.id,
                last_seen_sequence=1,
            )
        }
    )
    service = DefaultConversationMemoryService(memory_store, snapshot_store)

    stored = await service.record_user_message(
        conversation=resolved_conversation,
        event=inbound_event(resolved_conversation=resolved_conversation),
    )

    assert stored.sequence == 2
    assert stored.role is ConversationMemoryRole.USER
    assert stored.metadata["idempotency_key"] == "telegram:12345:42"
    saved_snapshot = snapshot_store.save_calls[-1]
    assert saved_snapshot.last_seen_sequence == 2
    assert saved_snapshot.last_seen_message_id == stored.id
    assert [message.id for message in saved_snapshot.recent_messages] == [
        first_message.id,
        stored.id,
    ]


async def test_memory_service_uses_fresh_snapshot_without_reloading_history() -> None:
    resolved_conversation = conversation()
    previous = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=1,
        text="previous answer",
    )
    latest = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=2,
        text="latest",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [previous, latest]})
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                summary="compressed",
                recent_messages=[previous, latest],
                last_seen_message_id=latest.id,
                last_seen_sequence=2,
            )
        }
    )
    service = DefaultConversationMemoryService(memory_store, snapshot_store)

    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=latest,
    )

    assert memory_store.recent_calls == []
    assert prepared.snapshot is snapshot_store.snapshots[resolved_conversation.id]
    assert prepared.metadata["snapshot_source"] == "redis"
    assert prepared.agent_context.system_prompt_parts == ["compressed"]
    assert [message.message_id for message in prepared.agent_context.recent_messages] == [
        previous.id,
        latest.id,
    ]
    assert prepared.pydantic_ai.user_prompt == "latest"
    history_message = prepared.pydantic_ai.message_history[0]
    assert isinstance(history_message, ModelResponse)
    assert history_message.timestamp == previous.created_at
    assert len(history_message.parts) == 1
    assert isinstance(history_message.parts[0], TextPart)
    assert history_message.parts[0].content == "previous answer"


async def test_memory_service_rebuilds_and_replaces_stale_snapshot() -> None:
    resolved_conversation = conversation()
    old_message = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=1,
        text="old",
    )
    latest = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=2,
        text="latest",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [old_message, latest]})
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                recent_messages=[old_message],
                last_seen_message_id=old_message.id,
                last_seen_sequence=1,
            )
        }
    )
    service = DefaultConversationMemoryService(memory_store, snapshot_store)

    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=latest,
    )

    assert snapshot_store.delete_calls == [resolved_conversation.id]
    assert memory_store.recent_calls == [(resolved_conversation.id, 100)]
    assert snapshot_store.save_calls[-1].last_seen_sequence == 2
    assert prepared.snapshot == snapshot_store.save_calls[-1]
    assert prepared.metadata["snapshot_source"] == "postgres"
    assert prepared.pydantic_ai.user_prompt == "latest"


async def test_memory_service_rejects_latest_message_from_another_user() -> None:
    resolved_conversation = conversation()
    service = DefaultConversationMemoryService(FakeMemoryStore(), FakeSnapshotStore())
    latest = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=uuid4(),
        sequence=1,
        role=ConversationMemoryRole.USER,
        text="wrong user",
    )

    with pytest.raises(ConversationMemoryServiceError):
        await service.prepare_agent_context(
            conversation=resolved_conversation,
            latest_user_message=latest,
        )


async def test_memory_service_keeps_tool_messages_in_active_context() -> None:
    resolved_conversation = conversation()
    tool_call = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=1,
        role=ConversationMemoryRole.TOOL_CALL,
        tool_name="search",
        tool_call_id="call-1",
        metadata={"args": {"query": "weather"}},
    )
    tool_result = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=2,
        role=ConversationMemoryRole.TOOL_RESULT,
        text="sunny",
        tool_name="search",
        tool_call_id="call-1",
    )
    latest = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=3,
        text="thanks",
    )
    memory_store = FakeMemoryStore(
        messages={resolved_conversation.id: [tool_call, tool_result, latest]}
    )
    service = DefaultConversationMemoryService(memory_store, FakeSnapshotStore())

    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=latest,
    )

    assert [message.role.value for message in prepared.agent_context.recent_messages] == [
        "tool_call",
        "tool_result",
        "user",
    ]
    tool_call_message = prepared.pydantic_ai.message_history[0]
    tool_result_message = prepared.pydantic_ai.message_history[1]
    assert isinstance(tool_call_message, ModelResponse)
    assert isinstance(tool_call_message.parts[0], ToolCallPart)
    assert tool_call_message.parts[0].tool_name == "search"
    assert tool_call_message.parts[0].args == {"query": "weather"}
    assert isinstance(tool_result_message, ModelRequest)
    assert isinstance(tool_result_message.parts[0], ToolReturnPart)
    assert tool_result_message.parts[0].tool_name == "search"
    assert tool_result_message.parts[0].content == "sunny"


async def test_memory_service_records_assistant_message_with_usage_and_tool_info() -> None:
    resolved_conversation = conversation()
    memory_store = FakeMemoryStore()
    service = DefaultConversationMemoryService(memory_store, FakeSnapshotStore())

    stored = await service.record_assistant_message(
        conversation=resolved_conversation,
        response=AgentResponse(
            text="answer",
            metadata={"model": "test"},
            usage=AgentUsage(output_tokens=5),
            tool_info=[
                AgentToolInfo(
                    tool_name="search",
                    status=AgentToolStatus.SUCCEEDED,
                    call_id="call-1",
                )
            ],
            trace_id="trace-response",
        ),
        outbound_event_id=uuid4(),
    )

    assert stored.sequence == 1
    assert stored.role is ConversationMemoryRole.ASSISTANT
    assert stored.token_count == 5
    assert stored.trace_id == "trace-response"
    assert stored.metadata["model"] == "test"
    assert stored.metadata["usage"]["output_tokens"] == 5
    assert stored.metadata["tool_info"][0]["tool_name"] == "search"
