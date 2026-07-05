from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agent_service.agents import (
    AgentModelResponseUsage,
    AgentResponse,
    AgentToolInfo,
    AgentToolStatus,
    AgentUsage,
)
from agent_service.channels import Attachment, AttachmentType, InboundEvent
from agent_service.conversations import Conversation
from agent_service.memory import (
    ConversationCompactionPolicy,
    ConversationCompactionResult,
    ConversationContextSnapshot,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationMemoryServiceError,
    ConversationSummary,
    DefaultConversationMemoryService,
    estimate_message_tokens,
)


@dataclass(slots=True)
class FakeMemoryStore:
    messages: dict[UUID, list[ConversationMemoryMessage]] = field(default_factory=dict)
    append_calls: list[ConversationMemoryMessage] = field(default_factory=list)
    recent_calls: list[tuple[UUID, int]] = field(default_factory=list)
    after_sequence_calls: list[tuple[UUID, UUID, int, int]] = field(default_factory=list)
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

    async def list_messages_after_sequence(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        after_sequence: int,
        limit: int,
    ) -> list[ConversationMemoryMessage]:
        self.after_sequence_calls.append((conversation_id, user_id, after_sequence, limit))
        return [
            message
            for message in self.messages.get(conversation_id, [])
            if message.user_id == user_id
            and message.sequence is not None
            and message.sequence > after_sequence
        ][-limit:]

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


@dataclass(slots=True)
class FakeSummaryStore:
    summaries: list[ConversationSummary] = field(default_factory=list)

    async def append_summary(
        self,
        *,
        summary: ConversationSummary,
    ) -> ConversationSummary:
        self.summaries.append(summary)
        return summary

    async def get_latest_completed_summary(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> ConversationSummary | None:
        matching = [
            summary
            for summary in self.summaries
            if summary.conversation_id == conversation_id and summary.user_id == user_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda summary: summary.to_sequence)

    async def get_completed_summary_by_sequence(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        to_sequence: int,
    ) -> ConversationSummary | None:
        for summary in self.summaries:
            if (
                summary.conversation_id == conversation_id
                and summary.user_id == user_id
                and summary.to_sequence == to_sequence
            ):
                return summary
        return None


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


def conversation_summary(
    *,
    resolved_conversation: Conversation,
    to_sequence: int,
    summary: str = "compressed context",
    output_token_count: int = 11,
    last_compacted_message_id: UUID | None = None,
) -> ConversationSummary:
    compacted_message_id = last_compacted_message_id or uuid4()
    return ConversationSummary(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        from_sequence=1,
        to_sequence=to_sequence,
        summary=summary,
        compacted_message_ids=[compacted_message_id],
        last_compacted_message_id=compacted_message_id,
        output_token_count=output_token_count,
        created_at=datetime(2026, 5, 30, 12, 30, tzinfo=UTC),
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


async def test_memory_service_extends_compacted_snapshot_without_losing_summary_tokens() -> None:
    resolved_conversation = conversation()
    compacted = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=2,
        text="compacted",
    )
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                summary="compressed old context",
                recent_messages=[],
                last_compacted_message_id=compacted.id,
                last_seen_message_id=compacted.id,
                last_compacted_sequence=2,
                last_seen_sequence=2,
                token_count=11,
                metadata={"latest_summary_token_count": 11},
            )
        }
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [compacted]})
    service = DefaultConversationMemoryService(memory_store, snapshot_store)

    stored = await service.record_user_message(
        conversation=resolved_conversation,
        event=inbound_event(resolved_conversation=resolved_conversation),
    )

    saved_snapshot = snapshot_store.save_calls[-1]
    assert saved_snapshot.summary == "compressed old context"
    assert saved_snapshot.recent_messages == [stored]
    assert saved_snapshot.last_seen_sequence == 3
    assert saved_snapshot.token_count == 11 + estimate_message_tokens(stored)


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
    assert prepared.pydantic_ai.instructions is None
    summary_message = prepared.pydantic_ai.message_history[0]
    assert isinstance(summary_message, ModelRequest)
    assert summary_message.conversation_id == str(resolved_conversation.id)
    assert summary_message.metadata == {"source": "conversation_summary"}
    assert len(summary_message.parts) == 1
    assert isinstance(summary_message.parts[0], SystemPromptPart)
    assert summary_message.parts[0].content == "compressed"
    history_message = prepared.pydantic_ai.message_history[1]
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
    assert memory_store.recent_calls == [(resolved_conversation.id, 2)]
    assert snapshot_store.save_calls[-1].last_seen_sequence == 2
    assert prepared.snapshot == snapshot_store.save_calls[-1]
    assert prepared.metadata["snapshot_source"] == "postgres"
    assert prepared.pydantic_ai.user_prompt == "latest"


async def test_memory_service_rebuilds_snapshot_from_latest_summary_and_recent_tail() -> None:
    resolved_conversation = conversation()
    compacted = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=2,
        text="old compacted answer",
    )
    recent = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=3,
        text="latest",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [compacted, recent]})
    snapshot_store = FakeSnapshotStore()
    summary_store = FakeSummaryStore(
        summaries=[
            conversation_summary(
                resolved_conversation=resolved_conversation,
                to_sequence=2,
                summary="compressed old context",
                output_token_count=11,
                last_compacted_message_id=compacted.id,
            )
        ]
    )
    service = DefaultConversationMemoryService(
        memory_store,
        snapshot_store,
        compaction_store=summary_store,
    )

    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=recent,
    )

    assert memory_store.recent_calls == []
    assert memory_store.after_sequence_calls == [
        (resolved_conversation.id, resolved_conversation.user_id, 2, 1)
    ]
    assert prepared.snapshot is not None
    assert prepared.snapshot.summary == "compressed old context"
    assert prepared.snapshot.recent_messages == [recent]
    assert prepared.snapshot.last_compacted_message_id == compacted.id
    assert prepared.snapshot.last_compacted_sequence == 2
    assert prepared.snapshot.last_seen_message_id == recent.id
    assert prepared.snapshot.last_seen_sequence == 3
    assert prepared.snapshot.token_count == 11 + estimate_message_tokens(recent)
    assert prepared.agent_context.system_prompt_parts == ["compressed old context"]
    assert prepared.pydantic_ai.instructions is None
    assert len(prepared.pydantic_ai.message_history) == 1
    summary_message = prepared.pydantic_ai.message_history[0]
    assert isinstance(summary_message, ModelRequest)
    assert isinstance(summary_message.parts[0], SystemPromptPart)
    assert summary_message.parts[0].content == "compressed old context"


async def test_memory_service_rebuilds_snapshot_when_summary_covers_current_sequence() -> None:
    resolved_conversation = conversation()
    compacted = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=2,
        text="old compacted answer",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [compacted]})
    snapshot_store = FakeSnapshotStore()
    summary_store = FakeSummaryStore(
        summaries=[
            conversation_summary(
                resolved_conversation=resolved_conversation,
                to_sequence=2,
                output_token_count=9,
                last_compacted_message_id=compacted.id,
            )
        ]
    )
    service = DefaultConversationMemoryService(
        memory_store,
        snapshot_store,
        compaction_store=summary_store,
    )

    decision = await service.evaluate_compaction(
        conversation=resolved_conversation,
        policy=ConversationCompactionPolicy(enabled=True),
    )

    saved_snapshot = snapshot_store.save_calls[-1]
    assert not decision.should_compact
    assert saved_snapshot.recent_messages == []
    assert saved_snapshot.last_seen_message_id == compacted.id
    assert saved_snapshot.last_seen_sequence == 2
    assert saved_snapshot.last_compacted_sequence == 2
    assert saved_snapshot.token_count == 9


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


async def test_memory_service_excludes_tool_messages_from_model_context() -> None:
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
    assert prepared.pydantic_ai.message_history == []


async def test_memory_service_model_context_includes_only_user_and_assistant_roles() -> None:
    resolved_conversation = conversation()
    old_user = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="old question",
    )
    tool_call = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=2,
        role=ConversationMemoryRole.TOOL_CALL,
        tool_name="search",
        tool_call_id="call-1",
    )
    tool_result = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=3,
        role=ConversationMemoryRole.TOOL_RESULT,
        text="tool output",
        tool_name="search",
        tool_call_id="call-1",
    )
    old_assistant = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=4,
        text="old answer",
    )
    latest = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=5,
        text="latest question",
    )
    memory_store = FakeMemoryStore(
        messages={
            resolved_conversation.id: [
                old_user,
                tool_call,
                tool_result,
                old_assistant,
                latest,
            ]
        }
    )
    service = DefaultConversationMemoryService(memory_store, FakeSnapshotStore())

    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=latest,
    )

    assert len(prepared.pydantic_ai.message_history) == 2
    user_message = prepared.pydantic_ai.message_history[0]
    assistant_message = prepared.pydantic_ai.message_history[1]
    assert isinstance(user_message, ModelRequest)
    assert isinstance(user_message.parts[0], UserPromptPart)
    assert user_message.parts[0].content == "old question"
    assert isinstance(assistant_message, ModelResponse)
    assert isinstance(assistant_message.parts[0], TextPart)
    assert assistant_message.parts[0].content == "old answer"


async def test_memory_service_preserves_tool_run_when_recent_limit_cuts_inside_it() -> None:
    resolved_conversation = conversation()
    old_user = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="old",
    )
    tool_call = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=2,
        role=ConversationMemoryRole.TOOL_CALL,
        tool_name="search",
        tool_call_id="call-1",
        metadata={"args": {"query": "weather"}},
    )
    tool_result = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=3,
        role=ConversationMemoryRole.TOOL_RESULT,
        text="sunny",
        tool_name="search",
        tool_call_id="call-1",
    )
    latest = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=4,
        text="thanks",
    )
    memory_store = FakeMemoryStore(
        messages={resolved_conversation.id: [old_user, tool_call, tool_result, latest]}
    )
    service = DefaultConversationMemoryService(
        memory_store,
        FakeSnapshotStore(),
        recent_message_limit=2,
    )

    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=latest,
    )

    assert memory_store.recent_calls == [(resolved_conversation.id, 4)]
    assert prepared.snapshot is not None
    assert prepared.snapshot.recent_messages == [tool_call, tool_result, latest]
    assert prepared.pydantic_ai.message_history == []


async def test_memory_service_expands_rebuild_fetch_until_tool_run_start() -> None:
    resolved_conversation = conversation()
    old_user = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="old",
    )
    first_call = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=2,
        role=ConversationMemoryRole.TOOL_CALL,
        tool_name="search",
        tool_call_id="call-1",
    )
    first_result = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=3,
        role=ConversationMemoryRole.TOOL_RESULT,
        text="first",
        tool_name="search",
        tool_call_id="call-1",
    )
    second_call = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=4,
        role=ConversationMemoryRole.TOOL_CALL,
        tool_name="search",
        tool_call_id="call-2",
    )
    second_result = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=5,
        role=ConversationMemoryRole.TOOL_RESULT,
        text="second",
        tool_name="search",
        tool_call_id="call-2",
    )
    latest = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=6,
        text="thanks",
    )
    memory_store = FakeMemoryStore(
        messages={
            resolved_conversation.id: [
                old_user,
                first_call,
                first_result,
                second_call,
                second_result,
                latest,
            ]
        }
    )
    service = DefaultConversationMemoryService(
        memory_store,
        FakeSnapshotStore(),
        recent_message_limit=2,
    )

    prepared = await service.prepare_agent_context(
        conversation=resolved_conversation,
        latest_user_message=latest,
    )

    assert memory_store.recent_calls == [
        (resolved_conversation.id, 4),
        (resolved_conversation.id, 6),
    ]
    assert prepared.snapshot is not None
    assert prepared.snapshot.recent_messages == [
        first_call,
        first_result,
        second_call,
        second_result,
        latest,
    ]
    assert prepared.pydantic_ai.message_history == []


async def test_memory_service_records_assistant_message_with_usage_and_tool_info() -> None:
    resolved_conversation = conversation()
    memory_store = FakeMemoryStore()
    service = DefaultConversationMemoryService(memory_store, FakeSnapshotStore())

    stored = await service.record_assistant_message(
        conversation=resolved_conversation,
        response=AgentResponse(
            text="answer",
            metadata={"model": "test"},
            context_usage=AgentUsage(output_tokens=5),
            run_usage=AgentUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            model_response_usages=[
                AgentModelResponseUsage(
                    message_index=1,
                    model_response_index=0,
                    part_types=["TextPart"],
                    usage=AgentUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            ],
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
    assert stored.trace_id == "trace-response"
    assert stored.metadata["model"] == "test"
    assert stored.metadata["context_usage"]["output_tokens"] == 5
    assert stored.metadata["run_usage"]["total_tokens"] == 15
    assert stored.metadata["model_response_usages"][0]["usage"]["total_tokens"] == 15
    assert "replayable_context_usage" not in stored.metadata
    assert stored.metadata["tool_info"][0]["tool_name"] == "search"


async def test_memory_service_records_assistant_generated_image_attachment() -> None:
    resolved_conversation = conversation()
    memory_store = FakeMemoryStore()
    service = DefaultConversationMemoryService(memory_store, FakeSnapshotStore())
    attachment = Attachment(
        attachment_id="generated-1",
        attachment_type=AttachmentType.IMAGE,
        content_type="image/png",
        metadata={"media_id": "generated-1", "generated": True},
    )

    stored = await service.record_assistant_message(
        conversation=resolved_conversation,
        response=AgentResponse(text="Готово", attachments=[attachment]),
    )

    assert stored.role is ConversationMemoryRole.ASSISTANT
    assert stored.text == "Готово"
    assert stored.attachments == [attachment]


async def test_memory_service_records_pydantic_ai_tool_messages_before_assistant() -> None:
    resolved_conversation = conversation()
    memory_store = FakeMemoryStore()
    service = DefaultConversationMemoryService(memory_store, FakeSnapshotStore())

    stored = await service.record_assistant_message(
        conversation=resolved_conversation,
        response=AgentResponse(
            text="done",
            pydantic_ai_new_messages=[
                ModelRequest(parts=[UserPromptPart(content="добавь сок")]),
                ModelResponse(
                    parts=[
                        LoadCapabilityCallPart(
                            args={"id": "vkusvill-shopping"},
                            tool_call_id="load-1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        LoadCapabilityReturnPart(
                            content={"instructions": "Use VkusVill tools."},
                            tool_call_id="load-1",
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="mcp_vkusvill_vkusvill_products_search",
                            args={"query": "сок"},
                            tool_call_id="call-1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="mcp_vkusvill_vkusvill_products_search",
                            content={"items": [{"xml_id": "123"}]},
                            tool_call_id="call-1",
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart(content="done")]),
            ],
            trace_id="trace-1",
        ),
        trace_id="trace-1",
    )

    assert stored.role is ConversationMemoryRole.ASSISTANT
    assert stored.text == "done"
    assert [message.role for message in memory_store.append_calls] == [
        ConversationMemoryRole.TOOL_CALL,
        ConversationMemoryRole.TOOL_RESULT,
        ConversationMemoryRole.TOOL_CALL,
        ConversationMemoryRole.TOOL_RESULT,
        ConversationMemoryRole.ASSISTANT,
    ]
    assert memory_store.append_calls[0].tool_name == "load_capability"
    assert memory_store.append_calls[0].metadata["tool_kind"] == "capability-load"
    assert memory_store.append_calls[1].metadata["content"] == {
        "instructions": "Use VkusVill tools."
    }
    assert memory_store.append_calls[2].metadata["args"] == {"query": "сок"}
    assert memory_store.append_calls[3].metadata["content"] == {"items": [{"xml_id": "123"}]}


async def test_memory_service_uses_first_model_response_input_for_snapshot_tokens() -> None:
    resolved_conversation = conversation()
    latest_user = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="latest",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [latest_user]})
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                recent_messages=[latest_user],
                last_seen_message_id=latest_user.id,
                last_seen_sequence=1,
                token_count=estimate_message_tokens(latest_user),
            )
        }
    )
    service = DefaultConversationMemoryService(memory_store, snapshot_store)

    stored = await service.record_assistant_message(
        conversation=resolved_conversation,
        response=AgentResponse(
            text="answer",
            context_usage=AgentUsage(input_tokens=36_425, output_tokens=546, total_tokens=36_971),
            run_usage=AgentUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            model_response_usages=[
                AgentModelResponseUsage(
                    message_index=1,
                    model_response_index=0,
                    part_types=["ThinkingPart", "LoadCapabilityCallPart"],
                    usage=AgentUsage(input_tokens=6_332, output_tokens=67, total_tokens=6_399),
                ),
                AgentModelResponseUsage(
                    message_index=5,
                    model_response_index=2,
                    part_types=["ThinkingPart", "ToolCallPart"],
                    usage=AgentUsage(
                        input_tokens=32_333,
                        output_tokens=116,
                        total_tokens=32_449,
                    ),
                ),
                AgentModelResponseUsage(
                    message_index=7,
                    model_response_index=3,
                    part_types=["ThinkingPart", "TextPart"],
                    usage=AgentUsage(
                        input_tokens=36_425,
                        output_tokens=546,
                        total_tokens=36_971,
                    ),
                ),
            ],
            pydantic_ai_new_messages=[
                ModelRequest(parts=[UserPromptPart(content="find docs")]),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="search",
                            args={"query": "large output"},
                            tool_call_id="call-1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="search",
                            content={"content": "tool output " * 500},
                            tool_call_id="call-1",
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart(content="answer")]),
            ],
        ),
    )

    saved_snapshot = snapshot_store.save_calls[-1]
    assert saved_snapshot.recent_messages[-1] == stored
    assert "replayable_context_usage" not in stored.metadata
    assert saved_snapshot.token_count == 6_332 + estimate_message_tokens(stored)


async def test_memory_service_falls_back_to_dialog_estimate_without_model_response_usages() -> None:
    resolved_conversation = conversation()
    latest_user = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="latest",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [latest_user]})
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                recent_messages=[latest_user],
                last_seen_message_id=latest_user.id,
                last_seen_sequence=1,
                token_count=estimate_message_tokens(latest_user),
            )
        }
    )
    service = DefaultConversationMemoryService(memory_store, snapshot_store)

    stored = await service.record_assistant_message(
        conversation=resolved_conversation,
        response=AgentResponse(
            text="answer",
            context_usage=AgentUsage(input_tokens=41, output_tokens=9, total_tokens=50),
            run_usage=AgentUsage(input_tokens=100, output_tokens=20, total_tokens=120),
        ),
    )

    saved_snapshot = snapshot_store.save_calls[-1]
    assert saved_snapshot.recent_messages == [latest_user, stored]
    assert saved_snapshot.token_count == estimate_message_tokens(
        latest_user
    ) + estimate_message_tokens(stored)


async def test_memory_service_prepares_compaction_request_from_memory_state() -> None:
    resolved_conversation = conversation()
    previous_summary = "User is building an agent service."
    first = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="first",
    )
    already_compacted = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=2,
        text="already compacted",
    )
    tool_result = ConversationMemoryMessage(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        sequence=3,
        role=ConversationMemoryRole.TOOL_RESULT,
        text="tool output",
        tool_name="search",
        tool_call_id="call-1",
    )
    second = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=4,
        text="second",
    )
    memory_store = FakeMemoryStore(
        messages={resolved_conversation.id: [first, already_compacted, tool_result, second]}
    )
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                summary=previous_summary,
                recent_messages=[first, already_compacted, tool_result, second],
                last_compacted_message_id=already_compacted.id,
                last_seen_message_id=second.id,
                last_compacted_sequence=2,
                last_seen_sequence=4,
            )
        }
    )
    service = DefaultConversationMemoryService(memory_store, snapshot_store)

    request = await service.prepare_compaction_request(conversation=resolved_conversation)

    assert request.previous_summary == previous_summary
    assert request.last_compacted_sequence == 2
    assert request.messages == [second]
    assert request.metadata["last_seen_sequence"] == 4


async def test_memory_service_records_compaction_result_and_updates_hot_snapshot() -> None:
    resolved_conversation = conversation()
    first = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="first",
    )
    second = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=2,
        text="second",
    )
    recent = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=3,
        text="recent",
    )
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: [first, second, recent]})
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                recent_messages=[first, second, recent],
                last_seen_message_id=recent.id,
                last_seen_sequence=3,
                token_count=27,
            )
        }
    )
    compaction_store = FakeSummaryStore()
    service = DefaultConversationMemoryService(
        memory_store,
        snapshot_store,
        compaction_store=compaction_store,
    )
    request = await service.prepare_compaction_request(conversation=resolved_conversation)
    result = ConversationCompactionResult(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        summary="first two messages compressed",
        compacted_message_ids=[first.id, second.id],
        last_compacted_message_id=second.id,
        last_compacted_sequence=2,
        token_count=8,
        metadata={
            "model": "summary-model",
            "run_usage": {"input_tokens": 22, "output_tokens": 8, "total_tokens": 30},
        },
        created_at=datetime(2026, 5, 30, 12, 30, tzinfo=UTC),
    )

    summary = await service.record_compaction_result(
        conversation=resolved_conversation,
        request=request,
        result=result,
        trace_id="trace-summary",
    )

    assert compaction_store.summaries == [summary]
    assert summary.from_sequence == 1
    assert summary.to_sequence == 2
    assert summary.input_token_count == 22
    assert summary.output_token_count == 8
    assert summary.model == "summary-model"
    assert summary.trace_id == "trace-summary"
    saved_snapshot = snapshot_store.save_calls[-1]
    assert saved_snapshot.summary == "first two messages compressed"
    assert saved_snapshot.recent_messages == [recent]
    assert saved_snapshot.last_compacted_message_id == second.id
    assert saved_snapshot.last_compacted_sequence == 2
    assert saved_snapshot.last_seen_message_id == recent.id
    assert saved_snapshot.last_seen_sequence == 3
    assert saved_snapshot.token_count == 8 + estimate_message_tokens(recent)
    assert saved_snapshot.metadata["latest_summary_id"] == str(summary.id)


async def test_memory_service_record_compaction_result_is_idempotent_by_sequence() -> None:
    resolved_conversation = conversation()
    first = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
        text="first",
    )
    second = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.ASSISTANT,
        sequence=2,
        text="second",
    )
    recent = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=3,
        text="recent",
    )
    existing = conversation_summary(
        resolved_conversation=resolved_conversation,
        to_sequence=2,
        summary="already compacted",
        output_token_count=6,
        last_compacted_message_id=second.id,
    )
    compaction_store = FakeSummaryStore(summaries=[existing])
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                recent_messages=[first, second, recent],
                last_seen_message_id=recent.id,
                last_seen_sequence=3,
            )
        }
    )
    service = DefaultConversationMemoryService(
        FakeMemoryStore(messages={resolved_conversation.id: [first, second, recent]}),
        snapshot_store,
        compaction_store=compaction_store,
    )
    request = await service.prepare_compaction_request(conversation=resolved_conversation)
    result = ConversationCompactionResult(
        conversation_id=resolved_conversation.id,
        user_id=resolved_conversation.user_id,
        summary="duplicate compacted",
        compacted_message_ids=[first.id, second.id],
        last_compacted_message_id=second.id,
        last_compacted_sequence=2,
        token_count=8,
    )

    summary = await service.record_compaction_result(
        conversation=resolved_conversation,
        request=request,
        result=result,
    )

    assert summary == existing
    assert compaction_store.summaries == [existing]
    saved_snapshot = snapshot_store.save_calls[-1]
    assert saved_snapshot.summary == "already compacted"
    assert saved_snapshot.recent_messages == [recent]
    assert saved_snapshot.last_compacted_sequence == 2


async def test_memory_service_evaluates_compaction_policy_from_fresh_snapshot() -> None:
    resolved_conversation = conversation()
    messages = [
        memory_message(
            resolved_conversation=resolved_conversation,
            role=ConversationMemoryRole.USER,
            sequence=1,
            text="old",
        ),
        memory_message(
            resolved_conversation=resolved_conversation,
            role=ConversationMemoryRole.ASSISTANT,
            sequence=2,
            text="answer",
        ),
    ]
    memory_store = FakeMemoryStore(messages={resolved_conversation.id: messages})
    snapshot_store = FakeSnapshotStore(
        snapshots={
            resolved_conversation.id: ConversationContextSnapshot(
                conversation_id=resolved_conversation.id,
                user_id=resolved_conversation.user_id,
                recent_messages=messages,
                last_seen_message_id=messages[-1].id,
                last_seen_sequence=2,
                token_count=90,
            )
        }
    )
    service = DefaultConversationMemoryService(memory_store, snapshot_store)
    policy = ConversationCompactionPolicy(
        enabled=True,
        context_window_tokens=100,
        reserved_output_tokens=0,
        trigger_fraction=0.80,
        recent_tail_fraction=0.10,
    )

    decision = await service.evaluate_compaction(
        conversation=resolved_conversation,
        policy=policy,
    )

    assert decision.should_compact
    assert decision.compact_through_sequence == 1


async def test_memory_service_rejects_compaction_result_from_another_user() -> None:
    resolved_conversation = conversation()
    service = DefaultConversationMemoryService(
        FakeMemoryStore(),
        FakeSnapshotStore(),
        compaction_store=FakeSummaryStore(),
    )
    message = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
    )
    request = await service.prepare_compaction_request(conversation=resolved_conversation)
    request = request.model_copy(update={"messages": [message]})
    result = ConversationCompactionResult(
        conversation_id=resolved_conversation.id,
        user_id=uuid4(),
        summary="wrong user",
        compacted_message_ids=[message.id],
        last_compacted_message_id=message.id,
        last_compacted_sequence=1,
    )

    with pytest.raises(ConversationMemoryServiceError):
        await service.record_compaction_result(
            conversation=resolved_conversation,
            request=request,
            result=result,
        )


async def test_memory_service_rejects_wrong_user_before_idempotency_lookup() -> None:
    resolved_conversation = conversation()
    message = memory_message(
        resolved_conversation=resolved_conversation,
        role=ConversationMemoryRole.USER,
        sequence=1,
    )
    existing = conversation_summary(
        resolved_conversation=resolved_conversation,
        to_sequence=1,
        last_compacted_message_id=message.id,
    )
    service = DefaultConversationMemoryService(
        FakeMemoryStore(messages={resolved_conversation.id: [message]}),
        FakeSnapshotStore(),
        compaction_store=FakeSummaryStore(summaries=[existing]),
    )
    request = await service.prepare_compaction_request(conversation=resolved_conversation)
    request = request.model_copy(update={"messages": [message]})
    result = ConversationCompactionResult(
        conversation_id=resolved_conversation.id,
        user_id=uuid4(),
        summary="wrong user",
        compacted_message_ids=[message.id],
        last_compacted_message_id=message.id,
        last_compacted_sequence=1,
    )

    with pytest.raises(ConversationMemoryServiceError):
        await service.record_compaction_result(
            conversation=resolved_conversation,
            request=request,
            result=result,
        )
