from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agent_service.memory import (
    ConversationCompactionPolicy,
    ConversationCompactionRequest,
    ConversationCompactor,
    ConversationContextSnapshot,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    NoopConversationCompactor,
    compactable_messages_from_snapshot,
    compaction_request_from_snapshot,
    estimate_messages_tokens,
)


def memory_message(
    *,
    role: ConversationMemoryRole,
    conversation_id: UUID,
    user_id: UUID,
    sequence: int,
    text: str | None = "hello",
) -> ConversationMemoryMessage:
    return ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=sequence,
        role=role,
        text=text,
        tool_name=(
            "search"
            if role in {ConversationMemoryRole.TOOL_CALL, ConversationMemoryRole.TOOL_RESULT}
            else None
        ),
        tool_call_id=(
            "call-1"
            if role in {ConversationMemoryRole.TOOL_CALL, ConversationMemoryRole.TOOL_RESULT}
            else None
        ),
        created_at=datetime(2026, 5, 30, 12, sequence, tzinfo=UTC),
    )


def test_compaction_request_rejects_tool_messages() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    tool_result = memory_message(
        role=ConversationMemoryRole.TOOL_RESULT,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=1,
        text="tool result",
    )

    with pytest.raises(ValidationError):
        ConversationCompactionRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[tool_result],
        )


def test_compaction_request_rejects_out_of_order_messages() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    first = memory_message(
        role=ConversationMemoryRole.USER,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=2,
    )
    second = memory_message(
        role=ConversationMemoryRole.ASSISTANT,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=1,
    )

    with pytest.raises(ValidationError):
        ConversationCompactionRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[first, second],
        )


def test_compaction_request_rejects_messages_from_another_user_or_conversation() -> None:
    conversation_id = uuid4()
    user_id = uuid4()

    with pytest.raises(ValidationError):
        ConversationCompactionRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[
                memory_message(
                    role=ConversationMemoryRole.USER,
                    conversation_id=uuid4(),
                    user_id=user_id,
                    sequence=1,
                )
            ],
        )

    with pytest.raises(ValidationError):
        ConversationCompactionRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[
                memory_message(
                    role=ConversationMemoryRole.USER,
                    conversation_id=conversation_id,
                    user_id=uuid4(),
                    sequence=1,
                )
            ],
        )


def test_compaction_request_from_snapshot_excludes_tool_messages() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    user_message = memory_message(
        role=ConversationMemoryRole.USER,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=2,
        text="user",
    )
    tool_call = memory_message(
        role=ConversationMemoryRole.TOOL_CALL,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=3,
        text=None,
    )
    assistant_message = memory_message(
        role=ConversationMemoryRole.ASSISTANT,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=4,
        text="assistant",
    )
    snapshot = ConversationContextSnapshot(
        conversation_id=conversation_id,
        user_id=user_id,
        summary="previous summary",
        recent_messages=[user_message, tool_call, assistant_message],
        last_compacted_sequence=1,
        last_seen_message_id=assistant_message.id,
        last_seen_sequence=4,
        version=2,
    )

    request = compaction_request_from_snapshot(snapshot)

    assert compactable_messages_from_snapshot(snapshot) == [user_message, assistant_message]
    assert request.previous_summary == "previous summary"
    assert request.messages == [user_message, assistant_message]
    assert request.last_compacted_sequence == 1
    assert request.metadata["snapshot_version"] == 2
    assert request.metadata["last_seen_sequence"] == 4


def test_compaction_request_from_snapshot_can_limit_compacted_prefix() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    first = memory_message(
        role=ConversationMemoryRole.USER,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=2,
    )
    second = memory_message(
        role=ConversationMemoryRole.ASSISTANT,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=3,
    )
    recent = memory_message(
        role=ConversationMemoryRole.USER,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=4,
    )
    snapshot = ConversationContextSnapshot(
        conversation_id=conversation_id,
        user_id=user_id,
        recent_messages=[first, second, recent],
        last_compacted_sequence=1,
        last_seen_message_id=recent.id,
        last_seen_sequence=4,
    )

    request = compaction_request_from_snapshot(snapshot, compact_through_sequence=3)

    assert request.messages == [first, second]


def test_compaction_policy_uses_token_trigger_and_retains_recent_tail_by_tokens() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    messages = [
        memory_message(
            role=ConversationMemoryRole.USER,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=1,
        ),
        memory_message(
            role=ConversationMemoryRole.ASSISTANT,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=2,
        ),
        memory_message(
            role=ConversationMemoryRole.USER,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=3,
        ),
        memory_message(
            role=ConversationMemoryRole.ASSISTANT,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=4,
        ),
    ]
    snapshot = ConversationContextSnapshot(
        conversation_id=conversation_id,
        user_id=user_id,
        recent_messages=messages,
        last_seen_message_id=messages[-1].id,
        last_seen_sequence=4,
        token_count=90,
    )
    policy = ConversationCompactionPolicy(
        enabled=True,
        context_window_tokens=100,
        reserved_output_tokens=0,
        trigger_fraction=0.80,
        recent_tail_fraction=0.10,
    )

    decision = policy.decide(snapshot=snapshot)

    assert decision.should_compact
    assert decision.reason == "trigger_reached"
    assert decision.trigger_tokens == 80
    assert decision.recent_tail_budget_tokens == 10
    assert decision.compact_through_sequence == 3
    assert decision.keep_from_sequence == 4
    assert decision.compactable_token_count == estimate_messages_tokens(messages[:3])
    assert decision.retained_tail_token_count == estimate_messages_tokens(messages[3:])


def test_compaction_policy_uses_dialog_messages_for_recent_tail() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    messages = [
        memory_message(
            role=ConversationMemoryRole.USER,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=1,
        ),
        memory_message(
            role=ConversationMemoryRole.TOOL_CALL,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=2,
            text=None,
        ),
        memory_message(
            role=ConversationMemoryRole.TOOL_RESULT,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=3,
            text="tool result " * 100,
        ),
        memory_message(
            role=ConversationMemoryRole.ASSISTANT,
            conversation_id=conversation_id,
            user_id=user_id,
            sequence=4,
        ),
    ]
    snapshot = ConversationContextSnapshot(
        conversation_id=conversation_id,
        user_id=user_id,
        recent_messages=messages,
        last_seen_message_id=messages[-1].id,
        last_seen_sequence=4,
        token_count=90,
    )
    policy = ConversationCompactionPolicy(
        enabled=True,
        context_window_tokens=100,
        reserved_output_tokens=0,
        trigger_fraction=0.80,
        recent_tail_fraction=0.10,
    )

    decision = policy.decide(snapshot=snapshot)

    assert decision.should_compact
    assert decision.compact_through_sequence == 1
    assert decision.keep_from_sequence == 4
    assert decision.retained_tail_token_count == estimate_messages_tokens([messages[3]])


def test_compaction_policy_does_not_trigger_below_token_threshold() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    message = memory_message(
        role=ConversationMemoryRole.USER,
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=1,
    )
    snapshot = ConversationContextSnapshot(
        conversation_id=conversation_id,
        user_id=user_id,
        recent_messages=[message],
        last_seen_message_id=message.id,
        last_seen_sequence=1,
        token_count=79,
    )
    policy = ConversationCompactionPolicy(
        enabled=True,
        context_window_tokens=100,
        reserved_output_tokens=0,
        trigger_fraction=0.80,
        recent_tail_fraction=0.30,
    )

    decision = policy.decide(snapshot=snapshot)

    assert not decision.should_compact
    assert decision.reason == "below_trigger"


def test_compaction_policy_rejects_invalid_budget_settings() -> None:
    with pytest.raises(ValueError):
        ConversationCompactionPolicy(
            context_window_tokens=100,
            reserved_output_tokens=100,
        )

    with pytest.raises(ValueError):
        ConversationCompactionPolicy(
            trigger_fraction=0.30,
            recent_tail_fraction=0.30,
        )


async def test_noop_compactor_preserves_previous_summary_without_compacting() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    request = ConversationCompactionRequest(
        conversation_id=conversation_id,
        user_id=user_id,
        previous_summary="stable summary",
        messages=[
            memory_message(
                role=ConversationMemoryRole.USER,
                conversation_id=conversation_id,
                user_id=user_id,
                sequence=3,
            )
        ],
        last_compacted_sequence=2,
        metadata={"reason": "test"},
    )
    compactor = NoopConversationCompactor()

    result = await compactor.compact(request=request)

    assert isinstance(compactor, ConversationCompactor)
    assert result.conversation_id == conversation_id
    assert result.user_id == user_id
    assert result.summary == "stable summary"
    assert result.compacted_message_ids == []
    assert result.last_compacted_message_id is None
    assert result.last_compacted_sequence == 2
    assert result.metadata == {"noop": True, "reason": "test"}
