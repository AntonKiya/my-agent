from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic_ai.usage import RunUsage

from agent_service.memory import (
    CONVERSATION_SUMMARY_SYSTEM_PROMPT,
    ConversationCompactionRequest,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationSummaryOutput,
    PydanticAIConversationCompactor,
    render_conversation_summary,
    render_summary_user_prompt,
    render_transcript,
)


@dataclass(slots=True)
class FakeSummaryResult:
    output: ConversationSummaryOutput
    run_usage: RunUsage = field(default_factory=RunUsage)

    def usage(self) -> RunUsage:
        return self.run_usage


@dataclass(slots=True)
class FakeSummaryAgent:
    output: ConversationSummaryOutput
    run_usage: RunUsage = field(default_factory=RunUsage)
    prompts: list[str | None] = field(default_factory=list)
    instructions: list[str | object] = field(default_factory=list)
    metadata: list[dict[str, object] | None] = field(default_factory=list)

    async def run(
        self,
        user_prompt: str | None = None,
        *,
        output_type: type[ConversationSummaryOutput] | None = None,
        instructions: str | object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> FakeSummaryResult:
        assert output_type is ConversationSummaryOutput
        self.prompts.append(user_prompt)
        self.instructions.append(instructions)
        self.metadata.append(metadata)
        return FakeSummaryResult(output=self.output, run_usage=self.run_usage)


def memory_message(
    *,
    role: ConversationMemoryRole,
    sequence: int,
    text: str = "hello",
) -> ConversationMemoryMessage:
    return ConversationMemoryMessage(
        conversation_id=uuid4(),
        user_id=uuid4(),
        sequence=sequence,
        role=role,
        text=text,
        tool_name="search" if role is ConversationMemoryRole.TOOL_RESULT else None,
        tool_call_id="call-1" if role is ConversationMemoryRole.TOOL_RESULT else None,
        created_at=datetime(2026, 5, 30, 12, sequence, tzinfo=UTC),
    )


def compaction_request(
    *,
    messages: list[ConversationMemoryMessage],
    previous_summary: str | None = "previous",
) -> ConversationCompactionRequest:
    conversation_id = uuid4()
    user_id = uuid4()
    normalized = [
        message.model_copy(update={"conversation_id": conversation_id, "user_id": user_id})
        for message in messages
    ]
    return ConversationCompactionRequest(
        conversation_id=conversation_id,
        user_id=user_id,
        previous_summary=previous_summary,
        messages=normalized,
        last_compacted_sequence=1,
        metadata={"snapshot_version": 1},
    )


def test_render_transcript_uses_sequence_role_and_text() -> None:
    user = memory_message(role=ConversationMemoryRole.USER, sequence=2, text="Build compaction")
    assistant = memory_message(
        role=ConversationMemoryRole.ASSISTANT,
        sequence=3,
        text="Use token policy",
    )

    transcript = render_transcript([user, assistant])

    assert "[2] user: Build compaction" in transcript
    assert "[3] assistant: Use token policy" in transcript


def test_render_summary_user_prompt_includes_inputs_and_scope_metadata() -> None:
    request = compaction_request(
        messages=[memory_message(role=ConversationMemoryRole.USER, sequence=2, text="hello")]
    )

    prompt = render_summary_user_prompt(
        request=request,
        messages=request.messages,
        message_sequence_range="2-2",
        summary_version=2,
        current_datetime=datetime(2026, 5, 30, 13, 0, tzinfo=UTC),
        target_summary_tokens=900,
    )

    assert "previous_summary:\nprevious" in prompt
    assert "transcript_to_compact:\n[2] user: hello" in prompt
    assert "- message_sequence_range: 2-2" in prompt
    assert "- summary_version: 2" in prompt
    assert "- target_summary_tokens: 900" in prompt


def test_render_conversation_summary_omits_empty_sections_and_uses_exact_headings() -> None:
    output = ConversationSummaryOutput(
        current_state=["Compaction pipeline is being implemented."],
        next_likely_steps=["Add Pydantic AI compactor wiring."],
    )

    summary = render_conversation_summary(
        output=output,
        message_sequence_range="2-4",
        summary_version=2,
        current_datetime=datetime(2026, 5, 30, 13, 0, tzinfo=UTC),
    )

    assert summary.startswith("<conversation_summary>")
    assert "Scope:" in summary
    assert "- Covers: 2-4" in summary
    assert "Current state:" in summary
    assert "- Compaction pipeline is being implemented." in summary
    assert "Next likely steps:" in summary
    assert "Open questions / blockers:" not in summary
    assert summary.endswith("</conversation_summary>")


async def test_pydantic_ai_compactor_returns_structured_summary_result() -> None:
    first = memory_message(role=ConversationMemoryRole.USER, sequence=2, text="Need compaction")
    second = memory_message(role=ConversationMemoryRole.ASSISTANT, sequence=3, text="Will add it")
    request = compaction_request(messages=[first, second])
    agent = FakeSummaryAgent(
        output=ConversationSummaryOutput(
            current_state=["Production compaction is being added."],
            progress_so_far=["Compaction queue and worker already exist."],
        ),
        run_usage=RunUsage(input_tokens=100, output_tokens=42, requests=1),
    )
    compactor = PydanticAIConversationCompactor(
        agent=agent,
        target_summary_tokens=900,
        clock=lambda: datetime(2026, 5, 30, 13, 0, tzinfo=UTC),
    )

    result = await compactor.compact(request=request)

    assert agent.instructions == [CONVERSATION_SUMMARY_SYSTEM_PROMPT]
    assert result.conversation_id == request.conversation_id
    assert result.user_id == request.user_id
    assert result.compacted_message_ids == [message.id for message in request.messages]
    assert result.last_compacted_message_id == request.messages[-1].id
    assert result.last_compacted_sequence == 3
    assert result.token_count == 42
    assert result.metadata["compactor"] == "pydantic_ai"
    assert result.metadata["summary_version"] == 2
    assert result.metadata["message_sequence_range"] == "2-3"
    assert "Current state:" in (result.summary or "")
    assert "- Production compaction is being added." in (result.summary or "")


async def test_pydantic_ai_compactor_filters_tool_messages_before_prompt() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    user = ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=2,
        role=ConversationMemoryRole.USER,
        text="safe user text",
    )
    tool_result = ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=3,
        role=ConversationMemoryRole.TOOL_RESULT,
        text="SECRET TOOL RESULT",
        tool_name="search",
        tool_call_id="call-1",
    )
    assistant = ConversationMemoryMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        sequence=4,
        role=ConversationMemoryRole.ASSISTANT,
        text="safe assistant text",
    )
    request = ConversationCompactionRequest.model_construct(
        conversation_id=conversation_id,
        user_id=user_id,
        previous_summary=None,
        messages=[user, tool_result, assistant],
        last_compacted_sequence=1,
        metadata={},
    )
    agent = FakeSummaryAgent(output=ConversationSummaryOutput(current_state=["Safe summary."]))
    compactor = PydanticAIConversationCompactor(
        agent=agent,
        clock=lambda: datetime(2026, 5, 30, 13, 0, tzinfo=UTC),
    )

    await compactor.compact(request=request)

    prompt = agent.prompts[0] or ""
    assert "safe user text" in prompt
    assert "safe assistant text" in prompt
    assert "SECRET TOOL RESULT" not in prompt
    assert "tool_result" not in prompt


async def test_pydantic_ai_compactor_rejects_empty_safe_transcript() -> None:
    request = ConversationCompactionRequest(
        conversation_id=uuid4(),
        user_id=uuid4(),
        messages=[],
    )
    compactor = PydanticAIConversationCompactor(
        agent=FakeSummaryAgent(output=ConversationSummaryOutput()),
    )

    with pytest.raises(ValueError, match="user or assistant messages"):
        await compactor.compact(request=request)
