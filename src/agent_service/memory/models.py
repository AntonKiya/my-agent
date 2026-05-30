from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_service.agents import AgentContext, PydanticAIRunContext
from agent_service.channels.models import Attachment
from agent_service.conversations import Conversation

MemoryMetadata = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationMemoryRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ConversationSummaryStatus(StrEnum):
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    DEAD_LETTER = "dead_letter"


class ConversationMemoryMessage(MemoryModel):
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    user_id: UUID
    sequence: int | None = Field(default=None, gt=0)
    role: ConversationMemoryRole
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    tool_name: str | None = None
    tool_call_id: str | None = None
    inbound_event_id: UUID | None = None
    outbound_event_id: UUID | None = None
    trace_id: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    metadata: MemoryMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationContextSnapshot(MemoryModel):
    conversation_id: UUID
    user_id: UUID
    summary: str | None = None
    recent_messages: list[ConversationMemoryMessage] = Field(default_factory=list)
    last_compacted_message_id: UUID | None = None
    last_seen_message_id: UUID | None = None
    last_compacted_sequence: int | None = Field(default=None, ge=0)
    last_seen_sequence: int | None = Field(default=None, ge=0)
    version: int = Field(default=1, ge=1)
    token_count: int = Field(default=0, ge=0)
    metadata: MemoryMetadata = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class ConversationCompactionRequest(MemoryModel):
    conversation_id: UUID
    user_id: UUID
    previous_summary: str | None = None
    messages: list[ConversationMemoryMessage] = Field(default_factory=list)
    last_compacted_sequence: int | None = Field(default=None, ge=0)
    metadata: MemoryMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def messages_must_be_safe_for_summary(self) -> "ConversationCompactionRequest":
        previous_sequence = self.last_compacted_sequence or 0
        for message in self.messages:
            if message.conversation_id != self.conversation_id:
                raise ValueError("Compaction message belongs to another conversation")
            if message.user_id != self.user_id:
                raise ValueError("Compaction message belongs to another user")
            if message.role not in {
                ConversationMemoryRole.USER,
                ConversationMemoryRole.ASSISTANT,
            }:
                raise ValueError("Tool messages must not be included in compaction input")
            if message.sequence is None:
                raise ValueError("Compaction message must have a sequence")
            if message.sequence <= previous_sequence:
                raise ValueError("Compaction messages must be after last compacted sequence")
            previous_sequence = message.sequence
        return self


class ConversationCompactionResult(MemoryModel):
    conversation_id: UUID
    user_id: UUID
    summary: str | None = None
    compacted_message_ids: list[UUID] = Field(default_factory=list)
    last_compacted_message_id: UUID | None = None
    last_compacted_sequence: int | None = Field(default=None, ge=0)
    token_count: int = Field(default=0, ge=0)
    metadata: MemoryMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationCompactionDecision(MemoryModel):
    should_compact: bool
    reason: str
    estimated_input_tokens: int = Field(ge=0)
    usable_input_budget_tokens: int = Field(gt=0)
    trigger_tokens: int = Field(gt=0)
    recent_tail_budget_tokens: int = Field(gt=0)
    compact_through_sequence: int | None = Field(default=None, gt=0)
    keep_from_sequence: int | None = Field(default=None, gt=0)
    compactable_token_count: int = Field(default=0, ge=0)
    retained_tail_token_count: int = Field(default=0, ge=0)


class ConversationCompactionJob(MemoryModel):
    event_id: UUID = Field(default_factory=uuid4)
    conversation: Conversation
    compact_through_sequence: int = Field(gt=0)
    reason: str
    trace_id: str | None = None
    metadata: MemoryMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationSummary(MemoryModel):
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    user_id: UUID
    from_sequence: int = Field(gt=0)
    to_sequence: int = Field(gt=0)
    previous_summary: str | None = None
    summary: str | None = None
    compacted_message_ids: list[UUID] = Field(default_factory=list)
    last_compacted_message_id: UUID | None = None
    input_token_count: int = Field(default=0, ge=0)
    output_token_count: int = Field(default=0, ge=0)
    model: str | None = None
    status: ConversationSummaryStatus = ConversationSummaryStatus.COMPLETED
    trace_id: str | None = None
    metadata: MemoryMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def summary_bounds_must_be_consistent(self) -> "ConversationSummary":
        if self.to_sequence < self.from_sequence:
            raise ValueError("Conversation summary to_sequence must be >= from_sequence")
        if self.status is ConversationSummaryStatus.COMPLETED:
            if not self.summary:
                raise ValueError("Completed conversation summary must include summary text")
            if self.last_compacted_message_id is None:
                raise ValueError("Completed conversation summary must include compacted message id")
        return self


class PreparedConversationContext(MemoryModel):
    conversation_id: UUID
    user_id: UUID
    latest_user_message_id: UUID
    agent_context: AgentContext
    pydantic_ai: PydanticAIRunContext
    snapshot: ConversationContextSnapshot | None = None
    metadata: MemoryMetadata = Field(default_factory=dict)
