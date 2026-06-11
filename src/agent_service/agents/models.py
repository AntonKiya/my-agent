from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.messages import ModelMessage

from agent_service.channels.models import Attachment, ChannelName

AgentMetadata = dict[str, Any]
PydanticAIMessage = ModelMessage


def utc_now() -> datetime:
    return datetime.now(UTC)


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentContextRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class AgentToolStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentContextMessage(AgentModel):
    role: AgentContextRole
    text: str = Field(min_length=1)
    message_id: UUID | None = None
    metadata: AgentMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentContext(AgentModel):
    system_prompt_parts: list[str] = Field(default_factory=list)
    recent_messages: list[AgentContextMessage] = Field(default_factory=list)
    metadata: AgentMetadata = Field(default_factory=dict)


class PydanticAIRunContext(AgentModel):
    user_prompt: str | None = None
    message_history: list[PydanticAIMessage] = Field(default_factory=list)
    conversation_id: str | None = None
    instructions: str | None = None
    metadata: AgentMetadata = Field(default_factory=dict)


class AgentUsage(AgentModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    metadata: AgentMetadata = Field(default_factory=dict)


class AgentModelResponseUsage(AgentModel):
    message_index: int = Field(ge=0)
    model_response_index: int = Field(ge=0)
    part_types: list[str] = Field(default_factory=list)
    usage: AgentUsage


class AgentToolInfo(AgentModel):
    tool_name: str = Field(min_length=1)
    status: AgentToolStatus
    call_id: str | None = None
    error_message: str | None = None
    metadata: AgentMetadata = Field(default_factory=dict)


class AgentRequest(AgentModel):
    user_id: UUID
    conversation_id: UUID
    inbound_event_id: UUID
    channel: ChannelName = Field(min_length=1)
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    context: AgentContext = Field(default_factory=AgentContext)
    pydantic_ai: PydanticAIRunContext | None = None
    metadata: AgentMetadata = Field(default_factory=dict)
    trace_id: str | None = None

    @model_validator(mode="after")
    def request_must_include_message_content(self) -> "AgentRequest":
        if not self.text and not self.attachments:
            raise ValueError("Agent request must include text or attachments")
        return self


class AgentResponse(AgentModel):
    text: str = Field(min_length=1)
    metadata: AgentMetadata = Field(default_factory=dict)
    context_usage: AgentUsage | None = None
    run_usage: AgentUsage | None = None
    model_response_usages: list[AgentModelResponseUsage] = Field(default_factory=list)
    tool_info: list[AgentToolInfo] | None = None
    pydantic_ai_new_messages: list[PydanticAIMessage] = Field(default_factory=list)
    trace_id: str | None = None
