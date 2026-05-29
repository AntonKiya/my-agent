from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_service.agents import AgentContext, PydanticAIRunContext
from agent_service.channels.models import Attachment

MemoryMetadata = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationMemoryRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMemoryMessage(MemoryModel):
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    user_id: UUID
    role: ConversationMemoryRole
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
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
    version: int = Field(default=1, ge=1)
    token_count: int = Field(default=0, ge=0)
    metadata: MemoryMetadata = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class PreparedConversationContext(MemoryModel):
    conversation_id: UUID
    user_id: UUID
    latest_user_message_id: UUID
    agent_context: AgentContext
    pydantic_ai: PydanticAIRunContext
    snapshot: ConversationContextSnapshot | None = None
    metadata: MemoryMetadata = Field(default_factory=dict)
