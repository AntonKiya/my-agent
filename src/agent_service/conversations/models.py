from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_service.channels.models import ChannelName

ConversationMetadata = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ConversationType(StrEnum):
    PRIVATE = "private"
    GROUP = "group"
    THREAD = "thread"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class ConversationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Conversation(ConversationModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    channel: ChannelName = Field(min_length=1)
    conversation_key: str = Field(min_length=1)
    external_chat_id: str = Field(min_length=1)
    type: ConversationType = ConversationType.PRIVATE
    thread_id: str | None = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    metadata: ConversationMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConversationLookup(ConversationModel):
    conversation_key: str = Field(min_length=1)


class ObservedConversation(ConversationModel):
    user_id: UUID
    channel: ChannelName = Field(min_length=1)
    conversation_key: str = Field(min_length=1)
    external_chat_id: str = Field(min_length=1)
    type: ConversationType = ConversationType.PRIVATE
    thread_id: str | None = None
    metadata: ConversationMetadata = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)

    def lookup(self) -> ConversationLookup:
        return ConversationLookup(conversation_key=self.conversation_key)
