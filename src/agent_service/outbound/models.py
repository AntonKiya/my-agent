from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_service.channels.models import (
    Attachment,
    ChannelMetadata,
    ChannelName,
    MessageType,
)
from agent_service.delivery.models import DeliveryStatus

OutboundEventStatus = DeliveryStatus
OutboundMetadata = ChannelMetadata


def utc_now() -> datetime:
    return datetime.now(UTC)


class OutboundModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutboundEvent(OutboundModel):
    event_id: UUID = Field(default_factory=uuid4)
    channel: ChannelName = Field(min_length=1)
    user_id: UUID
    conversation_id: UUID
    external_chat_id: str = Field(min_length=1)
    message_type: MessageType = MessageType.TEXT
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    thread_id: str | None = None
    reply_to_message_id: str | None = None
    channel_metadata: ChannelMetadata = Field(default_factory=dict)
    metadata: OutboundMetadata = Field(default_factory=dict)
    trace_id: str | None = None
    status: OutboundEventStatus = OutboundEventStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
