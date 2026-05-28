from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

ChannelName = str
ChannelMetadata = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(UTC)


class MessageType(StrEnum):
    TEXT = "text"
    MEDIA = "media"
    VOICE = "voice"
    AUDIO = "audio"
    DOCUMENT = "document"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"


class AttachmentType(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VOICE = "voice"
    VIDEO = "video"
    DOCUMENT = "document"
    OTHER = "other"


class InboundEventStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    DEAD_LETTER = "dead_letter"
    FALLBACK_SENT = "fallback_sent"


class OutboundEventStatus(StrEnum):
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    FAILED_RETRYABLE = "failed_retryable"
    DEAD_LETTER = "dead_letter"


class DeliveryStatus(StrEnum):
    SENT = "sent"
    FAILED_RETRYABLE = "failed_retryable"
    DEAD_LETTER = "dead_letter"


class ChannelModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Attachment(ChannelModel):
    attachment_id: str | None = None
    attachment_type: AttachmentType = AttachmentType.OTHER
    external_id: str | None = None
    content_type: str | None = None
    url: str | None = None
    metadata: ChannelMetadata = Field(default_factory=dict)


class InboundEvent(ChannelModel):
    event_id: UUID = Field(default_factory=uuid4)
    channel: ChannelName = Field(min_length=1)
    external_user_id: str = Field(min_length=1)
    external_chat_id: str = Field(min_length=1)
    external_message_id: str | None = None
    external_update_id: str | None = None
    idempotency_key: str = Field(min_length=1)
    user_id: UUID | None = None
    message_type: MessageType = MessageType.TEXT
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    thread_id: str | None = None
    reply_to_message_id: str | None = None
    channel_metadata: ChannelMetadata = Field(default_factory=dict)
    metadata: ChannelMetadata = Field(default_factory=dict)
    trace_id: str | None = None
    status: InboundEventStatus = InboundEventStatus.QUEUED
    received_at: datetime = Field(default_factory=utc_now)


class OutboundEvent(ChannelModel):
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
    metadata: ChannelMetadata = Field(default_factory=dict)
    trace_id: str | None = None
    status: OutboundEventStatus = OutboundEventStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)


class DeliveryResult(ChannelModel):
    event_id: UUID
    channel: ChannelName = Field(min_length=1)
    status: DeliveryStatus
    external_message_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = Field(default=None, gt=0)
    metadata: ChannelMetadata = Field(default_factory=dict)
    delivered_at: datetime | None = None
