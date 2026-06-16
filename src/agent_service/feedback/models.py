from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from agent_service.channels.models import ChannelName, utc_now

FeedbackMetadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class FeedbackEntry:
    user_id: UUID
    conversation_id: UUID
    source_channel: ChannelName
    source_external_user_id: str
    source_external_chat_id: str
    text: str
    id: UUID = field(default_factory=uuid4)
    source_thread_id: str | None = None
    source_inbound_event_id: UUID | None = None
    request_inbound_event_id: UUID | None = None
    metadata: FeedbackMetadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class PendingFeedback:
    user_id: UUID
    conversation_id: UUID
    channel: ChannelName
    request_inbound_event_id: UUID
    requested_at: datetime = field(default_factory=utc_now)
