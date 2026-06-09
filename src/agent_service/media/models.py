from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from agent_service.channels.models import Attachment, utc_now

MediaMetadata = dict[str, Any]


class MediaAssetType(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class MediaPayload:
    attachment: Attachment
    content: bytes
    content_type: str | None = None
    filename: str | None = None
    metadata: MediaMetadata = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class StoredMedia:
    path: Path
    content_type: str | None = None
    filename: str | None = None
    size_bytes: int = 0
    metadata: MediaMetadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MediaAsset:
    media_id: str
    user_id: UUID
    conversation_id: UUID
    media_type: MediaAssetType
    storage_key: str
    content_type: str | None = None
    size_bytes: int = 0
    source_channel: str = ""
    source_attachment_id: str | None = None
    source_inbound_event_id: UUID | None = None
    metadata: MediaMetadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
