from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_service.channels.models import Attachment

MediaMetadata = dict[str, Any]


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
