from dataclasses import dataclass, field
from typing import Any

from agent_service.media.models import MediaAsset

ImageGenerationMetadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    content: bytes
    content_type: str
    filename: str | None = None
    metadata: ImageGenerationMetadata = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    images: tuple[GeneratedImage, ...]
    provider: str
    model: str
    text: str | None = None
    metadata: ImageGenerationMetadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    prompt: str
    source_assets: tuple[MediaAsset, ...] = ()
