from dataclasses import dataclass, field
from typing import Any

from agent_service.media.models import MediaAsset

ImageAnalysisMetadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ImageAnalysisResult:
    analysis: str
    provider: str
    model: str
    metadata: ImageAnalysisMetadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageAnalysisRequest:
    prompt: str
    assets: tuple[MediaAsset, ...]
