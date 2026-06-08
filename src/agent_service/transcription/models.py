from dataclasses import dataclass, field
from typing import Any

TranscriptionMetadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    provider: str
    model: str
    language: str | None = None
    duration_seconds: float | None = None
    metadata: TranscriptionMetadata = field(default_factory=dict)
