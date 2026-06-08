from typing import Protocol, runtime_checkable

from agent_service.media.models import StoredMedia
from agent_service.transcription.models import TranscriptionResult


class TranscriptionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_code = error_code


class EmptyTranscriptionError(TranscriptionError):
    """Raised when audio transcription returns no usable text."""


@runtime_checkable
class AudioTranscriber(Protocol):
    async def transcribe(self, media: StoredMedia) -> TranscriptionResult:
        """Transcribe a stored audio file into text."""
        ...
