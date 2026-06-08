import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent_service.media.models import StoredMedia
from agent_service.observability.events import elapsed_ms, log_event, start_timer
from agent_service.transcription.interfaces import (
    AudioTranscriber,
    EmptyTranscriptionError,
    TranscriptionError,
)
from agent_service.transcription.models import TranscriptionResult

logger = logging.getLogger(__name__)

GROQ_TRANSCRIPTIONS_PATH = "/openai/v1/audio/transcriptions"
GROQ_TRANSCRIPTION_PROVIDER = "groq"
DEFAULT_GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
GROQ_SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".ogg",
        ".wav",
        ".webm",
    }
)
GROQ_EXTENSION_BY_CONTENT_TYPE = {
    "audio/flac": ".flac",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
}


@dataclass(slots=True)
class GroqWhisperTranscriber(AudioTranscriber):
    api_key: str
    client: httpx.AsyncClient
    model: str = DEFAULT_GROQ_TRANSCRIPTION_MODEL
    api_base_url: str = "https://api.groq.com"
    response_format: str = "json"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Groq API key must not be empty")
        if not self.model:
            raise ValueError("Groq transcription model must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("Groq transcription timeout must be greater than zero")

    async def transcribe(self, media: StoredMedia) -> TranscriptionResult:
        started_at = start_timer()
        multipart_filename = _multipart_filename(media)
        log_event(
            logger,
            logging.INFO,
            "Groq transcription request started",
            event="groq_transcription_request_started",
            model=self.model,
            audio_size_bytes=media.size_bytes,
            audio_content_type=media.content_type,
            multipart_filename_suffix=Path(multipart_filename).suffix.lower() or None,
        )
        try:
            response = await asyncio.wait_for(
                self._post_transcription(media, multipart_filename=multipart_filename),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise TranscriptionError(
                "Groq transcription timed out",
                retryable=True,
                error_code="groq_transcription_timeout",
            ) from exc
        except httpx.TransportError as exc:
            raise TranscriptionError(
                "Groq transcription transport error",
                retryable=True,
                error_code=_transport_error_code(exc),
            ) from exc

        body = _response_json(response)
        log_event(
            logger,
            logging.INFO if response.status_code < 400 else logging.WARNING,
            "Groq transcription response received",
            event="groq_transcription_response_received",
            http_status_code=response.status_code,
            model=self.model,
            duration_ms=elapsed_ms(started_at),
            audio_size_bytes=media.size_bytes,
            groq_request_id=response.headers.get("x-request-id"),
        )
        if response.status_code >= 400:
            raise TranscriptionError(
                _groq_error_message(response, body),
                retryable=_retryable_status(response.status_code),
                error_code=_groq_error_code(response, body),
            )

        if not isinstance(body, dict):
            raise TranscriptionError(
                "Groq transcription response was not a JSON object",
                retryable=True,
                error_code="groq_invalid_json",
            )
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise EmptyTranscriptionError(
                "Groq transcription returned empty text",
                retryable=False,
                error_code="groq_empty_transcription",
            )
        language = body.get("language")
        duration = body.get("duration")
        return TranscriptionResult(
            text=text.strip(),
            provider=GROQ_TRANSCRIPTION_PROVIDER,
            model=self.model,
            language=language if isinstance(language, str) else None,
            duration_seconds=duration if isinstance(duration, int | float) else None,
            metadata={
                "response_format": self.response_format,
                "request_id": response.headers.get("x-request-id"),
            },
        )

    async def _post_transcription(
        self,
        media: StoredMedia,
        *,
        multipart_filename: str,
    ) -> httpx.Response:
        content = await asyncio.to_thread(media.path.read_bytes)
        files = {
            "file": (
                multipart_filename,
                content,
                media.content_type or "application/octet-stream",
            )
        }
        data = {
            "model": self.model,
            "response_format": self.response_format,
        }
        return await self.client.post(
            self._url(),
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data,
            files=files,
        )

    def _url(self) -> str:
        return f"{self.api_base_url.rstrip('/')}{GROQ_TRANSCRIPTIONS_PATH}"


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _multipart_filename(media: StoredMedia) -> str:
    filename = media.filename or media.path.name
    suffix = Path(filename).suffix.lower()
    if suffix in GROQ_SUPPORTED_AUDIO_EXTENSIONS:
        return filename
    content_type_suffix = GROQ_EXTENSION_BY_CONTENT_TYPE.get(media.content_type or "")
    if content_type_suffix is not None:
        return f"audio{content_type_suffix}"
    return filename


def _groq_error_code(response: httpx.Response, body: Any) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("code") is not None:
        return f"groq_{error['code']}"
    return f"groq_http_{response.status_code}"


def _groq_error_message(response: httpx.Response, body: Any) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return response.reason_phrase or "Groq transcription request failed"


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _transport_error_code(exc: httpx.TransportError) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return "groq_connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "groq_read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "groq_write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "groq_pool_timeout"
    if isinstance(exc, httpx.TimeoutException):
        return "groq_timeout"
    if isinstance(exc, httpx.NetworkError):
        return "groq_network_error"
    if isinstance(exc, httpx.ProtocolError):
        return "groq_protocol_error"
    return "groq_transport_error"
