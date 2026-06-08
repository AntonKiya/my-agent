import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent_service.channels.models import (
    Attachment,
    AttachmentType,
    InboundEvent,
    MessageType,
)
from agent_service.media import (
    MediaError,
    MediaFetcherRegistry,
    MediaStore,
    StoredMedia,
)
from agent_service.media.registry import MediaFetcherNotFoundError
from agent_service.observability.events import business_span, elapsed_ms, log_event, start_timer
from agent_service.transcription import AudioTranscriber, TranscriptionError

logger = logging.getLogger(__name__)

SleepCallable = Callable[[float], Awaitable[None]]

DEFAULT_CONTENT_PROCESSING_RETRY_BACKOFF_SECONDS = (1.0, 5.0)
SUPPORTED_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/flac",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
        "audio/x-m4a",
        "audio/x-wav",
    }
)


class ContentProcessingError(RuntimeError):
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


@dataclass(frozen=True, slots=True)
class ContentProcessingRetryPolicy:
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = DEFAULT_CONTENT_PROCESSING_RETRY_BACKOFF_SECONDS

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("Content processing retry max_attempts must be at least one")
        if any(delay < 0 for delay in self.backoff_seconds):
            raise ValueError(
                "Content processing retry backoff delays must be greater than or equal to zero"
            )

    def delay_for_attempt(self, attempt_number: int) -> float:
        if not self.backoff_seconds:
            return 0
        index = min(attempt_number - 1, len(self.backoff_seconds) - 1)
        return self.backoff_seconds[index]


@dataclass(slots=True)
class InboundContentPreprocessor:
    media_fetchers: MediaFetcherRegistry
    audio_media_store: MediaStore
    audio_transcriber: AudioTranscriber
    retry_policy: ContentProcessingRetryPolicy = field(
        default_factory=ContentProcessingRetryPolicy
    )
    max_audio_size_bytes: int = 25_000_000
    supported_audio_content_types: frozenset[str] = SUPPORTED_AUDIO_CONTENT_TYPES
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.max_audio_size_bytes <= 0:
            raise ValueError("Max audio size must be greater than zero")

    async def process(self, event: InboundEvent) -> None:
        attachment = _audio_attachment(event)
        if attachment is None:
            return

        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            started_at = start_timer()
            try:
                with business_span(
                    "Preprocess inbound audio",
                    event="inbound_audio_preprocessing",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                    attempt=attempt_number,
                ):
                    await self._process_audio_once(event=event, attachment=attachment)
                log_event(
                    logger,
                    logging.INFO,
                    "Inbound audio preprocessing completed",
                    event="inbound_audio_preprocessing_completed",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                    attempt=attempt_number,
                    duration_ms=elapsed_ms(started_at),
                )
                return
            except ContentProcessingError as exc:
                if not exc.retryable or attempt_number >= self.retry_policy.max_attempts:
                    log_event(
                        logger,
                        logging.WARNING,
                        "Inbound audio preprocessing failed",
                        event="inbound_audio_preprocessing_failed",
                        inbound_event_id=str(event.event_id),
                        channel=event.channel,
                        user_id=str(event.user_id) if event.user_id is not None else None,
                        attempt=attempt_number,
                        retryable=exc.retryable,
                        error_code=exc.error_code,
                        error_type=type(exc).__name__,
                        duration_ms=elapsed_ms(started_at),
                    )
                    raise
                delay_seconds = self.retry_policy.delay_for_attempt(attempt_number)
                log_event(
                    logger,
                    logging.WARNING,
                    "Inbound audio preprocessing retry scheduled",
                    event="inbound_audio_preprocessing_retry_scheduled",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                    attempt=attempt_number,
                    delay_seconds=delay_seconds,
                    error_code=exc.error_code,
                    duration_ms=elapsed_ms(started_at),
                )
                await self.sleep(delay_seconds)

    async def _process_audio_once(self, *, event: InboundEvent, attachment: Attachment) -> None:
        self._validate_audio_attachment(attachment)
        try:
            fetcher = self.media_fetchers.get(event.channel)
        except MediaFetcherNotFoundError as exc:
            raise ContentProcessingError(
                "No media fetcher is configured for inbound audio",
                retryable=False,
                error_code="media_fetcher_not_found",
            ) from exc

        stored_media: StoredMedia | None = None
        try:
            payload = await fetcher.fetch(event=event, attachment=attachment)
            if payload.size_bytes > self.max_audio_size_bytes:
                raise ContentProcessingError(
                    "Inbound audio exceeds configured size limit",
                    retryable=False,
                    error_code="audio_too_large",
                )
            stored_media = await self.audio_media_store.store(payload)
            log_event(
                logger,
                logging.INFO,
                "Inbound audio stored temporarily",
                event="inbound_audio_temp_stored",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                attachment_type=attachment.attachment_type.value,
                content_type=stored_media.content_type,
                size_bytes=stored_media.size_bytes,
            )
            result = await self.audio_transcriber.transcribe(stored_media)
        except MediaError as exc:
            raise ContentProcessingError(
                "Inbound media could not be prepared",
                retryable=exc.retryable,
                error_code=exc.error_code,
            ) from exc
        except TranscriptionError as exc:
            raise ContentProcessingError(
                "Inbound audio could not be transcribed",
                retryable=exc.retryable,
                error_code=exc.error_code,
            ) from exc
        finally:
            if stored_media is not None:
                try:
                    await self.audio_media_store.delete(stored_media)
                except MediaError as exc:
                    log_event(
                        logger,
                        logging.WARNING,
                        "Temporary inbound media cleanup failed",
                        event="inbound_media_cleanup_failed",
                        inbound_event_id=str(event.event_id),
                        channel=event.channel,
                        user_id=str(event.user_id) if event.user_id is not None else None,
                        error_code=exc.error_code,
                        error_type=type(exc).__name__,
                    )

        original_message_type = event.message_type
        event.text = result.text
        event.attachments = []
        event.metadata["transcription"] = {
            "provider": result.provider,
            "model": result.model,
            "source_message_type": original_message_type.value,
            "source_attachment_type": attachment.attachment_type.value,
            "status": "completed",
            "language": result.language,
            "duration_seconds": result.duration_seconds,
        }
        event.message_type = MessageType.TEXT

    def _validate_audio_attachment(self, attachment: Attachment) -> None:
        if attachment.attachment_type not in {AttachmentType.AUDIO, AttachmentType.VOICE}:
            raise ContentProcessingError(
                "Attachment is not audio",
                retryable=False,
                error_code="unsupported_attachment_type",
            )
        content_type = attachment.content_type
        if content_type is not None and content_type not in self.supported_audio_content_types:
            raise ContentProcessingError(
                "Unsupported audio content type",
                retryable=False,
                error_code="unsupported_audio_content_type",
            )


def event_needs_content_preprocessing(event: InboundEvent) -> bool:
    return _audio_attachment(event) is not None


def _audio_attachment(event: InboundEvent) -> Attachment | None:
    if event.message_type not in {MessageType.AUDIO, MessageType.VOICE}:
        return None
    for attachment in event.attachments:
        if attachment.attachment_type in {AttachmentType.AUDIO, AttachmentType.VOICE}:
            return attachment
    return None
