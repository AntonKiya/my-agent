import asyncio
import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

from agent_service.channels.models import (
    Attachment,
    AttachmentType,
    InboundEvent,
    MessageType,
)
from agent_service.document_reading import is_supported_document_payload
from agent_service.media import (
    ChannelMediaFetcher,
    MediaAsset,
    MediaAssetStore,
    MediaAssetType,
    MediaError,
    MediaFetcherRegistry,
    MediaStore,
    PersistentMediaStore,
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
SUPPORTED_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
DEFAULT_SINGLE_IMAGE_PROMPT = "Что изображено на этом изображении? Опиши все в деталях."
DEFAULT_MULTI_IMAGE_PROMPT = "Что изображено на этих изображениях? Опиши каждое в деталях."
DEFAULT_SINGLE_DOCUMENT_PROMPT = "Прочитай этот файл и кратко опиши его содержимое."
DEFAULT_MULTI_DOCUMENT_PROMPT = "Прочитай эти файлы и кратко опиши содержимое каждого."
DEFAULT_MIXED_MEDIA_PROMPT = (
    "Проанализируй прикрепленные изображения и файлы с учетом их содержимого."
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
    audio_media_store: MediaStore | None = None
    audio_transcriber: AudioTranscriber | None = None
    image_media_store: PersistentMediaStore | None = None
    document_media_store: PersistentMediaStore | None = None
    media_asset_store: MediaAssetStore | None = None
    retry_policy: ContentProcessingRetryPolicy = field(default_factory=ContentProcessingRetryPolicy)
    max_audio_size_bytes: int = 25_000_000
    max_image_size_bytes: int = 10_000_000
    max_document_size_bytes: int = 2_000_000
    supported_audio_content_types: frozenset[str] = SUPPORTED_AUDIO_CONTENT_TYPES
    supported_image_content_types: frozenset[str] = SUPPORTED_IMAGE_CONTENT_TYPES
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.max_audio_size_bytes <= 0:
            raise ValueError("Max audio size must be greater than zero")
        if self.max_image_size_bytes <= 0:
            raise ValueError("Max image size must be greater than zero")
        if self.max_document_size_bytes <= 0:
            raise ValueError("Max document size must be greater than zero")

    async def process(self, event: InboundEvent, *, conversation_id: UUID | None = None) -> None:
        attachment = _audio_attachment(event)
        if attachment is not None:
            await self._process_audio_with_retry(event=event, attachment=attachment)
            return

        persistent_attachments = _persistent_attachments(event)
        if persistent_attachments:
            if conversation_id is None:
                raise ContentProcessingError(
                    "Conversation id is required for media preprocessing",
                    retryable=False,
                    error_code="conversation_id_required",
                )
            await self._process_persistent_media_with_retry(
                event=event,
                attachments=persistent_attachments,
                conversation_id=conversation_id,
            )

    async def _process_audio_with_retry(
        self,
        *,
        event: InboundEvent,
        attachment: Attachment,
    ) -> None:
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
        if self.audio_media_store is None or self.audio_transcriber is None:
            raise ContentProcessingError(
                "Audio preprocessing is not configured",
                retryable=False,
                error_code="audio_preprocessing_not_configured",
            )
        self._validate_audio_attachment(attachment)
        _validate_declared_size(
            attachment,
            max_size_bytes=self.max_audio_size_bytes,
            label="audio",
            error_code="audio_too_large",
        )
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

    async def _process_persistent_media_with_retry(
        self,
        *,
        event: InboundEvent,
        attachments: list[Attachment],
        conversation_id: UUID,
    ) -> None:
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            started_at = start_timer()
            try:
                with business_span(
                    "Preprocess inbound persistent media",
                    event="inbound_persistent_media_preprocessing",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                    conversation_id=str(conversation_id),
                    media_count=len(attachments),
                    attempt=attempt_number,
                ):
                    await self._process_persistent_media_once(
                        event=event,
                        attachments=attachments,
                        conversation_id=conversation_id,
                    )
                log_event(
                    logger,
                    logging.INFO,
                    "Inbound persistent media preprocessing completed",
                    event="inbound_persistent_media_preprocessing_completed",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                    conversation_id=str(conversation_id),
                    media_count=len(attachments),
                    attempt=attempt_number,
                    duration_ms=elapsed_ms(started_at),
                )
                return
            except ContentProcessingError as exc:
                if not exc.retryable or attempt_number >= self.retry_policy.max_attempts:
                    log_event(
                        logger,
                        logging.WARNING,
                        "Inbound persistent media preprocessing failed",
                        event="inbound_persistent_media_preprocessing_failed",
                        inbound_event_id=str(event.event_id),
                        channel=event.channel,
                        user_id=str(event.user_id) if event.user_id is not None else None,
                        conversation_id=str(conversation_id),
                        media_count=len(attachments),
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
                    "Inbound persistent media preprocessing retry scheduled",
                    event="inbound_persistent_media_preprocessing_retry_scheduled",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                    conversation_id=str(conversation_id),
                    media_count=len(attachments),
                    attempt=attempt_number,
                    delay_seconds=delay_seconds,
                    error_code=exc.error_code,
                    duration_ms=elapsed_ms(started_at),
                )
                await self.sleep(delay_seconds)

    async def _process_persistent_media_once(
        self,
        *,
        event: InboundEvent,
        attachments: list[Attachment],
        conversation_id: UUID,
    ) -> None:
        if event.user_id is None:
            raise ContentProcessingError(
                "Media preprocessing requires resolved user",
                retryable=False,
                error_code="user_id_required",
            )
        if self.media_asset_store is None:
            raise ContentProcessingError(
                "Media asset store is not configured",
                retryable=False,
                error_code="media_asset_store_not_configured",
            )
        try:
            fetcher = self.media_fetchers.get(event.channel)
        except MediaFetcherNotFoundError as exc:
            raise ContentProcessingError(
                "No media fetcher is configured for inbound media",
                retryable=False,
                error_code="media_fetcher_not_found",
            ) from exc

        markers: list[str] = []
        image_media_ids: list[str] = []
        document_media_ids: list[str] = []
        image_count = sum(
            1 for attachment in attachments if attachment.attachment_type is AttachmentType.IMAGE
        )
        document_count = sum(
            1 for attachment in attachments if attachment.attachment_type is AttachmentType.DOCUMENT
        )
        for attachment in attachments:
            if attachment.attachment_type is AttachmentType.IMAGE:
                media_id, marker = await self._store_image_attachment(
                    event=event,
                    attachment=attachment,
                    conversation_id=conversation_id,
                    fetcher=fetcher,
                    index=len(image_media_ids) + 1,
                    count=image_count,
                )
                image_media_ids.append(media_id)
                markers.append(marker)
                continue
            if attachment.attachment_type is AttachmentType.DOCUMENT:
                media_id, marker = await self._store_document_attachment(
                    event=event,
                    attachment=attachment,
                    conversation_id=conversation_id,
                    fetcher=fetcher,
                    index=len(document_media_ids) + 1,
                    count=document_count,
                )
                document_media_ids.append(media_id)
                markers.append(marker)

        original_text = event.text.strip() if event.text is not None else ""
        prompt = original_text or _default_persistent_media_prompt(
            image_count=len(image_media_ids),
            document_count=len(document_media_ids),
        )
        event.text = "\n".join([*markers, prompt] if prompt else markers)
        event.attachments = []
        if image_media_ids:
            event.metadata["image_media_ids"] = image_media_ids
            event.metadata["image_processing"] = {
                "status": "prepared",
                "image_count": len(image_media_ids),
            }
        if document_media_ids:
            event.metadata["document_media_ids"] = document_media_ids
            event.metadata["document_processing"] = {
                "status": "prepared",
                "document_count": len(document_media_ids),
            }
        event.message_type = MessageType.TEXT

    async def _store_image_attachment(
        self,
        *,
        event: InboundEvent,
        attachment: Attachment,
        conversation_id: UUID,
        fetcher: ChannelMediaFetcher,
        index: int,
        count: int,
    ) -> tuple[str, str]:
        if self.image_media_store is None:
            raise ContentProcessingError(
                "Image preprocessing is not configured",
                retryable=False,
                error_code="image_preprocessing_not_configured",
            )
        self._validate_image_attachment(attachment)
        _validate_declared_size(
            attachment,
            max_size_bytes=self.max_image_size_bytes,
            label="image",
            error_code="image_too_large",
        )
        try:
            payload = await fetcher.fetch(event=event, attachment=attachment)
            if payload.size_bytes > self.max_image_size_bytes:
                raise ContentProcessingError(
                    "Inbound image exceeds configured size limit",
                    retryable=False,
                    error_code="image_too_large",
                )
            self._validate_image_payload(payload.content_type)
            media_id = _new_media_id()
            stored_media = await self.image_media_store.store(
                media_id=media_id,
                payload=payload,
            )
            await self._create_media_asset(
                event=event,
                attachment=attachment,
                conversation_id=conversation_id,
                media_id=media_id,
                media_type=MediaAssetType.IMAGE,
                stored_media=stored_media,
            )
            log_event(
                logger,
                logging.INFO,
                "Inbound image stored",
                event="inbound_image_stored",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id),
                conversation_id=str(conversation_id),
                attachment_type=attachment.attachment_type.value,
                content_type=stored_media.content_type,
                size_bytes=stored_media.size_bytes,
                media_id=media_id,
            )
            return media_id, _image_marker(media_id=media_id, index=index, count=count)
        except MediaError as exc:
            raise ContentProcessingError(
                "Inbound image could not be prepared",
                retryable=exc.retryable,
                error_code=exc.error_code,
            ) from exc

    async def _store_document_attachment(
        self,
        *,
        event: InboundEvent,
        attachment: Attachment,
        conversation_id: UUID,
        fetcher: ChannelMediaFetcher,
        index: int,
        count: int,
    ) -> tuple[str, str]:
        if self.document_media_store is None:
            raise ContentProcessingError(
                "Document preprocessing is not configured",
                retryable=False,
                error_code="document_preprocessing_not_configured",
            )
        self._validate_document_attachment(attachment)
        _validate_declared_size(
            attachment,
            max_size_bytes=self.max_document_size_bytes,
            label="document",
            error_code="document_too_large",
        )
        try:
            payload = await fetcher.fetch(event=event, attachment=attachment)
            if payload.size_bytes > self.max_document_size_bytes:
                raise ContentProcessingError(
                    "Inbound document exceeds configured size limit",
                    retryable=False,
                    error_code="document_too_large",
                )
            self._validate_document_payload(
                filename=payload.filename,
                content_type=payload.content_type,
            )
            media_id = _new_media_id()
            stored_media = await self.document_media_store.store(
                media_id=media_id,
                payload=payload,
            )
            await self._create_media_asset(
                event=event,
                attachment=attachment,
                conversation_id=conversation_id,
                media_id=media_id,
                media_type=MediaAssetType.DOCUMENT,
                stored_media=stored_media,
            )
            log_event(
                logger,
                logging.INFO,
                "Inbound document stored",
                event="inbound_document_stored",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id),
                conversation_id=str(conversation_id),
                attachment_type=attachment.attachment_type.value,
                content_type=stored_media.content_type,
                size_bytes=stored_media.size_bytes,
                media_id=media_id,
            )
            return media_id, _document_marker(
                media_id=media_id,
                filename=stored_media.filename,
                content_type=stored_media.content_type,
                index=index,
                count=count,
            )
        except MediaError as exc:
            raise ContentProcessingError(
                "Inbound document could not be prepared",
                retryable=exc.retryable,
                error_code=exc.error_code,
            ) from exc

    async def _create_media_asset(
        self,
        *,
        event: InboundEvent,
        attachment: Attachment,
        conversation_id: UUID,
        media_id: str,
        media_type: MediaAssetType,
        stored_media: StoredMedia,
    ) -> None:
        if self.media_asset_store is None or event.user_id is None:
            raise ContentProcessingError(
                "Media asset store is not configured",
                retryable=False,
                error_code="media_asset_store_not_configured",
            )
        await self.media_asset_store.create(
            asset=MediaAsset(
                media_id=media_id,
                user_id=event.user_id,
                conversation_id=conversation_id,
                media_type=media_type,
                storage_key=str(stored_media.path),
                content_type=stored_media.content_type,
                size_bytes=stored_media.size_bytes,
                source_channel=event.channel,
                source_attachment_id=attachment.attachment_id,
                source_inbound_event_id=event.event_id,
                metadata={
                    "source_message_type": event.message_type.value,
                    "source_attachment_type": attachment.attachment_type.value,
                    "original_filename": stored_media.filename,
                },
            )
        )

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

    def _validate_image_attachment(self, attachment: Attachment) -> None:
        if attachment.attachment_type is not AttachmentType.IMAGE:
            raise ContentProcessingError(
                "Attachment is not an image",
                retryable=False,
                error_code="unsupported_attachment_type",
            )
        self._validate_image_payload(attachment.content_type)

    def _validate_image_payload(self, content_type: str | None) -> None:
        if content_type is not None and content_type not in self.supported_image_content_types:
            raise ContentProcessingError(
                "Unsupported image content type",
                retryable=False,
                error_code="unsupported_image_content_type",
            )

    def _validate_document_attachment(self, attachment: Attachment) -> None:
        if attachment.attachment_type is not AttachmentType.DOCUMENT:
            raise ContentProcessingError(
                "Attachment is not a document",
                retryable=False,
                error_code="unsupported_attachment_type",
            )
        filename = _attachment_filename(attachment)
        self._validate_document_payload(
            filename=filename,
            content_type=attachment.content_type,
        )

    def _validate_document_payload(
        self,
        *,
        filename: str | None,
        content_type: str | None,
    ) -> None:
        if not is_supported_document_payload(filename=filename, content_type=content_type):
            raise ContentProcessingError(
                "Unsupported document type",
                retryable=False,
                error_code="unsupported_document_type",
            )


def event_needs_content_preprocessing(event: InboundEvent) -> bool:
    return _audio_attachment(event) is not None or bool(_persistent_attachments(event))


def _audio_attachment(event: InboundEvent) -> Attachment | None:
    if event.message_type not in {MessageType.AUDIO, MessageType.VOICE}:
        return None
    for attachment in event.attachments:
        if attachment.attachment_type in {AttachmentType.AUDIO, AttachmentType.VOICE}:
            return attachment
    return None


def _persistent_attachments(event: InboundEvent) -> list[Attachment]:
    return [
        attachment
        for attachment in event.attachments
        if attachment.attachment_type in {AttachmentType.IMAGE, AttachmentType.DOCUMENT}
    ]


def _new_media_id() -> str:
    return secrets.token_urlsafe(9)


def _validate_declared_size(
    attachment: Attachment,
    *,
    max_size_bytes: int,
    label: str,
    error_code: str,
) -> None:
    declared_size = attachment.metadata.get("file_size")
    if not isinstance(declared_size, int) or isinstance(declared_size, bool):
        return
    if declared_size > max_size_bytes:
        raise ContentProcessingError(
            f"Inbound {label} exceeds configured size limit",
            retryable=False,
            error_code=error_code,
        )


def _image_marker(*, media_id: str, index: int, count: int) -> str:
    if count == 1:
        return f'[Attached image: media_id="{media_id}"]'
    return f'[Attached image {index}: media_id="{media_id}"]'


def _document_marker(
    *,
    media_id: str,
    filename: str | None,
    content_type: str | None,
    index: int,
    count: int,
) -> str:
    label = "Attached file" if count == 1 else f"Attached file {index}"
    attributes = [f'media_id="{media_id}"']
    if filename:
        attributes.append(f"filename={json.dumps(filename, ensure_ascii=False)}")
    if content_type:
        attributes.append(f"content_type={json.dumps(content_type, ensure_ascii=False)}")
    return f"[{label}: {' '.join(attributes)}]"


def _default_persistent_media_prompt(*, image_count: int, document_count: int) -> str:
    if image_count and document_count:
        return DEFAULT_MIXED_MEDIA_PROMPT
    if document_count == 1:
        return DEFAULT_SINGLE_DOCUMENT_PROMPT
    if document_count > 1:
        return DEFAULT_MULTI_DOCUMENT_PROMPT
    return _default_image_prompt(image_count)


def _default_image_prompt(image_count: int) -> str:
    if image_count == 1:
        return DEFAULT_SINGLE_IMAGE_PROMPT
    return DEFAULT_MULTI_IMAGE_PROMPT


def _attachment_filename(attachment: Attachment) -> str | None:
    filename = attachment.metadata.get("file_name")
    if isinstance(filename, str) and filename:
        return filename
    return None
