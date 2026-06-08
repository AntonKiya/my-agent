from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest

from agent_service.channels import Attachment, AttachmentType, InboundEvent, MessageType
from agent_service.inbound import (
    ContentProcessingError,
    ContentProcessingRetryPolicy,
    InboundContentPreprocessor,
)
from agent_service.media import MediaPayload, StoredMedia, TempFileMediaStore
from agent_service.media.registry import InMemoryMediaFetcherRegistry
from agent_service.transcription import TranscriptionError, TranscriptionResult


@dataclass(slots=True)
class FakeMediaFetcher:
    channel: str = "telegram"
    payloads: list[MediaPayload] = field(default_factory=list)

    async def fetch(self, *, event: InboundEvent, attachment: Attachment) -> MediaPayload:
        if self.payloads:
            return self.payloads.pop(0)
        return MediaPayload(
            attachment=attachment,
            content=b"audio-bytes",
            content_type=attachment.content_type,
            filename="voice.ogg",
        )


@dataclass(slots=True)
class RecordingTranscriber:
    text: str = "transcribed hello"
    errors: list[TranscriptionError] = field(default_factory=list)
    seen_paths: list[Path] = field(default_factory=list)
    seen_existing: list[bool] = field(default_factory=list)

    async def transcribe(self, media: StoredMedia) -> TranscriptionResult:
        path = media.path
        self.seen_paths.append(path)
        self.seen_existing.append(path.exists())
        if self.errors:
            raise self.errors.pop(0)
        return TranscriptionResult(
            text=self.text,
            provider="test",
            model="test-model",
        )


def voice_event() -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        external_message_id="42",
        idempotency_key="telegram:12345:42",
        user_id=uuid4(),
        message_type=MessageType.VOICE,
        attachments=[
            Attachment(
                attachment_type=AttachmentType.VOICE,
                external_id="voice-file-id",
                content_type="audio/ogg",
            )
        ],
    )


def preprocessor(
    *,
    transcriber: RecordingTranscriber | None = None,
    temp_dir: Path,
) -> InboundContentPreprocessor:
    registry = InMemoryMediaFetcherRegistry()
    registry.register(FakeMediaFetcher())
    return InboundContentPreprocessor(
        media_fetchers=registry,
        audio_media_store=TempFileMediaStore(base_dir=temp_dir),
        audio_transcriber=transcriber or RecordingTranscriber(),
        retry_policy=ContentProcessingRetryPolicy(max_attempts=1),
    )


async def test_audio_preprocessor_transcribes_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    transcriber = RecordingTranscriber(text="hello from voice")
    processor = preprocessor(transcriber=transcriber, temp_dir=tmp_path)
    event = voice_event()

    await processor.process(event)

    assert event.message_type is MessageType.TEXT
    assert event.text == "hello from voice"
    assert event.attachments == []
    assert event.metadata["transcription"]["provider"] == "test"
    assert event.metadata["transcription"]["source_message_type"] == "voice"
    assert transcriber.seen_existing == [True]
    assert len(transcriber.seen_paths) == 1
    assert not transcriber.seen_paths[0].exists()


async def test_audio_preprocessor_rejects_unsupported_content_type(tmp_path: Path) -> None:
    processor = preprocessor(temp_dir=tmp_path)
    event = voice_event()
    event.attachments[0].content_type = "application/octet-stream"

    with pytest.raises(ContentProcessingError) as exc_info:
        await processor.process(event)

    assert exc_info.value.error_code == "unsupported_audio_content_type"
    assert event.text is None
    assert event.attachments


async def test_audio_preprocessor_retries_retryable_transcription_error(tmp_path: Path) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    registry = InMemoryMediaFetcherRegistry()
    registry.register(FakeMediaFetcher())
    transcriber = RecordingTranscriber(
        text="recovered",
        errors=[
            TranscriptionError(
                "temporary",
                retryable=True,
                error_code="temporary",
            )
        ],
    )
    processor = InboundContentPreprocessor(
        media_fetchers=registry,
        audio_media_store=TempFileMediaStore(base_dir=tmp_path),
        audio_transcriber=transcriber,
        retry_policy=ContentProcessingRetryPolicy(max_attempts=2, backoff_seconds=(0.2,)),
        sleep=sleep,
    )
    event = voice_event()

    await processor.process(event)

    assert event.text == "recovered"
    assert delays == [0.2]
    assert len(transcriber.seen_paths) == 2
    assert all(not path.exists() for path in transcriber.seen_paths)
