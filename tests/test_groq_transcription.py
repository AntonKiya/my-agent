from pathlib import Path

from agent_service.media import StoredMedia
from agent_service.transcription.groq import _multipart_filename


def test_groq_transcription_normalizes_telegram_oga_filename() -> None:
    media = StoredMedia(
        path=Path("/tmp/file.oga"),
        content_type="audio/ogg",
        filename="voice/file_123.oga",
        size_bytes=123,
    )

    assert _multipart_filename(media) == "audio.ogg"


def test_groq_transcription_keeps_supported_filename() -> None:
    media = StoredMedia(
        path=Path("/tmp/file.ogg"),
        content_type="audio/ogg",
        filename="voice.ogg",
        size_bytes=123,
    )

    assert _multipart_filename(media) == "voice.ogg"
