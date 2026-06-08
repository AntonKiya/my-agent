from datetime import UTC

from agent_service.channels import AttachmentType, ChannelInboundNormalizer, MessageType
from agent_service.channels.telegram import TelegramInboundNormalizer


def private_text_update() -> dict[str, object]:
    return {
        "update_id": 100,
        "message": {
            "message_id": 42,
            "date": 1_700_000_000,
            "chat": {"id": 12345, "type": "private"},
            "from": {
                "id": 67890,
                "username": "handle",
                "first_name": "Anton",
            },
            "text": "hello",
        },
    }


async def test_telegram_inbound_normalizer_builds_inbound_event() -> None:
    normalizer: ChannelInboundNormalizer[dict[str, object]] = TelegramInboundNormalizer()

    event = await normalizer.normalize(private_text_update())

    assert event is not None
    assert event.channel == "telegram"
    assert event.external_user_id == "67890"
    assert event.external_chat_id == "12345"
    assert event.external_message_id == "42"
    assert event.external_update_id == "100"
    assert event.idempotency_key == "telegram:12345:42"
    assert event.message_type is MessageType.TEXT
    assert event.text == "hello"
    assert event.user_id is None
    assert event.external_user_id != event.channel_metadata["username"]
    assert event.channel_metadata["username"] == "handle"
    assert event.channel_metadata["first_name"] == "Anton"
    assert event.metadata == {}
    assert "message" not in event.channel_metadata
    assert "raw_update" not in event.channel_metadata
    assert event.received_at.tzinfo is UTC


async def test_telegram_inbound_normalizer_keeps_future_thread_and_reply_fields() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message["message_thread_id"] = 11
    message["reply_to_message"] = {"message_id": 22}

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is not None
    assert event.thread_id == "11"
    assert event.reply_to_message_id == "22"


async def test_telegram_inbound_normalizer_builds_voice_event() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["voice"] = {
        "file_id": "voice-file-id",
        "file_unique_id": "voice-unique-id",
        "duration": 7,
        "mime_type": "audio/ogg",
        "file_size": 1234,
    }

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is not None
    assert event.message_type is MessageType.VOICE
    assert event.text is None
    assert len(event.attachments) == 1
    attachment = event.attachments[0]
    assert attachment.attachment_type is AttachmentType.VOICE
    assert attachment.external_id == "voice-file-id"
    assert attachment.attachment_id == "voice-unique-id"
    assert attachment.content_type == "audio/ogg"
    assert attachment.metadata["duration_seconds"] == 7
    assert attachment.metadata["file_size"] == 1234


async def test_telegram_inbound_normalizer_builds_audio_event() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["audio"] = {
        "file_id": "audio-file-id",
        "file_unique_id": "audio-unique-id",
        "duration": 12,
        "mime_type": "audio/mpeg",
        "file_name": "clip.mp3",
    }

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is not None
    assert event.message_type is MessageType.AUDIO
    assert event.text is None
    assert len(event.attachments) == 1
    attachment = event.attachments[0]
    assert attachment.attachment_type is AttachmentType.AUDIO
    assert attachment.external_id == "audio-file-id"
    assert attachment.content_type == "audio/mpeg"
    assert attachment.metadata["file_name"] == "clip.mp3"


async def test_telegram_inbound_normalizer_ignores_unsupported_updates() -> None:
    normalizer = TelegramInboundNormalizer()
    group_payload = private_text_update()
    group_message = group_payload["message"]
    assert isinstance(group_message, dict)
    group_message["chat"] = {"id": 12345, "type": "group"}

    assert await normalizer.normalize({}) is None
    assert await normalizer.normalize({"message": {"text": "hello"}}) is None
    assert await normalizer.normalize(group_payload) is None


async def test_telegram_inbound_normalizer_ignores_media_without_text() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["photo"] = [{"file_id": "file-1"}]

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is None
