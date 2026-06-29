from datetime import UTC

from agent_service.channels import AttachmentType, ChannelInboundNormalizer, MessageType
from agent_service.channels.telegram import TelegramInboundNormalizer
from agent_service.channels.telegram.onboarding import (
    TELEGRAM_ONBOARDING_CALLBACK_TEXTS,
    TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA,
    TELEGRAM_ONBOARDING_GENERATE_IMAGE_TEXT,
    TELEGRAM_ONBOARDING_LECTURE_SUMMARY_CALLBACK_DATA,
    TELEGRAM_ONBOARDING_LECTURE_SUMMARY_TEXT,
    TELEGRAM_ONBOARDING_WEB_RESEARCH_CALLBACK_DATA,
    TELEGRAM_ONBOARDING_WEB_RESEARCH_TEXT,
    TELEGRAM_START_REPLY_MARKUP,
)


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


def private_onboarding_callback_update() -> dict[str, object]:
    return {
        "update_id": 101,
        "callback_query": {
            "id": "callback-1",
            "from": {
                "id": 67890,
                "username": "handle",
                "first_name": "Anton",
                "language_code": "ru",
            },
            "message": {
                "message_id": 42,
                "date": 1_700_000_000,
                "chat": {"id": 12345, "type": "private"},
                "text": "Привет",
            },
            "chat_instance": "chat-instance-1",
            "data": TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA,
        },
    }


def test_telegram_onboarding_buttons_match_synthetic_texts() -> None:
    rows = TELEGRAM_START_REPLY_MARKUP["inline_keyboard"]

    assert rows == [
        [
            {
                "text": "🩻 Создай иллюстрацию",
                "callback_data": TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA,
            }
        ],
        [
            {
                "text": "🔎 Найди материалы по теме",
                "callback_data": TELEGRAM_ONBOARDING_WEB_RESEARCH_CALLBACK_DATA,
            }
        ],
        [
            {
                "text": "📚 Вытащи главное из лекции",
                "callback_data": TELEGRAM_ONBOARDING_LECTURE_SUMMARY_CALLBACK_DATA,
            }
        ],
    ]
    assert TELEGRAM_ONBOARDING_CALLBACK_TEXTS == {
        TELEGRAM_ONBOARDING_LECTURE_SUMMARY_CALLBACK_DATA: (
            TELEGRAM_ONBOARDING_LECTURE_SUMMARY_TEXT
        ),
        TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA: TELEGRAM_ONBOARDING_GENERATE_IMAGE_TEXT,
        TELEGRAM_ONBOARDING_WEB_RESEARCH_CALLBACK_DATA: TELEGRAM_ONBOARDING_WEB_RESEARCH_TEXT,
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


async def test_telegram_inbound_normalizer_builds_onboarding_callback_event() -> None:
    event = await TelegramInboundNormalizer().normalize(private_onboarding_callback_update())

    assert event is not None
    assert event.channel == "telegram"
    assert event.external_user_id == "67890"
    assert event.external_chat_id == "12345"
    assert event.external_message_id == "callback:callback-1"
    assert event.external_update_id == "101"
    assert event.idempotency_key == "telegram:callback:12345:42:onboarding:generate_image"
    assert event.message_type is MessageType.TEXT
    assert event.text == TELEGRAM_ONBOARDING_GENERATE_IMAGE_TEXT
    assert event.channel_metadata["callback_query_id"] == "callback-1"
    assert (
        event.channel_metadata["callback_data"] == TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA
    )
    assert event.channel_metadata["callback_message_id"] == "42"
    assert event.channel_metadata["update_type"] == "callback_query"
    assert event.metadata["synthetic_user_message"] is True
    assert event.metadata["telegram_update_type"] == "callback_query"


async def test_telegram_inbound_normalizer_ignores_unknown_callback_data() -> None:
    payload = private_onboarding_callback_update()
    callback_query = payload["callback_query"]
    assert isinstance(callback_query, dict)
    callback_query["data"] = "unknown:action"

    assert await TelegramInboundNormalizer().normalize(payload) is None


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


async def test_telegram_inbound_normalizer_builds_photo_event_with_default_image_content() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["media_group_id"] = "album-1"
    message["photo"] = [
        {
            "file_id": "small-file-id",
            "file_unique_id": "small-unique-id",
            "width": 90,
            "height": 90,
            "file_size": 1000,
        },
        {
            "file_id": "large-file-id",
            "file_unique_id": "large-unique-id",
            "width": 1280,
            "height": 720,
            "file_size": 2000,
        },
    ]

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is not None
    assert event.message_type is MessageType.MEDIA
    assert event.text is None
    assert event.channel_metadata["media_group_id"] == "album-1"
    assert len(event.attachments) == 1
    attachment = event.attachments[0]
    assert attachment.attachment_type is AttachmentType.IMAGE
    assert attachment.external_id == "large-file-id"
    assert attachment.attachment_id == "large-unique-id"
    assert attachment.content_type == "image/jpeg"
    assert attachment.metadata["width"] == 1280
    assert attachment.metadata["height"] == 720


async def test_telegram_inbound_normalizer_uses_photo_caption_as_prompt() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["caption"] = "Что тут?"
    message["photo"] = [{"file_id": "file-1", "file_unique_id": "unique-1"}]

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is not None
    assert event.message_type is MessageType.MIXED
    assert event.text == "Что тут?"


async def test_telegram_inbound_normalizer_builds_image_document_event() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["document"] = {
        "file_id": "document-file-id",
        "file_unique_id": "document-unique-id",
        "mime_type": "image/png",
        "file_name": "screen.png",
        "file_size": 1234,
    }

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is not None
    assert event.message_type is MessageType.MEDIA
    assert event.text is None
    attachment = event.attachments[0]
    assert attachment.attachment_type is AttachmentType.IMAGE
    assert attachment.external_id == "document-file-id"
    assert attachment.content_type == "image/png"
    assert attachment.metadata["file_name"] == "screen.png"


async def test_telegram_inbound_normalizer_builds_document_event() -> None:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["caption"] = "Что в файле?"
    message["document"] = {
        "file_id": "document-file-id",
        "file_unique_id": "document-unique-id",
        "mime_type": "text/markdown",
        "file_name": "notes.md",
        "file_size": 1234,
    }

    event = await TelegramInboundNormalizer().normalize(payload)

    assert event is not None
    assert event.message_type is MessageType.MIXED
    assert event.text == "Что в файле?"
    attachment = event.attachments[0]
    assert attachment.attachment_type is AttachmentType.DOCUMENT
    assert attachment.external_id == "document-file-id"
    assert attachment.attachment_id == "document-unique-id"
    assert attachment.content_type == "text/markdown"
    assert attachment.metadata["file_name"] == "notes.md"
    assert attachment.metadata["file_size"] == 1234
