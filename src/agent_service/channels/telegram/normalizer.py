from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from agent_service.channels.interfaces import ChannelInboundNormalizer
from agent_service.channels.models import Attachment, AttachmentType, InboundEvent, MessageType
from agent_service.channels.telegram.adapter import TELEGRAM_CHANNEL
from agent_service.channels.telegram.onboarding import TELEGRAM_ONBOARDING_CALLBACK_TEXTS


class TelegramInboundNormalizer(ChannelInboundNormalizer[Mapping[str, Any]]):
    channel = TELEGRAM_CHANNEL

    async def normalize(self, payload: Mapping[str, Any]) -> InboundEvent | None:
        update_id = _value_as_str(payload.get("update_id"))
        message = payload.get("message")
        if isinstance(message, Mapping):
            return _message_event(channel=self.channel, update_id=update_id, message=message)

        callback_query = payload.get("callback_query")
        if isinstance(callback_query, Mapping):
            return _callback_query_event(
                channel=self.channel,
                update_id=update_id,
                callback_query=callback_query,
            )

        return None


def _message_event(
    *,
    channel: str,
    update_id: str | None,
    message: Mapping[str, Any],
) -> InboundEvent | None:
    chat = message.get("chat")
    sender = message.get("from")
    if not isinstance(chat, Mapping) or not isinstance(sender, Mapping):
        return None

    chat_type = chat.get("type")
    if chat_type != "private":
        return None

    telegram_user_id = _value_as_str(sender.get("id"))
    chat_id = _value_as_str(chat.get("id"))
    message_id = _value_as_str(message.get("message_id"))
    if telegram_user_id is None or chat_id is None or message_id is None:
        return None

    normalized_content = _normalized_content(message)
    if normalized_content is None:
        return None
    message_type, text, attachments = normalized_content

    return InboundEvent(
        channel=channel,
        external_user_id=telegram_user_id,
        external_chat_id=chat_id,
        external_message_id=message_id,
        external_update_id=update_id,
        idempotency_key=f"{channel}:{chat_id}:{message_id}",
        message_type=message_type,
        text=text,
        attachments=attachments,
        thread_id=_value_as_str(message.get("message_thread_id")),
        reply_to_message_id=_reply_to_message_id(message),
        channel_metadata={
            "username": _value_as_str(sender.get("username")),
            "first_name": _value_as_str(sender.get("first_name")),
            "language_code": _value_as_str(sender.get("language_code")),
            "chat_type": chat_type,
            "media_group_id": _value_as_str(message.get("media_group_id")),
        },
        received_at=_message_received_at(message),
    )


def _callback_query_event(
    *,
    channel: str,
    update_id: str | None,
    callback_query: Mapping[str, Any],
) -> InboundEvent | None:
    callback_query_id = _value_as_str(callback_query.get("id"))
    callback_data = _value_as_str(callback_query.get("data"))
    if callback_query_id is None or callback_data is None:
        return None

    text = TELEGRAM_ONBOARDING_CALLBACK_TEXTS.get(callback_data)
    if text is None:
        return None

    sender = callback_query.get("from")
    message = callback_query.get("message")
    if not isinstance(sender, Mapping) or not isinstance(message, Mapping):
        return None

    chat = message.get("chat")
    if not isinstance(chat, Mapping):
        return None

    chat_type = chat.get("type")
    if chat_type != "private":
        return None

    telegram_user_id = _value_as_str(sender.get("id"))
    chat_id = _value_as_str(chat.get("id"))
    callback_message_id = _value_as_str(message.get("message_id"))
    if telegram_user_id is None or chat_id is None:
        return None

    idempotency_key = _callback_idempotency_key(
        channel=channel,
        chat_id=chat_id,
        callback_query_id=callback_query_id,
        callback_message_id=callback_message_id,
        callback_data=callback_data,
    )
    return InboundEvent(
        channel=channel,
        external_user_id=telegram_user_id,
        external_chat_id=chat_id,
        external_message_id=f"callback:{callback_query_id}",
        external_update_id=update_id,
        idempotency_key=idempotency_key,
        message_type=MessageType.TEXT,
        text=text,
        thread_id=_value_as_str(message.get("message_thread_id")),
        channel_metadata={
            "username": _value_as_str(sender.get("username")),
            "first_name": _value_as_str(sender.get("first_name")),
            "language_code": _value_as_str(sender.get("language_code")),
            "chat_type": chat_type,
            "callback_query_id": callback_query_id,
            "callback_data": callback_data,
            "callback_message_id": callback_message_id,
            "chat_instance": _value_as_str(callback_query.get("chat_instance")),
            "update_type": "callback_query",
        },
        metadata={
            "synthetic_user_message": True,
            "telegram_update_type": "callback_query",
            "callback_query_id": callback_query_id,
            "callback_data": callback_data,
            "callback_message_id": callback_message_id,
        },
    )


def _callback_idempotency_key(
    *,
    channel: str,
    chat_id: str,
    callback_query_id: str,
    callback_message_id: str | None,
    callback_data: str,
) -> str:
    if callback_message_id is None:
        return f"{channel}:callback:{callback_query_id}"
    return f"{channel}:callback:{chat_id}:{callback_message_id}:{callback_data}"


def _value_as_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _reply_to_message_id(message: Mapping[str, Any]) -> str | None:
    reply_to_message = message.get("reply_to_message")
    if not isinstance(reply_to_message, Mapping):
        return None
    return _value_as_str(reply_to_message.get("message_id"))


def _message_received_at(message: Mapping[str, Any]) -> datetime:
    timestamp = message.get("date")
    if isinstance(timestamp, int | float):
        return datetime.fromtimestamp(timestamp, tz=UTC)
    return datetime.now(UTC)


def _normalized_content(
    message: Mapping[str, Any],
) -> tuple[MessageType, str | None, list[Attachment]] | None:
    text = message.get("text")
    if isinstance(text, str):
        return MessageType.TEXT, text, []

    voice = message.get("voice")
    if isinstance(voice, Mapping):
        attachment = _telegram_audio_attachment(voice, attachment_type=AttachmentType.VOICE)
        if attachment is None:
            return None
        return MessageType.VOICE, None, [attachment]

    audio = message.get("audio")
    if isinstance(audio, Mapping):
        attachment = _telegram_audio_attachment(audio, attachment_type=AttachmentType.AUDIO)
        if attachment is None:
            return None
        return MessageType.AUDIO, None, [attachment]

    photo = message.get("photo")
    if isinstance(photo, list):
        attachment = _telegram_photo_attachment(photo)
        if attachment is None:
            return None
        return _image_content(message, attachment)

    document = message.get("document")
    if isinstance(document, Mapping):
        attachment = _telegram_image_document_attachment(document)
        if attachment is not None:
            return _image_content(message, attachment)
        attachment = _telegram_document_attachment(document)
        if attachment is not None:
            return _document_content(message, attachment)

    return None


def _telegram_audio_attachment(
    media: Mapping[str, Any],
    *,
    attachment_type: AttachmentType,
) -> Attachment | None:
    file_id = _value_as_str(media.get("file_id"))
    if file_id is None:
        return None

    file_unique_id = _value_as_str(media.get("file_unique_id"))
    duration = media.get("duration")
    file_size = media.get("file_size")
    file_name = _value_as_str(media.get("file_name"))
    metadata: dict[str, Any] = {
        "file_unique_id": file_unique_id,
        "duration_seconds": duration if isinstance(duration, int | float) else None,
        "file_size": file_size if isinstance(file_size, int) else None,
    }
    if file_name is not None:
        metadata["file_name"] = file_name

    return Attachment(
        attachment_id=file_unique_id or file_id,
        attachment_type=attachment_type,
        external_id=file_id,
        content_type=_value_as_str(media.get("mime_type")),
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _telegram_photo_attachment(photo_sizes: list[Any]) -> Attachment | None:
    candidates = [item for item in photo_sizes if isinstance(item, Mapping)]
    if not candidates:
        return None
    photo = max(candidates, key=_photo_size_score)
    file_id = _value_as_str(photo.get("file_id"))
    if file_id is None:
        return None
    file_unique_id = _value_as_str(photo.get("file_unique_id"))
    file_size = photo.get("file_size")
    width = photo.get("width")
    height = photo.get("height")
    metadata: dict[str, Any] = {
        "file_unique_id": file_unique_id,
        "file_size": file_size if isinstance(file_size, int) else None,
        "width": width if isinstance(width, int) else None,
        "height": height if isinstance(height, int) else None,
    }
    return Attachment(
        attachment_id=file_unique_id or file_id,
        attachment_type=AttachmentType.IMAGE,
        external_id=file_id,
        content_type="image/jpeg",
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _telegram_image_document_attachment(document: Mapping[str, Any]) -> Attachment | None:
    content_type = _value_as_str(document.get("mime_type"))
    if content_type is None or not content_type.startswith("image/"):
        return None
    file_id = _value_as_str(document.get("file_id"))
    if file_id is None:
        return None
    file_unique_id = _value_as_str(document.get("file_unique_id"))
    file_size = document.get("file_size")
    file_name = _value_as_str(document.get("file_name"))
    metadata: dict[str, Any] = {
        "file_unique_id": file_unique_id,
        "file_size": file_size if isinstance(file_size, int) else None,
    }
    if file_name is not None:
        metadata["file_name"] = file_name
    return Attachment(
        attachment_id=file_unique_id or file_id,
        attachment_type=AttachmentType.IMAGE,
        external_id=file_id,
        content_type=content_type,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _telegram_document_attachment(document: Mapping[str, Any]) -> Attachment | None:
    file_id = _value_as_str(document.get("file_id"))
    if file_id is None:
        return None
    file_unique_id = _value_as_str(document.get("file_unique_id"))
    file_size = document.get("file_size")
    file_name = _value_as_str(document.get("file_name"))
    metadata: dict[str, Any] = {
        "file_unique_id": file_unique_id,
        "file_size": file_size if isinstance(file_size, int) else None,
    }
    if file_name is not None:
        metadata["file_name"] = file_name
    return Attachment(
        attachment_id=file_unique_id or file_id,
        attachment_type=AttachmentType.DOCUMENT,
        external_id=file_id,
        content_type=_value_as_str(document.get("mime_type")),
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _image_content(
    message: Mapping[str, Any],
    attachment: Attachment,
) -> tuple[MessageType, str | None, list[Attachment]]:
    caption = message.get("caption")
    text = caption if isinstance(caption, str) else None
    return (
        MessageType.MIXED if text else MessageType.MEDIA,
        text,
        [attachment],
    )


def _document_content(
    message: Mapping[str, Any],
    attachment: Attachment,
) -> tuple[MessageType, str | None, list[Attachment]]:
    caption = message.get("caption")
    text = caption if isinstance(caption, str) else None
    return (
        MessageType.MIXED if text else MessageType.DOCUMENT,
        text,
        [attachment],
    )


def _photo_size_score(photo: Mapping[str, Any]) -> int:
    width = photo.get("width")
    height = photo.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return width * height
    file_size = photo.get("file_size")
    if isinstance(file_size, int):
        return file_size
    return 0
