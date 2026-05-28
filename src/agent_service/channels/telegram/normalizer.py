from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from agent_service.channels.interfaces import ChannelInboundNormalizer
from agent_service.channels.models import InboundEvent, MessageType
from agent_service.channels.telegram.adapter import TELEGRAM_CHANNEL


class TelegramInboundNormalizer(ChannelInboundNormalizer[Mapping[str, Any]]):
    channel = TELEGRAM_CHANNEL

    async def normalize(self, payload: Mapping[str, Any]) -> InboundEvent | None:
        update_id = _value_as_str(payload.get("update_id"))
        message = payload.get("message")
        if not isinstance(message, Mapping):
            return None

        chat = message.get("chat")
        sender = message.get("from")
        text = message.get("text")
        if (
            not isinstance(chat, Mapping)
            or not isinstance(sender, Mapping)
            or not isinstance(text, str)
        ):
            return None

        chat_type = chat.get("type")
        if chat_type != "private":
            return None

        telegram_user_id = _value_as_str(sender.get("id"))
        chat_id = _value_as_str(chat.get("id"))
        message_id = _value_as_str(message.get("message_id"))
        if telegram_user_id is None or chat_id is None or message_id is None:
            return None

        return InboundEvent(
            channel=self.channel,
            external_user_id=telegram_user_id,
            external_chat_id=chat_id,
            external_message_id=message_id,
            external_update_id=update_id,
            idempotency_key=f"{self.channel}:{chat_id}:{message_id}",
            message_type=MessageType.TEXT,
            text=text,
            thread_id=_value_as_str(message.get("message_thread_id")),
            reply_to_message_id=_reply_to_message_id(message),
            channel_metadata={
                "username": _value_as_str(sender.get("username")),
                "first_name": _value_as_str(sender.get("first_name")),
                "chat_type": chat_type,
            },
            received_at=_message_received_at(message),
        )


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
