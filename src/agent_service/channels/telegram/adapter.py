from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from agent_service.channels.interfaces import ChannelAdapter
from agent_service.delivery.models import DeliveryResult, DeliveryStatus
from agent_service.outbound.models import OutboundEvent

TELEGRAM_CHANNEL = "telegram"
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_SEND_MESSAGE_METHOD = "sendMessage"


@dataclass(slots=True)
class TelegramSendAttempt:
    status: DeliveryStatus
    external_message_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = None


@dataclass(slots=True)
class TelegramAdapter(ChannelAdapter):
    bot_token: str | SecretStr
    client: httpx.AsyncClient
    api_base_url: str = "https://api.telegram.org"
    parse_mode: str | None = None
    channel: str = TELEGRAM_CHANNEL

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("Telegram bot token must not be empty")

    @property
    def token(self) -> str:
        if isinstance(self.bot_token, SecretStr):
            return self.bot_token.get_secret_value()
        return self.bot_token

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        if event.channel != self.channel:
            return self._dead_letter(
                event,
                error_code="unsupported_channel",
                error_message=f"Telegram adapter cannot send channel {event.channel!r}",
            )
        if event.attachments:
            return self._dead_letter(
                event,
                error_code="unsupported_attachments",
                error_message="Telegram adapter currently supports text messages only",
            )
        if event.text is None or not event.text:
            return self._dead_letter(
                event,
                error_code="empty_text",
                error_message="Telegram text message must not be empty",
            )

        external_message_ids: list[str] = []
        for chunk in split_telegram_text(event.text):
            attempt = await self._send_chunk(event, chunk)
            if attempt.status is not DeliveryStatus.SENT:
                return DeliveryResult(
                    event_id=event.event_id,
                    channel=self.channel,
                    status=attempt.status,
                    external_message_ids=external_message_ids,
                    error_code=attempt.error_code,
                    error_message=attempt.error_message,
                    retry_after_seconds=attempt.retry_after_seconds,
                    metadata={"partial_delivery": bool(external_message_ids)},
                )
            if attempt.external_message_id is not None:
                external_message_ids.append(attempt.external_message_id)

        return DeliveryResult(
            event_id=event.event_id,
            channel=self.channel,
            status=DeliveryStatus.SENT,
            external_message_ids=external_message_ids,
        )

    async def _send_chunk(self, event: OutboundEvent, text: str) -> TelegramSendAttempt:
        try:
            response = await self.client.post(
                self._method_url(TELEGRAM_SEND_MESSAGE_METHOD),
                json=self._send_message_payload(event, text),
            )
        except httpx.TransportError as exc:
            return TelegramSendAttempt(
                status=DeliveryStatus.FAILED_RETRYABLE,
                error_code="telegram_transport_error",
                error_message=str(exc),
            )

        body = self._response_json(response)
        if response.status_code >= 500 or response.status_code == 429:
            return TelegramSendAttempt(
                status=DeliveryStatus.FAILED_RETRYABLE,
                error_code=self._telegram_error_code(response, body),
                error_message=self._telegram_error_message(response, body),
                retry_after_seconds=self._retry_after_seconds(body),
            )
        if response.status_code >= 400:
            return TelegramSendAttempt(
                status=DeliveryStatus.DEAD_LETTER,
                error_code=self._telegram_error_code(response, body),
                error_message=self._telegram_error_message(response, body),
            )
        if not isinstance(body, dict):
            return TelegramSendAttempt(
                status=DeliveryStatus.FAILED_RETRYABLE,
                error_code="telegram_invalid_json",
                error_message="Telegram API response is not a JSON object",
            )
        if body.get("ok") is not True:
            return TelegramSendAttempt(
                status=DeliveryStatus.DEAD_LETTER,
                error_code=self._telegram_error_code(response, body),
                error_message=self._telegram_error_message(response, body),
                retry_after_seconds=self._retry_after_seconds(body),
            )

        message = body.get("result")
        external_message_id = None
        if isinstance(message, dict) and message.get("message_id") is not None:
            external_message_id = str(message["message_id"])

        return TelegramSendAttempt(
            status=DeliveryStatus.SENT,
            external_message_id=external_message_id,
        )

    def _send_message_payload(self, event: OutboundEvent, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": event.external_chat_id,
            "text": text,
        }
        if self.parse_mode is not None:
            payload["parse_mode"] = self.parse_mode

        thread_id = _numeric_string_to_int(event.thread_id)
        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        reply_to_message_id = _numeric_string_to_int(event.reply_to_message_id)
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id

        return payload

    def _method_url(self, method: str) -> str:
        return f"{self.api_base_url.rstrip('/')}/bot{self.token}/{method}"

    def _dead_letter(
        self,
        event: OutboundEvent,
        *,
        error_code: str,
        error_message: str,
    ) -> DeliveryResult:
        return DeliveryResult(
            event_id=event.event_id,
            channel=self.channel,
            status=DeliveryStatus.DEAD_LETTER,
            error_code=error_code,
            error_message=error_message,
        )

    def _response_json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _telegram_error_code(self, response: httpx.Response, body: Any) -> str:
        if isinstance(body, dict) and body.get("error_code") is not None:
            return f"telegram_{body['error_code']}"
        return f"telegram_http_{response.status_code}"

    def _telegram_error_message(self, response: httpx.Response, body: Any) -> str:
        if isinstance(body, dict):
            description = body.get("description")
            if isinstance(description, str):
                return description
        return response.reason_phrase or "Telegram API request failed"

    def _retry_after_seconds(self, body: Any) -> float | None:
        if not isinstance(body, dict):
            return None
        parameters = body.get("parameters")
        if not isinstance(parameters, dict):
            return None
        retry_after = parameters.get("retry_after")
        if isinstance(retry_after, int | float) and retry_after > 0:
            return float(retry_after)
        return None


def split_telegram_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    if limit < 1:
        raise ValueError("Telegram text split limit must be greater than zero")
    return [text[index : index + limit] for index in range(0, len(text), limit)]


def _numeric_string_to_int(value: str | None) -> int | None:
    if value is None:
        return None
    if not value.isdecimal():
        return None
    return int(value)
