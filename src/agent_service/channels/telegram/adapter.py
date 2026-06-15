from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from pydantic import SecretStr

from agent_service.channels.interfaces import ChannelAdapter
from agent_service.channels.models import InboundEvent
from agent_service.channels.telegram.formatting import (
    TELEGRAM_HTML_PARSE_MODE,
    markdown_to_telegram_html,
)
from agent_service.delivery.models import DeliveryResult, DeliveryStatus
from agent_service.outbound.models import OutboundEvent

TELEGRAM_CHANNEL = "telegram"
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_SEND_MESSAGE_METHOD = "sendMessage"
TELEGRAM_SEND_RICH_MESSAGE_METHOD = "sendRichMessage"
TELEGRAM_SEND_MESSAGE_DRAFT_METHOD = "sendMessageDraft"
TELEGRAM_MAX_DRAFT_ID = 2_147_483_647
TELEGRAM_THINKING_DRAFT_CUSTOM_EMOJI_ID = "5443038326535759644"
TELEGRAM_THINKING_DRAFT_TEXT = (
    f'Думаю <tg-emoji emoji-id="{TELEGRAM_THINKING_DRAFT_CUSTOM_EMOJI_ID}">💬</tg-emoji>'
)


@dataclass(slots=True)
class TelegramSendAttempt:
    status: DeliveryStatus
    external_message_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class TelegramAdapter(ChannelAdapter):
    bot_token: str | SecretStr
    client: httpx.AsyncClient
    api_base_url: str = "https://api.telegram.org"
    parse_mode: str | None = None
    render_markdown: bool = False
    rich_messages_enabled: bool = False
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
                    metadata={
                        **(attempt.metadata or {}),
                        "partial_delivery": bool(external_message_ids),
                    },
                )
            if attempt.external_message_id is not None:
                external_message_ids.append(attempt.external_message_id)

        return DeliveryResult(
            event_id=event.event_id,
            channel=self.channel,
            status=DeliveryStatus.SENT,
            external_message_ids=external_message_ids,
        )

    async def send_thinking_indicator(self, event: InboundEvent) -> DeliveryResult:
        if event.channel != self.channel:
            return DeliveryResult(
                event_id=event.event_id,
                channel=self.channel,
                status=DeliveryStatus.DEAD_LETTER,
                error_code="unsupported_channel",
                error_message=f"Telegram adapter cannot draft channel {event.channel!r}",
            )

        attempt = await self._send_message_draft(
            event,
            text=TELEGRAM_THINKING_DRAFT_TEXT,
            draft_id=telegram_draft_id(event.event_id),
        )
        return DeliveryResult(
            event_id=event.event_id,
            channel=self.channel,
            status=attempt.status,
            error_code=attempt.error_code,
            error_message=attempt.error_message,
            retry_after_seconds=attempt.retry_after_seconds,
            metadata=attempt.metadata or {},
        )

    async def _send_chunk(self, event: OutboundEvent, text: str) -> TelegramSendAttempt:
        if self.rich_messages_enabled and should_send_rich_message(text):
            rich_attempt = await self._send_rich_chunk(event, text)
            if not _should_fallback_from_rich_attempt(rich_attempt):
                return rich_attempt

        return await self._send_regular_chunk(event, self._format_text(text))

    async def _send_regular_chunk(self, event: OutboundEvent, text: str) -> TelegramSendAttempt:
        try:
            response = await self.client.post(
                self._method_url(TELEGRAM_SEND_MESSAGE_METHOD),
                json=self._send_message_payload(event, text),
            )
        except httpx.TransportError as exc:
            return TelegramSendAttempt(
                status=DeliveryStatus.FAILED_RETRYABLE,
                error_code=_transport_error_code(exc),
                error_message=_transport_error_message(exc),
                metadata={"error_type": type(exc).__name__},
            )

        return self._send_attempt_from_response(response)

    async def _send_rich_chunk(self, event: OutboundEvent, text: str) -> TelegramSendAttempt:
        try:
            response = await self.client.post(
                self._method_url(TELEGRAM_SEND_RICH_MESSAGE_METHOD),
                json=self._send_rich_message_payload(event, text),
            )
        except httpx.TransportError as exc:
            return TelegramSendAttempt(
                status=DeliveryStatus.FAILED_RETRYABLE,
                error_code=_transport_error_code(exc),
                error_message=_transport_error_message(exc),
                metadata={"error_type": type(exc).__name__},
            )

        return self._send_attempt_from_response(response)

    async def _send_message_draft(
        self,
        event: InboundEvent,
        *,
        text: str,
        draft_id: int,
    ) -> TelegramSendAttempt:
        try:
            response = await self.client.post(
                self._method_url(TELEGRAM_SEND_MESSAGE_DRAFT_METHOD),
                json=self._send_message_draft_payload(event, text=text, draft_id=draft_id),
            )
        except httpx.TransportError as exc:
            return TelegramSendAttempt(
                status=DeliveryStatus.FAILED_RETRYABLE,
                error_code=_transport_error_code(exc),
                error_message=_transport_error_message(exc),
                metadata={"error_type": type(exc).__name__},
            )

        return self._send_attempt_from_response(response)

    def _send_attempt_from_response(self, response: httpx.Response) -> TelegramSendAttempt:
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

    def _send_message_draft_payload(
        self,
        event: InboundEvent,
        *,
        text: str,
        draft_id: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": event.external_chat_id,
            "draft_id": draft_id,
            "parse_mode": TELEGRAM_HTML_PARSE_MODE,
            "text": text,
        }

        thread_id = _numeric_string_to_int(event.thread_id)
        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        return payload

    def _send_message_payload(self, event: OutboundEvent, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": event.external_chat_id,
            "text": text,
        }
        if self.parse_mode is not None:
            payload["parse_mode"] = self.parse_mode
        elif self.render_markdown:
            payload["parse_mode"] = TELEGRAM_HTML_PARSE_MODE

        thread_id = _numeric_string_to_int(event.thread_id)
        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        reply_to_message_id = _numeric_string_to_int(event.reply_to_message_id)
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id

        return payload

    def _send_rich_message_payload(self, event: OutboundEvent, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": event.external_chat_id,
            "rich_message": {"markdown": text},
        }

        thread_id = _numeric_string_to_int(event.thread_id)
        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        reply_to_message_id = _numeric_string_to_int(event.reply_to_message_id)
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {"message_id": reply_to_message_id}

        return payload

    def _format_text(self, text: str) -> str:
        if not self.render_markdown:
            return text
        return markdown_to_telegram_html(text)

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


def telegram_draft_id(event_id: UUID) -> int:
    return event_id.int % TELEGRAM_MAX_DRAFT_ID + 1


def should_send_rich_message(text: str) -> bool:
    return has_markdown_table(text) or has_block_math(text)


def has_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    in_code_block = False
    previous_table_columns: int | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            previous_table_columns = None
            continue
        if in_code_block:
            continue

        table_columns = _table_row_column_count(stripped)
        if table_columns is not None:
            if previous_table_columns is not None and _is_markdown_table_delimiter(stripped):
                return previous_table_columns == table_columns
            previous_table_columns = table_columns
            continue

        previous_table_columns = None

    return False


def has_block_math(text: str) -> bool:
    in_code_block = False
    open_math_block = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        delimiter_count = stripped.count("$$")
        if delimiter_count >= 2:
            return True
        if delimiter_count == 1:
            if open_math_block:
                return True
            open_math_block = True

    return False


def _table_row_column_count(stripped_line: str) -> int | None:
    if "|" not in stripped_line:
        return None
    cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
    if len(cells) < 2 or any(cell == "" for cell in cells):
        return None
    return len(cells)


def _is_markdown_table_delimiter(stripped_line: str) -> bool:
    cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
    if len(cells) < 2:
        return False
    return all(_is_markdown_table_delimiter_cell(cell) for cell in cells)


def _is_markdown_table_delimiter_cell(cell: str) -> bool:
    if len(cell) < 3:
        return False
    if cell.startswith(":"):
        cell = cell[1:]
    if cell.endswith(":"):
        cell = cell[:-1]
    return len(cell) >= 3 and set(cell) == {"-"}


def _numeric_string_to_int(value: str | None) -> int | None:
    if value is None:
        return None
    if not value.isdecimal():
        return None
    return int(value)


def _should_fallback_from_rich_attempt(attempt: TelegramSendAttempt) -> bool:
    return attempt.status is DeliveryStatus.DEAD_LETTER and attempt.error_code in {
        "telegram_400",
        "telegram_http_400",
        "telegram_404",
        "telegram_http_404",
    }


def _transport_error_code(exc: httpx.TransportError) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return "telegram_connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "telegram_read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "telegram_write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "telegram_pool_timeout"
    if isinstance(exc, httpx.TimeoutException):
        return "telegram_timeout"
    if isinstance(exc, httpx.NetworkError):
        return "telegram_network_error"
    if isinstance(exc, httpx.ProtocolError):
        return "telegram_protocol_error"
    return "telegram_transport_error"


def _transport_error_message(exc: httpx.TransportError) -> str:
    message = str(exc)
    error_type = type(exc).__name__
    if not message:
        return f"Telegram API transport error ({error_type})"
    return f"Telegram API transport error ({error_type}): {_redact_bot_token(message)}"


def _redact_bot_token(message: str) -> str:
    marker = "/bot"
    method_separator = "/"
    start = message.find(marker)
    if start == -1:
        return message

    token_start = start + len(marker)
    token_end = message.find(method_separator, token_start)
    if token_end == -1:
        return f"{message[:token_start]}<redacted>"
    return f"{message[:token_start]}<redacted>{message[token_end:]}"
