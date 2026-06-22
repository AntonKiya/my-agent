import logging
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from agent_service.channels.models import Attachment, InboundEvent
from agent_service.channels.telegram.adapter import TELEGRAM_CHANNEL
from agent_service.media import ChannelMediaFetcher, MediaFetchError, MediaPayload
from agent_service.observability.events import elapsed_ms, log_event, start_timer

logger = logging.getLogger(__name__)

TELEGRAM_GET_FILE_METHOD = "getFile"
TELEGRAM_FILE_DOWNLOAD_BASE_URL = "https://api.telegram.org/file"


@dataclass(slots=True)
class TelegramMediaFetcher(ChannelMediaFetcher):
    bot_token: str | SecretStr
    client: httpx.AsyncClient
    api_base_url: str = "https://api.telegram.org"
    file_base_url: str = TELEGRAM_FILE_DOWNLOAD_BASE_URL
    max_file_size_bytes: int | None = None
    channel: str = TELEGRAM_CHANNEL

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("Telegram bot token must not be empty")
        if self.max_file_size_bytes is not None and self.max_file_size_bytes <= 0:
            raise ValueError("Telegram max file size must be greater than zero")

    @property
    def token(self) -> str:
        if isinstance(self.bot_token, SecretStr):
            return self.bot_token.get_secret_value()
        return self.bot_token

    async def fetch(self, *, event: InboundEvent, attachment: Attachment) -> MediaPayload:
        started_at = start_timer()
        if event.channel != self.channel:
            raise MediaFetchError(
                f"Telegram media fetcher cannot fetch channel {event.channel!r}",
                retryable=False,
                error_code="unsupported_channel",
            )
        if not attachment.external_id:
            raise MediaFetchError(
                "Telegram media attachment is missing file_id",
                retryable=False,
                error_code="telegram_missing_file_id",
            )

        file_info = await self._get_file(
            attachment.external_id,
            event=event,
            attachment=attachment,
        )
        file_size = _optional_int(file_info.get("file_size"))
        if (
            self.max_file_size_bytes is not None
            and file_size is not None
            and file_size > self.max_file_size_bytes
        ):
            raise MediaFetchError(
                "Telegram media file exceeds configured size limit",
                retryable=False,
                error_code="telegram_file_too_large",
            )
        file_path = file_info.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise MediaFetchError(
                "Telegram getFile response did not include file_path",
                retryable=True,
                error_code="telegram_missing_file_path",
            )

        content = await self._download_file(file_path, event=event, attachment=attachment)
        if self.max_file_size_bytes is not None and len(content) > self.max_file_size_bytes:
            raise MediaFetchError(
                "Telegram media download exceeds configured size limit",
                retryable=False,
                error_code="telegram_download_too_large",
            )
        log_event(
            logger,
            logging.INFO,
            "Telegram media fetched",
            event="telegram_media_fetched",
            inbound_event_id=str(event.event_id),
            channel=event.channel,
            user_id=str(event.user_id) if event.user_id is not None else None,
            attachment_type=attachment.attachment_type.value,
            content_type=attachment.content_type,
            declared_size_bytes=file_size,
            downloaded_size_bytes=len(content),
            duration_ms=elapsed_ms(started_at),
        )
        return MediaPayload(
            attachment=attachment,
            content=content,
            content_type=attachment.content_type,
            filename=_filename(file_path, attachment),
            metadata={
                "telegram_file_size": file_size,
                "attachment_id": attachment.attachment_id,
            },
        )

    async def _get_file(
        self,
        file_id: str,
        *,
        event: InboundEvent,
        attachment: Attachment,
    ) -> dict[str, Any]:
        try:
            response = await self.client.post(
                self._method_url(TELEGRAM_GET_FILE_METHOD),
                json={"file_id": file_id},
            )
        except httpx.TransportError as exc:
            error_code = _transport_error_code(exc)
            _log_telegram_media_fetch_transport_error(
                event=event,
                attachment=attachment,
                telegram_method=TELEGRAM_GET_FILE_METHOD,
                error_code=error_code,
                error_type=type(exc).__name__,
            )
            raise MediaFetchError(
                "Telegram getFile transport error",
                retryable=True,
                error_code=error_code,
            ) from exc

        body = _response_json(response)
        if response.status_code >= 500 or response.status_code == 429:
            error_code = _telegram_error_code(response, body)
            error_message = _telegram_error_message(response, body)
            _log_telegram_media_fetch_response_error(
                event=event,
                attachment=attachment,
                telegram_method=TELEGRAM_GET_FILE_METHOD,
                response=response,
                body=body,
                retryable=True,
            )
            raise MediaFetchError(
                error_message,
                retryable=True,
                error_code=error_code,
            )
        if response.status_code >= 400:
            error_code = _telegram_error_code(response, body)
            error_message = _telegram_error_message(response, body)
            _log_telegram_media_fetch_response_error(
                event=event,
                attachment=attachment,
                telegram_method=TELEGRAM_GET_FILE_METHOD,
                response=response,
                body=body,
                retryable=False,
            )
            raise MediaFetchError(
                error_message,
                retryable=False,
                error_code=error_code,
            )
        if not isinstance(body, dict) or body.get("ok") is not True:
            _log_telegram_media_fetch_response_error(
                event=event,
                attachment=attachment,
                telegram_method=TELEGRAM_GET_FILE_METHOD,
                response=response,
                body=body,
                retryable=True,
                fallback_error_code="telegram_get_file_failed",
                fallback_error_message="Telegram getFile response was not successful",
            )
            raise MediaFetchError(
                "Telegram getFile response was not successful",
                retryable=True,
                error_code="telegram_get_file_failed",
            )
        result = body.get("result")
        if not isinstance(result, dict):
            raise MediaFetchError(
                "Telegram getFile response result was not an object",
                retryable=True,
                error_code="telegram_invalid_file_result",
            )
        return result

    async def _download_file(
        self,
        file_path: str,
        *,
        event: InboundEvent,
        attachment: Attachment,
    ) -> bytes:
        try:
            response = await self.client.get(self._file_url(file_path))
        except httpx.TransportError as exc:
            error_code = _transport_error_code(exc)
            _log_telegram_media_fetch_transport_error(
                event=event,
                attachment=attachment,
                telegram_method="downloadFile",
                error_code=error_code,
                error_type=type(exc).__name__,
            )
            raise MediaFetchError(
                "Telegram file download transport error",
                retryable=True,
                error_code=error_code,
            ) from exc

        if response.status_code >= 500 or response.status_code == 429:
            _log_telegram_media_fetch_download_error(
                event=event,
                attachment=attachment,
                response=response,
                retryable=True,
            )
            raise MediaFetchError(
                "Telegram file download failed temporarily",
                retryable=True,
                error_code=f"telegram_file_http_{response.status_code}",
            )
        if response.status_code >= 400:
            _log_telegram_media_fetch_download_error(
                event=event,
                attachment=attachment,
                response=response,
                retryable=False,
            )
            raise MediaFetchError(
                "Telegram file download failed",
                retryable=False,
                error_code=f"telegram_file_http_{response.status_code}",
            )
        return response.content

    def _method_url(self, method: str) -> str:
        return f"{self.api_base_url.rstrip('/')}/bot{self.token}/{method}"

    def _file_url(self, file_path: str) -> str:
        safe_path = file_path.lstrip("/")
        return f"{self.file_base_url.rstrip('/')}/bot{self.token}/{safe_path}"


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _telegram_error_code(response: httpx.Response, body: Any) -> str:
    if isinstance(body, dict) and body.get("error_code") is not None:
        return f"telegram_{body['error_code']}"
    return f"telegram_http_{response.status_code}"


def _telegram_error_message(response: httpx.Response, body: Any) -> str:
    description = body.get("description") if isinstance(body, dict) else None
    if isinstance(description, str):
        return description
    return response.reason_phrase or "Telegram API request failed"


def _log_telegram_media_fetch_response_error(
    *,
    event: InboundEvent,
    attachment: Attachment,
    telegram_method: str,
    response: httpx.Response,
    body: Any,
    retryable: bool,
    fallback_error_code: str | None = None,
    fallback_error_message: str | None = None,
) -> None:
    error_code = fallback_error_code or _telegram_error_code(response, body)
    error_message = fallback_error_message or _telegram_error_message(response, body)
    log_event(
        logger,
        logging.WARNING,
        "Telegram media fetch API request failed",
        event="telegram_media_fetch_api_failed",
        **_telegram_media_fetch_log_fields(event=event, attachment=attachment),
        telegram_method=telegram_method,
        telegram_status_code=response.status_code,
        telegram_error_code=error_code,
        telegram_error_message=error_message,
        retryable=retryable,
    )


def _log_telegram_media_fetch_download_error(
    *,
    event: InboundEvent,
    attachment: Attachment,
    response: httpx.Response,
    retryable: bool,
) -> None:
    log_event(
        logger,
        logging.WARNING,
        "Telegram media fetch download failed",
        event="telegram_media_fetch_download_failed",
        **_telegram_media_fetch_log_fields(event=event, attachment=attachment),
        telegram_method="downloadFile",
        telegram_status_code=response.status_code,
        telegram_error_code=f"telegram_file_http_{response.status_code}",
        telegram_error_message=response.reason_phrase or "Telegram file download failed",
        retryable=retryable,
    )


def _log_telegram_media_fetch_transport_error(
    *,
    event: InboundEvent,
    attachment: Attachment,
    telegram_method: str,
    error_code: str,
    error_type: str,
) -> None:
    log_event(
        logger,
        logging.WARNING,
        "Telegram media fetch transport error",
        event="telegram_media_fetch_transport_error",
        **_telegram_media_fetch_log_fields(event=event, attachment=attachment),
        telegram_method=telegram_method,
        telegram_error_code=error_code,
        telegram_error_type=error_type,
        retryable=True,
    )


def _telegram_media_fetch_log_fields(
    *,
    event: InboundEvent,
    attachment: Attachment,
) -> dict[str, object]:
    return {
        "inbound_event_id": str(event.event_id),
        "channel": event.channel,
        "user_id": str(event.user_id) if event.user_id is not None else None,
        "attachment_type": attachment.attachment_type.value,
        "attachment_id": attachment.attachment_id,
        "content_type": attachment.content_type,
        "file_name": _metadata_str(attachment.metadata.get("file_name")),
        "declared_size_bytes": _optional_int(attachment.metadata.get("file_size")),
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


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _metadata_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _filename(file_path: str, attachment: Attachment) -> str | None:
    filename = attachment.metadata.get("file_name")
    if isinstance(filename, str) and filename:
        return filename
    candidate = file_path.rsplit("/", 1)[-1]
    return candidate or None
