import asyncio
import logging
import secrets
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agent_service.container import AppContainer
from agent_service.delivery.models import DeliveryStatus
from agent_service.inbound import InboundIntakeStatus
from agent_service.observability.events import business_span, elapsed_ms, log_event, start_timer
from agent_service.observability.tracing import create_trace_id, reset_trace_id, set_trace_id

from .normalizer import TelegramInboundNormalizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["channels"])
telegram_normalizer = TelegramInboundNormalizer()
TELEGRAM_SECRET_TOKEN_HEADER = "X-Telegram-Bot-Api-Secret-Token"
TELEGRAM_CALLBACK_ANSWER_TIMEOUT_SECONDS = 1.0


class TelegramWebhookResponse(BaseModel):
    status: Literal["accepted"]
    published: bool


@router.post("/telegram", response_model=TelegramWebhookResponse)
async def telegram_webhook(
    payload: dict[str, Any],
    request: Request,
) -> TelegramWebhookResponse:
    started_at = start_timer()
    container = cast(AppContainer, request.app.state.container)
    with business_span(
        "Telegram webhook",
        event="telegram_webhook",
        channel="telegram",
        client_host=request.client.host if request.client is not None else None,
    ):
        _verify_webhook_secret(request=request, container=container)

        event = await telegram_normalizer.normalize(payload)
        callback_query_id = _callback_query_id(payload)
        if callback_query_id is not None:
            await _answer_callback_query(
                container=container,
                callback_query_id=callback_query_id,
                inbound_event_id=(str(event.event_id) if event is not None else None),
            )

        if event is None:
            log_event(
                logger,
                logging.INFO,
                "Telegram webhook ignored",
                event="telegram_webhook_ignored",
                channel="telegram",
                reason=_unsupported_update_reason(payload),
                duration_ms=elapsed_ms(started_at),
            )
            return TelegramWebhookResponse(status="accepted", published=False)

        trace_id = event.trace_id or create_trace_id()
        token = set_trace_id(trace_id)
        event.trace_id = trace_id
        try:
            log_event(
                logger,
                logging.INFO,
                "Telegram webhook received",
                event="telegram_webhook_received",
                channel=event.channel,
                inbound_event_id=str(event.event_id),
                external_update_id=event.external_update_id,
                external_message_id=event.external_message_id,
                message_type=event.message_type.value,
            )

            if container.inbound_intake_service is None:
                log_event(
                    logger,
                    logging.ERROR,
                    "Telegram webhook cannot be accepted without inbound intake",
                    event="telegram_webhook_intake_unconfigured",
                    channel=event.channel,
                    inbound_event_id=str(event.event_id),
                )
                raise HTTPException(
                    status_code=503,
                    detail="Inbound intake service is not configured",
                )

            result = await container.inbound_intake_service.accept(event)
            if result.status is InboundIntakeStatus.OVERLOADED:
                log_event(
                    logger,
                    logging.WARNING,
                    "Telegram webhook rejected because inbound queue is overloaded",
                    event="telegram_webhook_overloaded",
                    channel=event.channel,
                    inbound_event_id=str(event.event_id),
                    queue_size=result.queue_size,
                    queue_maxsize=result.queue_maxsize,
                    duration_ms=elapsed_ms(started_at),
                )
                raise HTTPException(
                    status_code=503,
                    detail="Inbound queue is overloaded",
                )
            log_event(
                logger,
                logging.INFO,
                "Telegram webhook accepted",
                event="telegram_webhook_accepted",
                channel=event.channel,
                inbound_event_id=str(event.event_id),
                intake_status=result.status.value,
                published=result.published,
                queue_size=result.queue_size,
                queue_maxsize=result.queue_maxsize,
                duration_ms=elapsed_ms(started_at),
            )
            return TelegramWebhookResponse(status="accepted", published=result.published)
        finally:
            reset_trace_id(token)


async def _answer_callback_query(
    *,
    container: AppContainer,
    callback_query_id: str,
    inbound_event_id: str | None,
) -> None:
    adapter = container.telegram_adapter
    if adapter is None:
        log_event(
            logger,
            logging.DEBUG,
            "Telegram callback answer skipped",
            event="telegram_callback_answer_skipped",
            reason="telegram_adapter_not_configured",
            inbound_event_id=inbound_event_id,
        )
        return
    try:
        attempt = await asyncio.wait_for(
            adapter.answer_callback_query(callback_query_id),
            timeout=TELEGRAM_CALLBACK_ANSWER_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        log_event(
            logger,
            logging.DEBUG,
            "Telegram callback answer skipped",
            event="telegram_callback_answer_skipped",
            reason="answer_callback_query_failed",
            inbound_event_id=inbound_event_id,
            error_type=type(exc).__name__,
        )
        return
    if attempt.status is DeliveryStatus.SENT:
        log_event(
            logger,
            logging.DEBUG,
            "Telegram callback answered",
            event="telegram_callback_answered",
            inbound_event_id=inbound_event_id,
        )
        return
    log_event(
        logger,
        logging.DEBUG,
        "Telegram callback answer skipped",
        event="telegram_callback_answer_skipped",
        reason="telegram_callback_answer_not_sent",
        inbound_event_id=inbound_event_id,
        status=attempt.status.value,
        error_code=attempt.error_code,
        retry_after_seconds=attempt.retry_after_seconds,
    )


def _verify_webhook_secret(*, request: Request, container: AppContainer) -> None:
    expected_secret = container.settings.telegram_webhook_secret_token
    if expected_secret is None:
        # Only reachable in the "test" environment: AppSettings requires the
        # secret in every other environment, so real deployments always verify.
        log_event(
            logger,
            logging.INFO,
            "Telegram webhook secret verification skipped",
            event="telegram_webhook_secret_skipped",
            reason="secret_not_configured",
            client_host=request.client.host if request.client is not None else None,
        )
        return

    received_secret = request.headers.get(TELEGRAM_SECRET_TOKEN_HEADER)
    if received_secret is None or not secrets.compare_digest(
        received_secret,
        expected_secret.get_secret_value(),
    ):
        log_event(
            logger,
            logging.WARNING,
            "Telegram webhook secret verification failed",
            event="telegram_webhook_secret_rejected",
            reason="missing_secret_header" if received_secret is None else "secret_mismatch",
            secret_header_present=received_secret is not None,
            client_host=request.client.host if request.client is not None else None,
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid Telegram webhook secret token",
        )
    log_event(
        logger,
        logging.INFO,
        "Telegram webhook secret verified",
        event="telegram_webhook_secret_verified",
        secret_header_present=True,
        client_host=request.client.host if request.client is not None else None,
    )


def _unsupported_update_reason(payload: dict[str, Any]) -> str:
    if "callback_query" in payload:
        return "unsupported_callback_query"
    if "message" not in payload:
        return "missing_message"
    message = payload.get("message")
    if not isinstance(message, dict):
        return "invalid_message"
    if not _has_supported_message_content(message):
        return "non_text_message"
    chat = message.get("chat")
    if isinstance(chat, dict) and chat.get("type") != "private":
        return "non_private_chat"
    return "unsupported_message_shape"


def _callback_query_id(payload: dict[str, Any]) -> str | None:
    callback_query = payload.get("callback_query")
    if not isinstance(callback_query, dict):
        return None
    callback_query_id = callback_query.get("id")
    if isinstance(callback_query_id, str) and callback_query_id:
        return callback_query_id
    return None


def _has_supported_message_content(message: dict[str, Any]) -> bool:
    return (
        isinstance(message.get("text"), str)
        or isinstance(message.get("voice"), dict)
        or isinstance(message.get("audio"), dict)
    )
