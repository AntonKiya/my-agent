import secrets
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agent_service.container import AppContainer
from agent_service.inbound import InboundIntakeStatus

from .normalizer import TelegramInboundNormalizer

router = APIRouter(prefix="/webhooks", tags=["channels"])
telegram_normalizer = TelegramInboundNormalizer()
TELEGRAM_SECRET_TOKEN_HEADER = "X-Telegram-Bot-Api-Secret-Token"


class TelegramWebhookResponse(BaseModel):
    status: Literal["accepted"]
    published: bool


@router.post("/telegram", response_model=TelegramWebhookResponse)
async def telegram_webhook(
    payload: dict[str, Any],
    request: Request,
) -> TelegramWebhookResponse:
    container = cast(AppContainer, request.app.state.container)
    _verify_webhook_secret(request=request, container=container)

    event = await telegram_normalizer.normalize(payload)
    if event is None:
        return TelegramWebhookResponse(status="accepted", published=False)

    if container.inbound_intake_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inbound intake service is not configured",
        )

    result = await container.inbound_intake_service.accept(event)
    if result.status is InboundIntakeStatus.OVERLOADED:
        raise HTTPException(
            status_code=503,
            detail="Inbound queue is overloaded",
        )
    return TelegramWebhookResponse(status="accepted", published=result.published)


def _verify_webhook_secret(*, request: Request, container: AppContainer) -> None:
    expected_secret = container.settings.telegram_webhook_secret_token
    if expected_secret is None:
        return

    received_secret = request.headers.get(TELEGRAM_SECRET_TOKEN_HEADER)
    if received_secret is None or not secrets.compare_digest(
        received_secret,
        expected_secret.get_secret_value(),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid Telegram webhook secret token",
        )
