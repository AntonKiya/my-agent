from typing import Any, Literal, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agent_service.container import AppContainer

from .normalizer import TelegramInboundNormalizer

router = APIRouter(prefix="/webhooks", tags=["channels"])
telegram_normalizer = TelegramInboundNormalizer()


class TelegramWebhookResponse(BaseModel):
    status: Literal["accepted"]
    published: bool


@router.post("/telegram", response_model=TelegramWebhookResponse)
async def telegram_webhook(
    payload: dict[str, Any],
    request: Request,
) -> TelegramWebhookResponse:
    event = await telegram_normalizer.normalize(payload)
    if event is None:
        return TelegramWebhookResponse(status="accepted", published=False)

    container = cast(AppContainer, request.app.state.container)
    await container.inbound_queue.publish(event)
    return TelegramWebhookResponse(status="accepted", published=True)
