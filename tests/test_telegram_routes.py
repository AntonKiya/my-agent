from uuid import uuid4

import httpx

from agent_service.app import create_app
from agent_service.channels import InboundEvent
from agent_service.config import AppSettings
from agent_service.inbound import InboundIntakeResult, InboundIntakeStatus
from agent_service.messaging import InboundQueue
from agent_service.users import UserResolutionStatus


class AcceptingInboundIntake:
    def __init__(self, queue: InboundQueue) -> None:
        self._queue = queue
        self.user_id = uuid4()

    async def accept(self, event: InboundEvent) -> InboundIntakeResult:
        await self._queue.publish(event.model_copy(update={"user_id": self.user_id}))
        return InboundIntakeResult(
            status=InboundIntakeStatus.PUBLISHED,
            published=True,
            user_resolution_status=UserResolutionStatus.RESOLVED,
        )


class OverloadedInboundIntake:
    async def accept(self, event: InboundEvent) -> InboundIntakeResult:
        return InboundIntakeResult(
            status=InboundIntakeStatus.OVERLOADED,
            published=False,
            user_resolution_status=UserResolutionStatus.RESOLVED,
            reason="inbound queue is overloaded",
            queue_size=1,
            queue_maxsize=1,
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


async def test_create_app_registers_telegram_webhook_route() -> None:
    app = create_app(AppSettings(environment="test"))
    app.state.container.inbound_intake_service = AcceptingInboundIntake(
        app.state.container.inbound_queue,
    )
    transport = httpx.ASGITransport(app=app)

    assert app.state.container.telegram_adapter is None

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json=private_text_update())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "published": True}

    event = await app.state.container.inbound_queue.consume()
    assert event.channel == "telegram"
    assert event.text == "hello"
    assert event.external_user_id == "67890"
    assert event.user_id is not None
    assert event.channel_metadata["username"] == "handle"


async def test_telegram_webhook_does_not_require_send_adapter_or_bot_token() -> None:
    app = create_app(AppSettings(environment="test", telegram_bot_token=None))
    app.state.container.inbound_intake_service = AcceptingInboundIntake(
        app.state.container.inbound_queue,
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json=private_text_update())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "published": True}
    assert app.state.container.telegram_adapter is None
    assert not app.state.container.inbound_queue.is_empty


async def test_telegram_webhook_ignores_unsupported_updates() -> None:
    app = create_app(AppSettings(environment="test"))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json={"edited_message": {}})

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "published": False}
    assert app.state.container.inbound_queue.is_empty


async def test_telegram_webhook_requires_inbound_intake_for_supported_updates() -> None:
    app = create_app(AppSettings(environment="test"))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json=private_text_update())

    assert response.status_code == 503
    assert app.state.container.inbound_queue.is_empty


async def test_telegram_webhook_returns_503_when_inbound_queue_is_overloaded() -> None:
    app = create_app(AppSettings(environment="test"))
    app.state.container.inbound_intake_service = OverloadedInboundIntake()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json=private_text_update())

    assert response.status_code == 503
    assert response.json()["detail"] == "Inbound queue is overloaded"
    assert app.state.container.inbound_queue.is_empty
