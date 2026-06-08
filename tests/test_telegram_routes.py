import logging
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from agent_service.app import create_app
from agent_service.channels import InboundEvent
from agent_service.config import AppSettings
from agent_service.inbound import InboundIntakeResult, InboundIntakeStatus
from agent_service.messaging.interfaces import InboundQueue
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


class DuplicateInboundIntake:
    async def accept(self, event: InboundEvent) -> InboundIntakeResult:
        return InboundIntakeResult(
            status=InboundIntakeStatus.DUPLICATE,
            published=False,
            user_resolution_status=UserResolutionStatus.RESOLVED,
            reason="duplicate inbound event",
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


def private_voice_update() -> dict[str, object]:
    payload = private_text_update()
    message = payload["message"]
    assert isinstance(message, dict)
    message.pop("text")
    message["voice"] = {
        "file_id": "voice-file-id",
        "file_unique_id": "voice-unique-id",
        "duration": 5,
        "mime_type": "audio/ogg",
    }
    return payload


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


async def test_telegram_webhook_accepts_voice_updates() -> None:
    app = create_app(AppSettings(environment="test"))
    app.state.container.inbound_intake_service = AcceptingInboundIntake(
        app.state.container.inbound_queue,
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json=private_voice_update())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "published": True}
    event = await app.state.container.inbound_queue.consume()
    assert event.text is None
    assert len(event.attachments) == 1
    assert event.attachments[0].external_id == "voice-file-id"


async def test_telegram_webhook_ignores_unsupported_updates() -> None:
    app = create_app(AppSettings(environment="test"))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json={"edited_message": {}})

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "published": False}
    assert app.state.container.inbound_queue.is_empty


async def test_telegram_webhook_rejects_missing_secret_before_normalize() -> None:
    app = create_app(
        AppSettings(
            environment="test",
            telegram_webhook_secret_token=SecretStr("secret"),
        )
    )
    app.state.container.inbound_intake_service = AcceptingInboundIntake(
        app.state.container.inbound_queue,
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json={"edited_message": {}})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid Telegram webhook secret token"
    assert app.state.container.inbound_queue.is_empty


async def test_telegram_webhook_rejects_wrong_secret() -> None:
    app = create_app(
        AppSettings(
            environment="test",
            telegram_webhook_secret_token=SecretStr("secret"),
        )
    )
    app.state.container.inbound_intake_service = AcceptingInboundIntake(
        app.state.container.inbound_queue,
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/webhooks/telegram",
            json=private_text_update(),
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )

    assert response.status_code == 401
    assert app.state.container.inbound_queue.is_empty


async def test_telegram_webhook_accepts_matching_secret() -> None:
    app = create_app(
        AppSettings(
            environment="test",
            telegram_webhook_secret_token=SecretStr("secret"),
        )
    )
    app.state.container.inbound_intake_service = AcceptingInboundIntake(
        app.state.container.inbound_queue,
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/webhooks/telegram",
            json=private_text_update(),
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "published": True}
    assert not app.state.container.inbound_queue.is_empty


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


async def test_telegram_webhook_acknowledges_duplicate_updates_without_publish() -> None:
    app = create_app(AppSettings(environment="test"))
    app.state.container.inbound_intake_service = DuplicateInboundIntake()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/webhooks/telegram", json=private_text_update())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "published": False}
    assert app.state.container.inbound_queue.is_empty


def _secret_app() -> FastAPI:
    app = create_app(
        AppSettings(
            environment="test",
            telegram_webhook_secret_token=SecretStr("secret-token-value"),
        )
    )
    app.state.container.inbound_intake_service = AcceptingInboundIntake(
        app.state.container.inbound_queue,
    )
    return app


def _rejected_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "telegram_webhook_secret_rejected"
    ]


async def test_telegram_webhook_audit_logs_wrong_secret_without_leaking_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _secret_app()
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.WARNING, logger="agent_service.channels.telegram.routes"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/webhooks/telegram",
                json=private_text_update(),
                headers={"X-Telegram-Bot-Api-Secret-Token": "attacker-supplied-value"},
            )

    assert response.status_code == 401
    records = _rejected_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert cast(Any, record).reason == "secret_mismatch"
    assert cast(Any, record).secret_header_present is True
    # Neither the configured secret nor the attacker value may appear in the log.
    for value in vars(record).values():
        assert "secret-token-value" not in str(value)
        assert "attacker-supplied-value" not in str(value)


async def test_telegram_webhook_audit_logs_missing_secret_header(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _secret_app()
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.WARNING, logger="agent_service.channels.telegram.routes"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/webhooks/telegram", json=private_text_update())

    assert response.status_code == 401
    records = _rejected_records(caplog)
    assert len(records) == 1
    assert cast(Any, records[0]).reason == "missing_secret_header"
    assert cast(Any, records[0]).secret_header_present is False
