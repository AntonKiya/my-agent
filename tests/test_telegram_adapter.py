import json
from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest

from agent_service.channels import Attachment, ChannelAdapter
from agent_service.channels.telegram.adapter import TELEGRAM_TEXT_LIMIT, TelegramAdapter
from agent_service.channels.telegram.formatting import markdown_to_telegram_html
from agent_service.delivery import DeliveryStatus
from agent_service.outbound import OutboundEvent


def make_outbound_event(
    *,
    text: str | None = "hello",
    channel: str = "telegram",
    external_chat_id: str = "12345",
) -> OutboundEvent:
    return OutboundEvent(
        channel=channel,
        user_id=uuid4(),
        conversation_id=uuid4(),
        external_chat_id=external_chat_id,
        text=text,
    )


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_telegram_adapter_sends_text_message() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 100}},
        )

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event(text="hello"))

    assert result.status is DeliveryStatus.SENT
    assert result.external_message_ids == ["100"]
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.telegram.org/bottoken/sendMessage"
    assert _request_json(requests[0]) == {
        "chat_id": "12345",
        "text": "hello",
    }


async def test_telegram_adapter_satisfies_channel_adapter_protocol() -> None:
    async with make_client(lambda _request: httpx.Response(200)) as client:
        adapter: ChannelAdapter = TelegramAdapter(bot_token="token", client=client)

    assert isinstance(adapter, ChannelAdapter)


async def test_telegram_adapter_splits_long_text_messages() -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        text = _request_json(request)["text"]
        assert isinstance(text, str)
        texts.append(text)
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": len(texts)}},
        )

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event(text="a" * (TELEGRAM_TEXT_LIMIT + 5)))

    assert result.status is DeliveryStatus.SENT
    assert result.external_message_ids == ["1", "2"]
    assert [len(text) for text in texts] == [TELEGRAM_TEXT_LIMIT, 5]


async def test_telegram_adapter_returns_retryable_api_errors_without_retrying() -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            429,
            json={
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests",
                "parameters": {"retry_after": 2},
            },
        )

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event())

    assert result.status is DeliveryStatus.FAILED_RETRYABLE
    assert result.error_code == "telegram_429"
    assert result.retry_after_seconds == 2
    assert request_count == 1


async def test_telegram_adapter_returns_retryable_failure_for_temporary_errors() -> None:
    async with make_client(lambda _request: httpx.Response(503, json={"ok": False})) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event())

    assert result.status is DeliveryStatus.FAILED_RETRYABLE
    assert result.error_code == "telegram_http_503"


async def test_telegram_adapter_dead_letters_non_retryable_errors() -> None:
    async with make_client(
        lambda _request: httpx.Response(
            400,
            json={"ok": False, "error_code": 400, "description": "Bad Request"},
        )
    ) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event())

    assert result.status is DeliveryStatus.DEAD_LETTER
    assert result.error_code == "telegram_400"
    assert result.error_message == "Bad Request"


async def test_telegram_adapter_rejects_non_telegram_events() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event(channel="slack"))

    assert result.status is DeliveryStatus.DEAD_LETTER
    assert result.error_code == "unsupported_channel"
    assert requests == []


async def test_telegram_adapter_rejects_unsupported_payloads() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        empty = await adapter.send(make_outbound_event(text=None))
        with_attachment = make_outbound_event(text="hello")
        with_attachment.attachments.append(Attachment())
        attachments = await adapter.send(with_attachment)

    assert empty.status is DeliveryStatus.DEAD_LETTER
    assert empty.error_code == "empty_text"
    assert attachments.status is DeliveryStatus.DEAD_LETTER
    assert attachments.error_code == "unsupported_attachments"
    assert requests == []


async def test_telegram_adapter_reports_partial_delivery_when_later_chunk_fails() -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": 100}},
            )
        return httpx.Response(
            400,
            json={"ok": False, "error_code": 400, "description": "Bad Request"},
        )

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event(text="a" * (TELEGRAM_TEXT_LIMIT + 1)))

    assert result.status is DeliveryStatus.DEAD_LETTER
    assert result.external_message_ids == ["100"]
    assert result.metadata["partial_delivery"] is True


async def test_telegram_adapter_allows_retry_after_partial_delivery() -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": 100}},
            )
        return httpx.Response(503, json={"ok": False})

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client)
        result = await adapter.send(make_outbound_event(text="a" * (TELEGRAM_TEXT_LIMIT + 1)))

    assert result.status is DeliveryStatus.FAILED_RETRYABLE
    assert result.error_code == "telegram_http_503"
    assert result.external_message_ids == ["100"]
    assert result.metadata["partial_delivery"] is True


async def test_telegram_adapter_includes_optional_thread_and_reply_ids() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(_request_json(request))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 100}})

    event = make_outbound_event(text="hello")
    event.thread_id = "11"
    event.reply_to_message_id = "22"

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client, parse_mode="HTML")
        result = await adapter.send(event)

    assert result.status is DeliveryStatus.SENT
    assert payloads == [
        {
            "chat_id": "12345",
            "text": "hello",
            "parse_mode": "HTML",
            "message_thread_id": 11,
            "reply_to_message_id": 22,
        }
    ]


async def test_telegram_adapter_can_render_markdown_as_telegram_html() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(_request_json(request))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 100}})

    async with make_client(handler) as client:
        adapter = TelegramAdapter(bot_token="token", client=client, render_markdown=True)
        result = await adapter.send(
            make_outbound_event(text='## Несколько мыслей\n**Позиционирование:** <safe>')
        )

    assert result.status is DeliveryStatus.SENT
    assert payloads == [
        {
            "chat_id": "12345",
            "text": "<b>Несколько мыслей</b>\n<b>Позиционирование:</b> &lt;safe&gt;",
            "parse_mode": "HTML",
        }
    ]


def test_markdown_to_telegram_html_escapes_plain_text() -> None:
    assert markdown_to_telegram_html("2 < 3 & **yes**") == "2 &lt; 3 &amp; <b>yes</b>"


def test_telegram_adapter_rejects_invalid_configuration() -> None:
    client = make_client(lambda _request: httpx.Response(200))

    with pytest.raises(ValueError, match="token"):
        TelegramAdapter(bot_token="", client=client)


def _request_json(request: httpx.Request) -> dict[str, object]:
    value = json.loads(request.content.decode())
    assert isinstance(value, dict)
    return value
