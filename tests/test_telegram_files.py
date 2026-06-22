import logging
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest

from agent_service.channels import Attachment, AttachmentType, InboundEvent, MessageType
from agent_service.channels.telegram.files import TelegramMediaFetcher
from agent_service.media import MediaFetchError


async def test_telegram_media_fetcher_logs_get_file_api_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/getFile")
        return httpx.Response(
            400,
            json={
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: file is too big",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = TelegramMediaFetcher(bot_token="secret-token", client=client)
    event = InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        external_message_id="42",
        idempotency_key="telegram:12345:42",
        user_id=uuid4(),
        message_type=MessageType.DOCUMENT,
    )
    attachment = Attachment(
        attachment_id="unique-file-id",
        attachment_type=AttachmentType.DOCUMENT,
        external_id="file-id",
        content_type="application/pdf",
        metadata={"file_name": "deck.pdf", "file_size": 12_345_678},
    )

    try:
        with caplog.at_level(logging.WARNING, logger="agent_service.channels.telegram.files"):
            with pytest.raises(MediaFetchError) as exc_info:
                await fetcher.fetch(event=event, attachment=attachment)
    finally:
        await client.aclose()

    assert exc_info.value.error_code == "telegram_400"
    record = next(
        item
        for item in caplog.records
        if getattr(item, "event", None) == "telegram_media_fetch_api_failed"
    )
    record_fields = cast(Any, record)
    assert record_fields.telegram_method == "getFile"
    assert record_fields.telegram_status_code == 400
    assert record_fields.telegram_error_code == "telegram_400"
    assert record_fields.telegram_error_message == "Bad Request: file is too big"
    assert record_fields.attachment_type == "document"
    assert record_fields.content_type == "application/pdf"
    assert record_fields.file_name == "deck.pdf"
    assert record_fields.declared_size_bytes == 12_345_678
    assert "secret-token" not in record.getMessage()
