from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_service.delivery import DeliveryResult, DeliveryStatus


def test_delivery_status_contains_full_lifecycle() -> None:
    assert [status.value for status in DeliveryStatus] == [
        "queued",
        "sending",
        "sent",
        "failed_retryable",
        "dead_letter",
    ]


def test_delivery_result_accepts_adapter_outcomes_only() -> None:
    event_id = uuid4()

    sent = DeliveryResult(
        event_id=event_id,
        channel="telegram",
        status=DeliveryStatus.SENT,
        external_message_ids=["101"],
    )
    retryable = DeliveryResult(
        event_id=event_id,
        channel="telegram",
        status=DeliveryStatus.FAILED_RETRYABLE,
        error_code="telegram_429",
        error_message="retry later",
        retry_after_seconds=3,
    )

    assert sent.terminal is True
    assert sent.retryable is False
    assert sent.external_message_ids == ["101"]
    assert retryable.terminal is False
    assert retryable.retryable is True
    assert retryable.error_code == "telegram_429"
    assert retryable.retry_after_seconds == 3

    with pytest.raises(ValidationError):
        DeliveryResult(
            event_id=event_id,
            channel="telegram",
            status=DeliveryStatus.SENDING,
        )
