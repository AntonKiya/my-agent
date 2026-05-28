from datetime import UTC
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_service.channels import (
    Attachment,
    AttachmentType,
    DeliveryResult,
    DeliveryStatus,
    InboundEvent,
    InboundEventStatus,
    MessageType,
    OutboundEvent,
    OutboundEventStatus,
)


def test_inbound_event_captures_transport_identity_without_trusting_username() -> None:
    user_id = uuid4()

    event = InboundEvent(
        channel="telegram",
        external_user_id="12345",
        external_chat_id="67890",
        external_message_id="42",
        external_update_id="99",
        idempotency_key="telegram:67890:42",
        user_id=user_id,
        text="hello",
        channel_metadata={
            "username": "handle",
            "first_name": "Anton",
            "raw_update_stored": False,
        },
    )

    assert event.user_id == user_id
    assert event.channel == "telegram"
    assert event.external_user_id == "12345"
    assert event.idempotency_key == "telegram:67890:42"
    assert event.channel_metadata["username"] == "handle"
    assert event.status is InboundEventStatus.QUEUED
    assert event.received_at.tzinfo is UTC


def test_channel_event_defaults_are_isolated_between_instances() -> None:
    first = InboundEvent(
        channel="telegram",
        external_user_id="1",
        external_chat_id="10",
        idempotency_key="telegram:10:1",
        text="first",
    )
    second = InboundEvent(
        channel="telegram",
        external_user_id="2",
        external_chat_id="20",
        idempotency_key="telegram:20:2",
        text="second",
    )

    first.attachments.append(Attachment(attachment_type=AttachmentType.DOCUMENT))
    first.metadata["key"] = "value"

    assert second.attachments == []
    assert second.metadata == {}


def test_inbound_event_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        InboundEvent.model_validate(
            {
                "channel": "telegram",
                "external_user_id": "1",
                "external_chat_id": "10",
                "idempotency_key": "telegram:10:1",
                "text": "hello",
                "raw_update": {"unsafe": True},
            }
        )


def test_outbound_event_contains_delivery_target_and_trace_context() -> None:
    user_id = uuid4()
    conversation_id = uuid4()

    event = OutboundEvent(
        channel="telegram",
        user_id=user_id,
        conversation_id=conversation_id,
        external_chat_id="67890",
        text="response",
        trace_id="trace-1",
    )

    assert event.user_id == user_id
    assert event.conversation_id == conversation_id
    assert event.external_chat_id == "67890"
    assert event.message_type is MessageType.TEXT
    assert event.status is OutboundEventStatus.QUEUED


def test_delivery_result_records_split_messages_and_retry_errors() -> None:
    event_id = uuid4()

    sent = DeliveryResult(
        event_id=event_id,
        channel="telegram",
        status=DeliveryStatus.SENT,
        external_message_ids=["101", "102"],
    )
    retryable = DeliveryResult(
        event_id=event_id,
        channel="telegram",
        status=DeliveryStatus.FAILED_RETRYABLE,
        error_code="too_many_requests",
        error_message="retry later",
        retry_after_seconds=3,
    )

    assert sent.external_message_ids == ["101", "102"]
    assert retryable.error_code == "too_many_requests"
    assert retryable.retry_after_seconds == 3
