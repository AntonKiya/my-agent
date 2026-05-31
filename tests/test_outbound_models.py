from uuid import uuid4

from agent_service.channels import MessageType
from agent_service.delivery import DeliveryStatus
from agent_service.outbound import OutboundEvent, OutboundEventStatus


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
    assert event.status is DeliveryStatus.QUEUED
