import asyncio
from uuid import uuid4

import pytest

from agent_service.channels import InboundEvent, OutboundEvent
from agent_service.messaging import (
    AsyncioInboundQueue,
    AsyncioOutboundQueue,
    InboundQueue,
    OutboundQueue,
)


def make_inbound_event(text: str = "hello") -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="123",
        external_chat_id="456",
        external_message_id="789",
        idempotency_key="telegram:456:789",
        text=text,
    )


def make_outbound_event(text: str = "response") -> OutboundEvent:
    return OutboundEvent(
        channel="telegram",
        user_id=uuid4(),
        conversation_id=uuid4(),
        external_chat_id="456",
        text=text,
    )


async def test_asyncio_inbound_queue_publishes_and_consumes_fifo() -> None:
    queue = AsyncioInboundQueue(maxsize=2)
    first = make_inbound_event("first")
    second = make_inbound_event("second")

    await queue.publish(first)
    await queue.publish(second)

    assert isinstance(queue, InboundQueue)
    assert queue.size == 2
    assert await queue.consume() == first
    assert await queue.consume() == second
    assert queue.is_empty


async def test_asyncio_outbound_queue_is_separate_from_inbound_queue() -> None:
    inbound_queue = AsyncioInboundQueue()
    outbound_queue = AsyncioOutboundQueue()
    inbound_event = make_inbound_event()
    outbound_event = make_outbound_event()

    await inbound_queue.publish(inbound_event)
    await outbound_queue.publish(outbound_event)

    assert isinstance(outbound_queue, OutboundQueue)
    assert await inbound_queue.consume() == inbound_event
    assert await outbound_queue.consume() == outbound_event


async def test_asyncio_queue_publish_waits_when_bounded_queue_is_full() -> None:
    queue = AsyncioInboundQueue(maxsize=1)
    first = make_inbound_event("first")
    second = make_inbound_event("second")

    await queue.publish(first)
    publish_task = asyncio.create_task(queue.publish(second))
    await asyncio.sleep(0)

    assert queue.is_full
    assert not publish_task.done()

    assert await queue.consume() == first
    await asyncio.wait_for(publish_task, timeout=0.1)

    assert await queue.consume() == second


def test_asyncio_queue_rejects_negative_maxsize() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        AsyncioInboundQueue(maxsize=-1)
