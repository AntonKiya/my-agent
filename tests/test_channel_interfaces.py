from collections.abc import Mapping
from uuid import uuid4

from agent_service.channels import (
    ChannelAdapter,
    ChannelAdapterRegistry,
    ChannelInboundNormalizer,
    InboundEvent,
)
from agent_service.delivery import DeliveryResult, DeliveryStatus
from agent_service.messaging import QueueStats
from agent_service.messaging.interfaces import InboundQueue
from agent_service.outbound import OutboundEvent, OutboundQueue


class FakeNormalizer:
    channel = "telegram"

    async def normalize(self, payload: Mapping[str, object]) -> InboundEvent | None:
        text = payload.get("text")
        if not isinstance(text, str):
            return None
        return InboundEvent(
            channel=self.channel,
            external_user_id="123",
            external_chat_id="456",
            external_message_id="789",
            idempotency_key="telegram:456:789",
            text=text,
        )


class FakeAdapter:
    channel = "telegram"

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        return DeliveryResult(
            event_id=event.event_id,
            channel=event.channel,
            status=DeliveryStatus.SENT,
            external_message_ids=["100"],
        )


class FakeRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ChannelAdapter] = {}

    def register(self, adapter: ChannelAdapter) -> None:
        self._adapters[adapter.channel] = adapter

    def get(self, channel: str) -> ChannelAdapter:
        return self._adapters[channel]


class FakeInboundQueue:
    def __init__(self) -> None:
        self._events: list[InboundEvent] = []

    async def publish(self, event: InboundEvent) -> None:
        self._events.append(event)

    async def consume(self) -> InboundEvent:
        return self._events.pop(0)

    async def acknowledge(self) -> None:
        return None

    async def join(self) -> None:
        return None

    @property
    def stats(self) -> QueueStats:
        return QueueStats(
            size=len(self._events),
            maxsize=0,
            is_empty=not self._events,
            is_full=False,
        )


class FakeOutboundQueue:
    def __init__(self) -> None:
        self._events: list[OutboundEvent] = []

    async def publish(self, event: OutboundEvent) -> None:
        self._events.append(event)

    async def consume(self) -> OutboundEvent:
        return self._events.pop(0)

    async def acknowledge(self) -> None:
        return None

    async def join(self) -> None:
        return None

    @property
    def stats(self) -> QueueStats:
        return QueueStats(
            size=len(self._events),
            maxsize=0,
            is_empty=not self._events,
            is_full=False,
        )


async def test_channel_normalizer_protocol_accepts_transport_payload() -> None:
    normalizer: ChannelInboundNormalizer[Mapping[str, object]] = FakeNormalizer()

    event = await normalizer.normalize({"text": "hello"})

    assert isinstance(normalizer, ChannelInboundNormalizer)
    assert event is not None
    assert event.channel == "telegram"
    assert event.text == "hello"


async def test_channel_adapter_protocol_sends_outbound_event() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    adapter: ChannelAdapter = FakeAdapter()
    event = OutboundEvent(
        channel="telegram",
        user_id=user_id,
        conversation_id=conversation_id,
        external_chat_id="456",
        text="response",
    )

    result = await adapter.send(event)

    assert isinstance(adapter, ChannelAdapter)
    assert result.status is DeliveryStatus.SENT
    assert result.external_message_ids == ["100"]


def test_channel_adapter_registry_protocol_resolves_by_channel() -> None:
    registry: ChannelAdapterRegistry = FakeRegistry()
    adapter = FakeAdapter()

    registry.register(adapter)

    assert isinstance(registry, ChannelAdapterRegistry)
    assert registry.get("telegram") is adapter


async def test_inbound_queue_protocol_publishes_and_consumes_inbound_events() -> None:
    queue: InboundQueue = FakeInboundQueue()
    event = InboundEvent(
        channel="telegram",
        external_user_id="123",
        external_chat_id="456",
        idempotency_key="telegram:456:789",
        text="hello",
    )

    await queue.publish(event)

    assert isinstance(queue, InboundQueue)
    assert await queue.consume() == event


async def test_outbound_queue_protocol_publishes_and_consumes_outbound_events() -> None:
    queue: OutboundQueue = FakeOutboundQueue()
    event = OutboundEvent(
        channel="telegram",
        user_id=uuid4(),
        conversation_id=uuid4(),
        external_chat_id="456",
        text="response",
    )

    await queue.publish(event)

    assert isinstance(queue, OutboundQueue)
    assert await queue.consume() == event
