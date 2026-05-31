import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from agent_service.channels import InMemoryChannelAdapterRegistry
from agent_service.conversations import AsyncioConversationLockManager
from agent_service.delivery import (
    DeliveryResult,
    DeliveryRetryPolicy,
    DeliveryStatus,
    DeliveryWorker,
)
from agent_service.messaging.in_memory import AsyncioOutboundQueue
from agent_service.outbound import OutboundEvent


@dataclass(slots=True)
class FakeAdapter:
    channel: str = "telegram"
    results: list[DeliveryResult | BaseException] = field(default_factory=list)
    events: list[OutboundEvent] = field(default_factory=list)

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        self.events.append(event)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@dataclass(slots=True)
class FakeSleep:
    delays: list[float] = field(default_factory=list)

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@dataclass(slots=True)
class BlockingAdapter:
    channel: str = "telegram"
    started_events: list[OutboundEvent] = field(default_factory=list)
    completed_events: list[OutboundEvent] = field(default_factory=list)
    active_by_conversation: dict[UUID, int] = field(default_factory=lambda: defaultdict(int))
    max_active_by_conversation: dict[UUID, int] = field(default_factory=lambda: defaultdict(int))
    active_total: int = 0
    max_active_total: int = 0
    _started_condition: asyncio.Condition = field(
        default_factory=asyncio.Condition,
        init=False,
        repr=False,
    )
    _release_events: dict[UUID, asyncio.Event] = field(default_factory=dict, init=False, repr=False)

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        async with self._started_condition:
            self.started_events.append(event)
            self.active_total += 1
            self.max_active_total = max(self.max_active_total, self.active_total)
            self.active_by_conversation[event.conversation_id] += 1
            self.max_active_by_conversation[event.conversation_id] = max(
                self.max_active_by_conversation[event.conversation_id],
                self.active_by_conversation[event.conversation_id],
            )
            self._started_condition.notify_all()

        try:
            await self._release_event(event.event_id).wait()
            return sent_result(event.event_id, channel=event.channel)
        finally:
            async with self._started_condition:
                self.active_total -= 1
                self.active_by_conversation[event.conversation_id] -= 1
                self.completed_events.append(event)
                self._started_condition.notify_all()

    async def wait_started_count(self, count: int) -> None:
        async with self._started_condition:
            await self._started_condition.wait_for(lambda: len(self.started_events) >= count)

    async def wait_completed_count(self, count: int) -> None:
        async with self._started_condition:
            await self._started_condition.wait_for(lambda: len(self.completed_events) >= count)

    def release(self, event: OutboundEvent) -> None:
        self._release_event(event.event_id).set()

    def release_all(self, events: Sequence[OutboundEvent]) -> None:
        for event in events:
            self.release(event)

    def _release_event(self, event_id: UUID) -> asyncio.Event:
        event = self._release_events.get(event_id)
        if event is None:
            event = asyncio.Event()
            self._release_events[event_id] = event
        return event


def outbound_event(*, channel: str = "telegram") -> OutboundEvent:
    return OutboundEvent(
        channel=channel,
        user_id=uuid4(),
        conversation_id=uuid4(),
        external_chat_id="12345",
        text="hello",
        trace_id="trace-1",
    )


def outbound_event_for_conversation(conversation_id: UUID) -> OutboundEvent:
    return OutboundEvent(
        channel="telegram",
        user_id=uuid4(),
        conversation_id=conversation_id,
        external_chat_id="12345",
        text="hello",
        trace_id=f"trace-{conversation_id}",
    )


def sent_result(event_id: UUID, *, channel: str = "telegram") -> DeliveryResult:
    return DeliveryResult(
        event_id=event_id,
        channel=channel,
        status=DeliveryStatus.SENT,
        external_message_ids=["501"],
    )


def retryable_result(event_id: UUID, *, channel: str = "telegram") -> DeliveryResult:
    return DeliveryResult(
        event_id=event_id,
        channel=channel,
        status=DeliveryStatus.FAILED_RETRYABLE,
        error_code="temporary_error",
        error_message="retry later",
    )


def worker(
    *,
    adapter: FakeAdapter | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    retry_policy: DeliveryRetryPolicy | None = None,
) -> DeliveryWorker:
    registry = InMemoryChannelAdapterRegistry()
    if adapter is not None:
        registry.register(adapter)
    return DeliveryWorker(
        outbound_queue=AsyncioOutboundQueue(),
        channel_adapters=registry,
        lock_manager=AsyncioConversationLockManager(),
        retry_policy=retry_policy or DeliveryRetryPolicy(),
        sleep=sleep or FakeSleep(),
    )


async def test_delivery_worker_sends_outbound_event_through_registered_adapter() -> None:
    event = outbound_event()
    adapter = FakeAdapter(results=[sent_result(event.event_id)])
    delivery_worker = worker(adapter=adapter)

    result = await delivery_worker.process_event(event)

    assert result.status is DeliveryStatus.SENT
    assert event.status is DeliveryStatus.SENT
    assert adapter.events == [event]
    assert result.external_message_ids == ["501"]


async def test_delivery_worker_dead_letters_unknown_channel_without_crashing() -> None:
    event = outbound_event(channel="slack")
    delivery_worker = worker()

    result = await delivery_worker.process_event(event)

    assert result.status is DeliveryStatus.DEAD_LETTER
    assert result.error_code == "adapter_not_found"
    assert event.status is DeliveryStatus.DEAD_LETTER


async def test_delivery_worker_retries_retryable_results_then_sends() -> None:
    event = outbound_event()
    fake_sleep = FakeSleep()
    adapter = FakeAdapter(
        results=[
            retryable_result(event.event_id),
            sent_result(event.event_id),
        ]
    )
    delivery_worker = worker(adapter=adapter, sleep=fake_sleep)

    result = await delivery_worker.process_event(event)

    assert result.status is DeliveryStatus.SENT
    assert fake_sleep.delays == [1.0]
    assert len(adapter.events) == 2
    assert event.status is DeliveryStatus.SENT


async def test_delivery_worker_converts_exhausted_retryable_failures_to_dead_letter() -> None:
    event = outbound_event()
    fake_sleep = FakeSleep()
    adapter = FakeAdapter(
        results=[
            RuntimeError("network down"),
            RuntimeError("network down"),
        ]
    )
    delivery_worker = worker(
        adapter=adapter,
        sleep=fake_sleep,
        retry_policy=DeliveryRetryPolicy(max_attempts=2, backoff_seconds=(0.25,)),
    )

    result = await delivery_worker.process_event(event)

    assert result.status is DeliveryStatus.DEAD_LETTER
    assert result.error_code == "adapter_send_exception"
    assert result.metadata["retry_exhausted"] is True
    assert fake_sleep.delays == [0.25]
    assert event.status is DeliveryStatus.DEAD_LETTER


async def test_delivery_worker_process_next_consumes_outbound_queue() -> None:
    event = outbound_event()
    adapter = FakeAdapter(results=[sent_result(event.event_id)])
    delivery_worker = worker(adapter=adapter)

    await delivery_worker.outbound_queue.publish(event)
    await delivery_worker.process_next()
    await asyncio.wait_for(delivery_worker.outbound_queue.join(), timeout=0.1)

    assert event.status is DeliveryStatus.SENT


async def test_delivery_workers_send_different_conversations_concurrently() -> None:
    first = outbound_event()
    second = outbound_event()
    adapter = BlockingAdapter()
    registry = InMemoryChannelAdapterRegistry()
    registry.register(adapter)
    queue = AsyncioOutboundQueue()
    lock_manager = AsyncioConversationLockManager()
    first_worker = DeliveryWorker(queue, registry, lock_manager)
    second_worker = DeliveryWorker(queue, registry, lock_manager)

    await queue.publish(first)
    await queue.publish(second)
    first_task = asyncio.create_task(first_worker.process_next())
    second_task = asyncio.create_task(second_worker.process_next())

    await adapter.wait_started_count(2)

    assert adapter.max_active_total == 2
    assert {event.event_id for event in adapter.started_events} == {
        first.event_id,
        second.event_id,
    }

    adapter.release_all([first, second])
    await asyncio.gather(first_task, second_task)

    assert first.status is DeliveryStatus.SENT
    assert second.status is DeliveryStatus.SENT


async def test_delivery_workers_serialize_same_conversation_delivery() -> None:
    conversation_id = uuid4()
    first = outbound_event_for_conversation(conversation_id)
    second = outbound_event_for_conversation(conversation_id)
    adapter = BlockingAdapter()
    registry = InMemoryChannelAdapterRegistry()
    registry.register(adapter)
    queue = AsyncioOutboundQueue()
    lock_manager = AsyncioConversationLockManager()
    first_worker = DeliveryWorker(queue, registry, lock_manager)
    second_worker = DeliveryWorker(queue, registry, lock_manager)

    await queue.publish(first)
    await queue.publish(second)
    first_task = asyncio.create_task(first_worker.process_next())
    second_task = asyncio.create_task(second_worker.process_next())

    await adapter.wait_started_count(1)
    await asyncio.sleep(0)

    assert adapter.started_events == [first]
    assert adapter.max_active_by_conversation[conversation_id] == 1

    adapter.release(first)
    await adapter.wait_completed_count(1)
    await adapter.wait_started_count(2)

    assert adapter.started_events == [first, second]
    assert adapter.max_active_by_conversation[conversation_id] == 1

    adapter.release(second)
    await asyncio.gather(first_task, second_task)

    assert first.status is DeliveryStatus.SENT
    assert second.status is DeliveryStatus.SENT
