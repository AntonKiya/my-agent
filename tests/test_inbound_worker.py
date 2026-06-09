import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic_ai.messages import ModelRequest, SystemPromptPart

from agent_service.agents import (
    AgentBoundary,
    AgentContext,
    AgentRequest,
    AgentResponse,
    PydanticAIRunContext,
)
from agent_service.channels import (
    Attachment,
    AttachmentType,
    InboundEvent,
    InboundEventStatus,
    MessageType,
)
from agent_service.conversations import AsyncioConversationLockManager, Conversation
from agent_service.delivery import DeliveryResult, DeliveryStatus
from agent_service.inbound import (
    AgentRetryPolicy,
    ContentProcessingError,
    InboundContentPreprocessor,
    InboundIdempotencyClaim,
    InboundIdempotencyStore,
    InboundWorker,
    OutboundOverloadedError,
    UnresolvedInboundEventError,
)
from agent_service.memory import (
    ConversationCompactionDecision,
    ConversationCompactionPolicy,
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationSummary,
    PreparedConversationContext,
)
from agent_service.messaging.in_memory import (
    AsyncioCompactionQueue,
    AsyncioInboundQueue,
    AsyncioOutboundQueue,
)
from agent_service.observability.tracing import get_trace_id, reset_trace_id, set_trace_id
from agent_service.outbound import OutboundEvent


@dataclass(slots=True)
class FakeConversationResolver:
    conversations_by_chat_id: dict[str, Conversation]
    events: list[InboundEvent] = field(default_factory=list)

    async def resolve(self, event: InboundEvent) -> Conversation:
        self.events.append(event)
        return self.conversations_by_chat_id[event.external_chat_id]


@dataclass(slots=True)
class FakeMemoryService:
    user_messages: list[ConversationMemoryMessage] = field(default_factory=list)
    assistant_messages: list[ConversationMemoryMessage] = field(default_factory=list)
    compaction_decision: ConversationCompactionDecision | None = None

    async def record_user_message(
        self,
        *,
        conversation: Conversation,
        event: InboundEvent,
    ) -> ConversationMemoryMessage:
        if event.user_id is None:
            raise ValueError("event must be user-resolved")
        message = ConversationMemoryMessage(
            conversation_id=conversation.id,
            user_id=event.user_id,
            role=ConversationMemoryRole.USER,
            text=event.text,
            inbound_event_id=event.event_id,
            trace_id=event.trace_id,
        )
        self.user_messages.append(message)
        return message

    async def prepare_agent_context(
        self,
        *,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> PreparedConversationContext:
        return PreparedConversationContext(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            latest_user_message_id=latest_user_message.id,
            agent_context=AgentContext(system_prompt_parts=["summary"]),
            pydantic_ai=PydanticAIRunContext(
                user_prompt=latest_user_message.text,
                message_history=[ModelRequest(parts=[SystemPromptPart(content="summary")])],
                conversation_id=str(conversation.id),
                instructions=None,
            ),
        )

    async def record_assistant_message(
        self,
        *,
        conversation: Conversation,
        response: AgentResponse,
        trace_id: str | None = None,
        outbound_event_id: UUID | None = None,
    ) -> ConversationMemoryMessage:
        message = ConversationMemoryMessage(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            role=ConversationMemoryRole.ASSISTANT,
            text=response.text,
            trace_id=trace_id or response.trace_id,
            outbound_event_id=outbound_event_id,
        )
        self.assistant_messages.append(message)
        return message

    async def prepare_compaction_request(
        self,
        *,
        conversation: Conversation,
        compact_through_sequence: int | None = None,
    ) -> ConversationCompactionRequest:
        raise NotImplementedError

    async def evaluate_compaction(
        self,
        *,
        conversation: Conversation,
        policy: object,
    ) -> ConversationCompactionDecision:
        if self.compaction_decision is None:
            return ConversationCompactionDecision(
                should_compact=False,
                reason="test_disabled",
                estimated_input_tokens=0,
                usable_input_budget_tokens=1,
                trigger_tokens=1,
                recent_tail_budget_tokens=1,
            )
        return self.compaction_decision

    async def record_compaction_result(
        self,
        *,
        conversation: Conversation,
        request: ConversationCompactionRequest,
        result: ConversationCompactionResult,
        trace_id: str | None = None,
    ) -> ConversationSummary:
        raise NotImplementedError


class FailingPrepareMemoryService(FakeMemoryService):
    async def prepare_agent_context(
        self,
        *,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> PreparedConversationContext:
        raise RuntimeError("context prepare failed")


@dataclass(slots=True)
class FlakyPrepareMemoryService(FakeMemoryService):
    failures_remaining: int = 1

    async def prepare_agent_context(
        self,
        *,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> PreparedConversationContext:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("transient context prepare failed")
        return await FakeMemoryService.prepare_agent_context(
            self,
            conversation=conversation,
            latest_user_message=latest_user_message,
        )


@dataclass(slots=True)
class FakeAgentBoundary:
    responses: list[AgentResponse] = field(default_factory=list)
    errors: list[BaseException] = field(default_factory=list)
    requests: list[AgentRequest] = field(default_factory=list)

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if self.errors:
            raise self.errors.pop(0)
        if self.responses:
            return self.responses.pop(0)
        return AgentResponse(text=f"answer: {request.text}", trace_id=request.trace_id)


@dataclass(slots=True)
class FakeThinkingIndicatorSender:
    errors: list[BaseException] = field(default_factory=list)
    events: list[InboundEvent] = field(default_factory=list)

    async def send_thinking_indicator(self, event: InboundEvent) -> DeliveryResult:
        self.events.append(event)
        if self.errors:
            raise self.errors.pop(0)
        return DeliveryResult(
            event_id=event.event_id,
            channel=event.channel,
            status=DeliveryStatus.SENT,
        )


@dataclass(slots=True)
class FakeContentPreprocessor:
    text: str = "transcribed voice"
    errors: list[ContentProcessingError] = field(default_factory=list)
    events: list[InboundEvent] = field(default_factory=list)

    async def process(self, event: InboundEvent, *, conversation_id: UUID | None = None) -> None:
        self.events.append(event)
        if self.errors:
            raise self.errors.pop(0)
        event.text = self.text
        event.message_type = MessageType.TEXT
        event.attachments = []
        event.metadata["transcription"] = {
            "provider": "test",
            "model": "test-model",
            "source_message_type": "voice",
            "status": "completed",
        }


@dataclass(slots=True)
class TrackingAgentBoundary:
    entered: asyncio.Event
    release: asyncio.Event
    max_active_reached: asyncio.Event | None = None
    active_count: int = 0
    max_active_count: int = 0
    requests: list[AgentRequest] = field(default_factory=list)

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        self.active_count += 1
        self.max_active_count = max(self.max_active_count, self.active_count)
        if self.max_active_reached is not None and self.max_active_count >= 2:
            self.max_active_reached.set()
        self.entered.set()
        try:
            await self.release.wait()
            return AgentResponse(text=f"answer: {request.text}", trace_id=request.trace_id)
        finally:
            self.active_count -= 1


@dataclass(slots=True)
class FakeIdempotencyStore:
    statuses: list[tuple[UUID, InboundEventStatus, str | None]] = field(default_factory=list)

    async def claim(self, event: InboundEvent) -> InboundIdempotencyClaim:
        raise NotImplementedError

    async def release_claim(self, *, event_id: UUID) -> None:
        raise NotImplementedError

    async def mark_status(
        self,
        *,
        event_id: UUID,
        status: InboundEventStatus,
        failure_reason: str | None = None,
    ) -> None:
        self.statuses.append((event_id, status, failure_reason))


def conversation(*, user_id: UUID, chat_id: str = "12345") -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=user_id,
        channel="telegram",
        conversation_key=f"telegram:private:{chat_id}",
        external_chat_id=chat_id,
    )


def inbound_event(
    *,
    user_id: UUID | None,
    chat_id: str = "12345",
    text: str = "hello",
) -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id=chat_id,
        external_message_id="42",
        idempotency_key=f"telegram:{chat_id}:42",
        user_id=user_id,
        text=text,
        trace_id="trace-1",
    )


def voice_event(
    *,
    user_id: UUID | None,
    chat_id: str = "12345",
) -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id=chat_id,
        external_message_id="42",
        idempotency_key=f"telegram:{chat_id}:42",
        user_id=user_id,
        message_type=MessageType.VOICE,
        attachments=[
            Attachment(
                attachment_type=AttachmentType.VOICE,
                external_id="voice-file-id",
                content_type="audio/ogg",
            )
        ],
        trace_id="trace-1",
    )


def worker(
    *,
    conversations_by_chat_id: dict[str, Conversation],
    agent_boundary: AgentBoundary,
    memory_service: FakeMemoryService | None = None,
    compaction_queue: AsyncioCompactionQueue | None = None,
    compaction_policy: ConversationCompactionPolicy | None = None,
    idempotency_store: InboundIdempotencyStore | None = None,
    retry_policy: AgentRetryPolicy | None = None,
    thinking_indicator_sender: FakeThinkingIndicatorSender | None = None,
    content_preprocessor: FakeContentPreprocessor | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[InboundWorker, AsyncioInboundQueue, AsyncioOutboundQueue, FakeMemoryService]:
    inbound_queue = AsyncioInboundQueue()
    outbound_queue = AsyncioOutboundQueue()
    memory = memory_service or FakeMemoryService()
    return (
        InboundWorker(
            inbound_queue=inbound_queue,
            outbound_queue=outbound_queue,
            conversation_resolver=FakeConversationResolver(conversations_by_chat_id),
            memory_service=memory,
            agent_boundary=agent_boundary,
            lock_manager=AsyncioConversationLockManager(),
            idempotency_store=idempotency_store,
            retry_policy=retry_policy or AgentRetryPolicy(max_attempts=1),
            thinking_indicator_sender=thinking_indicator_sender,
            content_preprocessor=cast(InboundContentPreprocessor | None, content_preprocessor),
            compaction_queue=compaction_queue,
            compaction_policy=compaction_policy,
            sleep=sleep or asyncio.sleep,
        ),
        inbound_queue,
        outbound_queue,
        memory,
    )


async def test_inbound_worker_processes_event_to_outbound_queue() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    agent = FakeAgentBoundary(responses=[AgentResponse(text="answer", metadata={"m": "v"})])
    inbound_worker, _inbound_queue, outbound_queue, memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert event.status is InboundEventStatus.COMPLETED
    assert outbound.user_id == user_id
    assert outbound.conversation_id == resolved_conversation.id
    assert outbound.external_chat_id == "12345"
    assert outbound.text == "answer"
    assert outbound.metadata == {"m": "v"}
    assert outbound.trace_id == "trace-1"
    assert memory.user_messages[0].inbound_event_id == event.event_id
    assert memory.assistant_messages[0].outbound_event_id == outbound.event_id
    assert agent.requests[0].conversation_id == resolved_conversation.id
    assert agent.requests[0].pydantic_ai is not None
    assert agent.requests[0].pydantic_ai.user_prompt == "hello"
    assert "external_chat_id" not in agent.requests[0].model_dump()
    assert "raw_update" not in agent.requests[0].model_dump()


async def test_inbound_worker_sends_best_effort_thinking_indicator() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    thinking_sender = FakeThinkingIndicatorSender()
    inbound_worker, _inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=FakeAgentBoundary(responses=[AgentResponse(text="answer")]),
        thinking_indicator_sender=thinking_sender,
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert thinking_sender.events == [event]
    assert outbound.text == "answer"
    assert event.status is InboundEventStatus.COMPLETED


async def test_inbound_worker_preprocesses_voice_before_memory_and_agent() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    preprocessor = FakeContentPreprocessor(text="hello from voice")
    agent = FakeAgentBoundary(responses=[AgentResponse(text="answer")])
    inbound_worker, _inbound_queue, outbound_queue, memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        content_preprocessor=preprocessor,
    )
    event = voice_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert outbound.text == "answer"
    assert event.status is InboundEventStatus.COMPLETED
    assert event.text == "hello from voice"
    assert event.attachments == []
    assert memory.user_messages[0].text == "hello from voice"
    assert agent.requests[0].text == "hello from voice"
    assert agent.requests[0].attachments == []
    assert preprocessor.events == [event]


async def test_inbound_worker_fallbacks_when_voice_preprocessor_is_missing() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    agent = FakeAgentBoundary()
    inbound_worker, _inbound_queue, outbound_queue, memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
    )
    event = voice_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert event.status is InboundEventStatus.FALLBACK_SENT
    assert outbound.text == inbound_worker.fallback_text
    assert memory.user_messages == []
    assert agent.requests == []


async def test_inbound_worker_fallbacks_when_voice_preprocessing_fails() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    preprocessor = FakeContentPreprocessor(
        errors=[
            ContentProcessingError(
                "no speech",
                retryable=False,
                error_code="empty_transcription",
            )
        ]
    )
    agent = FakeAgentBoundary()
    inbound_worker, _inbound_queue, outbound_queue, memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        content_preprocessor=preprocessor,
    )
    event = voice_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert event.status is InboundEventStatus.FALLBACK_SENT
    assert outbound.text == inbound_worker.fallback_text
    assert event.metadata["content_processing"]["error_code"] == "empty_transcription"
    assert memory.user_messages == []
    assert agent.requests == []


async def test_inbound_worker_continues_when_thinking_indicator_fails() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    thinking_sender = FakeThinkingIndicatorSender(errors=[RuntimeError("draft down")])
    agent = FakeAgentBoundary(responses=[AgentResponse(text="answer")])
    inbound_worker, _inbound_queue, outbound_queue, memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        thinking_indicator_sender=thinking_sender,
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert thinking_sender.events == [event]
    assert outbound.text == "answer"
    assert len(agent.requests) == 1
    assert len(memory.assistant_messages) == 1
    assert event.status is InboundEventStatus.COMPLETED


async def test_inbound_worker_updates_persistent_idempotency_statuses() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    idempotency_store = FakeIdempotencyStore()
    inbound_worker, _inbound_queue, _outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=FakeAgentBoundary(responses=[AgentResponse(text="answer")]),
        idempotency_store=idempotency_store,
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    assert idempotency_store.statuses == [
        (event.event_id, InboundEventStatus.PROCESSING, None),
        (event.event_id, InboundEventStatus.COMPLETED, None),
    ]


async def test_inbound_worker_emits_safe_observability_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agent_service.inbound.worker")
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    sensitive_text = "sensitive user message"
    inbound_worker, _inbound_queue, _outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=FakeAgentBoundary(),
    )

    await inbound_worker.process_event(inbound_event(user_id=user_id, text=sensitive_text))

    events = {getattr(record, "event", None): record for record in caplog.records}
    assert "agent_run_completed" in events
    assert "outbound_event_published" in events
    assert "inbound_event_processed" in events
    for record in events.values():
        assert sensitive_text not in record.getMessage()
        assert not hasattr(record, "text")
        assert not hasattr(record, "raw_update")
        assert not hasattr(record, "prompt")
    assert events["inbound_event_processed"].__dict__["user_id"] == str(user_id)


async def test_inbound_worker_schedules_compaction_after_successful_agent_run() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    compaction_queue = AsyncioCompactionQueue()
    memory = FakeMemoryService(
        compaction_decision=ConversationCompactionDecision(
            should_compact=True,
            reason="trigger_reached",
            estimated_input_tokens=90,
            usable_input_budget_tokens=100,
            trigger_tokens=80,
            recent_tail_budget_tokens=30,
            compact_through_sequence=7,
            keep_from_sequence=8,
            compactable_token_count=60,
            retained_tail_token_count=30,
        )
    )
    inbound_worker, _inbound_queue, _outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=FakeAgentBoundary(),
        memory_service=memory,
        compaction_queue=compaction_queue,
        compaction_policy=ConversationCompactionPolicy(enabled=True),
    )

    await inbound_worker.process_event(inbound_event(user_id=user_id))

    job = await compaction_queue.consume()
    assert job.conversation == resolved_conversation
    assert job.compact_through_sequence == 7
    assert job.reason == "trigger_reached"
    assert job.metadata["keep_from_sequence"] == 8
    assert job.metadata["retained_tail_token_count"] == 30


async def test_inbound_worker_process_next_consumes_from_inbound_queue() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    inbound_worker, inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=FakeAgentBoundary(),
    )
    event = inbound_event(user_id=user_id)

    await inbound_queue.publish(event)
    await inbound_worker.process_next()

    outbound = await outbound_queue.consume()
    assert outbound.text == "answer: hello"
    assert event.status is InboundEventStatus.COMPLETED


async def test_inbound_worker_rejects_unresolved_event() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    inbound_worker, _inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=FakeAgentBoundary(),
    )
    event = inbound_event(user_id=None)

    with pytest.raises(UnresolvedInboundEventError):
        await inbound_worker.process_event(event)

    assert event.status is InboundEventStatus.DEAD_LETTER
    assert outbound_queue.is_empty


async def test_inbound_worker_creates_trace_id_and_resets_trace_context() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    agent = FakeAgentBoundary()
    inbound_worker, _inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
    )
    event = inbound_event(user_id=user_id)
    event.trace_id = None

    assert get_trace_id() is None

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert event.trace_id is not None
    assert agent.requests[0].trace_id == event.trace_id
    assert outbound.trace_id == event.trace_id
    assert get_trace_id() is None


async def test_inbound_worker_restores_existing_trace_context() -> None:
    outer_token = set_trace_id("outer-trace")
    try:
        user_id = uuid4()
        resolved_conversation = conversation(user_id=user_id)
        agent = FakeAgentBoundary()
        inbound_worker, _inbound_queue, _outbound_queue, _memory = worker(
            conversations_by_chat_id={"12345": resolved_conversation},
            agent_boundary=agent,
        )

        await inbound_worker.process_event(inbound_event(user_id=user_id))

        assert get_trace_id() == "outer-trace"
    finally:
        reset_trace_id(outer_token)


async def test_inbound_worker_stops_before_agent_when_context_prepare_fails() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    agent = FakeAgentBoundary()
    memory = FailingPrepareMemoryService()
    inbound_worker, _inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        memory_service=memory,
    )
    event = inbound_event(user_id=user_id)

    with pytest.raises(RuntimeError, match="context prepare failed"):
        await inbound_worker.process_event(event)

    assert event.status is InboundEventStatus.FAILED_RETRYABLE
    assert len(memory.user_messages) == 1
    assert memory.assistant_messages == []
    assert agent.requests == []
    assert outbound_queue.is_empty


async def test_inbound_worker_times_out_when_outbound_queue_is_full() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    outbound_queue = AsyncioOutboundQueue(maxsize=1)
    memory = FakeMemoryService()
    inbound_worker = InboundWorker(
        inbound_queue=AsyncioInboundQueue(),
        outbound_queue=outbound_queue,
        conversation_resolver=FakeConversationResolver({"12345": resolved_conversation}),
        memory_service=memory,
        agent_boundary=FakeAgentBoundary(responses=[AgentResponse(text="answer")]),
        lock_manager=AsyncioConversationLockManager(),
        retry_policy=AgentRetryPolicy(max_attempts=1),
        outbound_publish_timeout_seconds=0.01,
    )
    # Saturate the outbound queue so the worker's publish cannot complete.
    await outbound_queue.publish(
        OutboundEvent(
            channel="telegram",
            user_id=user_id,
            conversation_id=resolved_conversation.id,
            external_chat_id="12345",
            text="blocking",
        )
    )
    event = inbound_event(user_id=user_id)

    with pytest.raises(OutboundOverloadedError):
        await inbound_worker.process_event(event)

    assert event.status is InboundEventStatus.FAILED_RETRYABLE
    assert memory.user_messages[0].inbound_event_id == event.event_id
    # No assistant message is persisted when delivery never happened.
    assert memory.assistant_messages == []
    assert outbound_queue.stats.is_full


async def test_inbound_worker_run_forever_continues_after_event_error() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    agent = FakeAgentBoundary()
    memory = FlakyPrepareMemoryService()
    inbound_worker, inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        memory_service=memory,
        sleep=lambda _delay: asyncio.sleep(0),
    )
    first = inbound_event(user_id=user_id, text="first")
    second = inbound_event(user_id=user_id, text="second")

    await inbound_queue.publish(first)
    await inbound_queue.publish(second)
    task = asyncio.create_task(inbound_worker.run_forever())
    try:
        outbound = await asyncio.wait_for(outbound_queue.consume(), timeout=0.1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert first.status is InboundEventStatus.FAILED_RETRYABLE
    assert second.status is InboundEventStatus.COMPLETED
    assert outbound.text == "answer: second"
    assert len(agent.requests) == 1


def test_inbound_worker_rejects_negative_error_backoff() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    inbound_queue = AsyncioInboundQueue()
    outbound_queue = AsyncioOutboundQueue()

    with pytest.raises(ValueError):
        InboundWorker(
            inbound_queue=inbound_queue,
            outbound_queue=outbound_queue,
            conversation_resolver=FakeConversationResolver({"12345": resolved_conversation}),
            memory_service=FakeMemoryService(),
            agent_boundary=FakeAgentBoundary(),
            lock_manager=AsyncioConversationLockManager(),
            error_backoff_seconds=-1,
        )


def test_inbound_worker_rejects_negative_outbound_publish_timeout() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)

    with pytest.raises(ValueError):
        InboundWorker(
            inbound_queue=AsyncioInboundQueue(),
            outbound_queue=AsyncioOutboundQueue(),
            conversation_resolver=FakeConversationResolver({"12345": resolved_conversation}),
            memory_service=FakeMemoryService(),
            agent_boundary=FakeAgentBoundary(),
            lock_manager=AsyncioConversationLockManager(),
            outbound_publish_timeout_seconds=-1,
        )


async def test_inbound_worker_retries_agent_before_success() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    agent = FakeAgentBoundary(
        errors=[RuntimeError("first"), RuntimeError("second")],
        responses=[AgentResponse(text="recovered")],
    )
    inbound_worker, _inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        retry_policy=AgentRetryPolicy(max_attempts=3, backoff_seconds=(0.1, 0.2)),
        sleep=sleep,
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert outbound.text == "recovered"
    assert event.status is InboundEventStatus.COMPLETED
    assert delays == [0.1, 0.2]
    assert len(agent.requests) == 3


async def test_inbound_worker_treats_agent_timeout_as_retryable_failure() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    agent = FakeAgentBoundary(
        errors=[TimeoutError("provider timed out")],
        responses=[AgentResponse(text="recovered")],
    )
    inbound_worker, _inbound_queue, outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        retry_policy=AgentRetryPolicy(max_attempts=2, backoff_seconds=(0.1,)),
        sleep=sleep,
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert outbound.text == "recovered"
    assert event.status is InboundEventStatus.COMPLETED
    assert delays == [0.1]
    assert len(agent.requests) == 2


async def test_inbound_worker_publishes_fallback_after_agent_failure() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    agent = FakeAgentBoundary(errors=[RuntimeError("first"), RuntimeError("second")])
    inbound_worker, _inbound_queue, outbound_queue, memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
        retry_policy=AgentRetryPolicy(max_attempts=2, backoff_seconds=(0,)),
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    outbound = await outbound_queue.consume()
    assert event.status is InboundEventStatus.FALLBACK_SENT
    assert outbound.text == inbound_worker.fallback_text
    assert outbound.metadata["fallback"] is True
    assert outbound.conversation_id == resolved_conversation.id
    assert memory.assistant_messages == []


async def test_inbound_worker_marks_fallback_status_in_idempotency_store() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    idempotency_store = FakeIdempotencyStore()
    inbound_worker, _inbound_queue, _outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=FakeAgentBoundary(errors=[RuntimeError("agent failed")]),
        idempotency_store=idempotency_store,
    )
    event = inbound_event(user_id=user_id)

    await inbound_worker.process_event(event)

    assert idempotency_store.statuses == [
        (event.event_id, InboundEventStatus.PROCESSING, None),
        (
            event.event_id,
            InboundEventStatus.FALLBACK_SENT,
            "agent call failed after retries",
        ),
    ]


async def test_inbound_worker_serializes_same_conversation_agent_runs() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id)
    entered = asyncio.Event()
    release = asyncio.Event()
    agent = TrackingAgentBoundary(entered=entered, release=release)
    inbound_worker, _inbound_queue, _outbound_queue, _memory = worker(
        conversations_by_chat_id={"12345": resolved_conversation},
        agent_boundary=agent,
    )

    first_task = asyncio.create_task(
        inbound_worker.process_event(inbound_event(user_id=user_id, text="first"))
    )
    await asyncio.wait_for(entered.wait(), timeout=0.1)
    second_task = asyncio.create_task(
        inbound_worker.process_event(inbound_event(user_id=user_id, text="second"))
    )
    await asyncio.sleep(0)

    assert agent.max_active_count == 1
    assert len(agent.requests) == 1

    release.set()
    await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=0.1)

    assert agent.max_active_count == 1
    assert len(agent.requests) == 2


async def test_inbound_worker_allows_different_conversations_in_parallel() -> None:
    user_id = uuid4()
    first_conversation = conversation(user_id=user_id, chat_id="111")
    second_conversation = conversation(user_id=user_id, chat_id="222")
    entered = asyncio.Event()
    release = asyncio.Event()
    max_active_reached = asyncio.Event()
    agent = TrackingAgentBoundary(
        entered=entered,
        release=release,
        max_active_reached=max_active_reached,
    )
    inbound_worker, _inbound_queue, _outbound_queue, _memory = worker(
        conversations_by_chat_id={
            "111": first_conversation,
            "222": second_conversation,
        },
        agent_boundary=agent,
    )

    first_task = asyncio.create_task(
        inbound_worker.process_event(inbound_event(user_id=user_id, chat_id="111"))
    )
    second_task = asyncio.create_task(
        inbound_worker.process_event(inbound_event(user_id=user_id, chat_id="222"))
    )

    await asyncio.wait_for(max_active_reached.wait(), timeout=0.1)

    release.set()
    await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=0.1)

    assert agent.max_active_count == 2
