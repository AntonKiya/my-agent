import asyncio
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID, uuid4

from agent_service.agents import AgentContext, AgentRequest, AgentResponse, PydanticAIRunContext
from agent_service.channels import InboundEvent, InMemoryChannelAdapterRegistry
from agent_service.conversations import AsyncioConversationLockManager, Conversation
from agent_service.delivery import DeliveryResult, DeliveryStatus, DeliveryWorker
from agent_service.inbound import AgentRetryPolicy, InboundWorker
from agent_service.memory import (
    ConversationCompactionDecision,
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationMemoryService,
    ConversationSummary,
    PreparedConversationContext,
)
from agent_service.messaging.in_memory import AsyncioInboundQueue, AsyncioOutboundQueue
from agent_service.outbound import OutboundEvent


@dataclass(slots=True)
class MappingConversationResolver:
    conversations_by_chat_id: dict[str, Conversation]

    async def resolve(self, event: InboundEvent) -> Conversation:
        return self.conversations_by_chat_id[event.external_chat_id]


@dataclass(slots=True)
class RecordingMemoryService:
    user_messages: list[ConversationMemoryMessage] = field(default_factory=list)
    assistant_messages: list[ConversationMemoryMessage] = field(default_factory=list)

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
            agent_context=AgentContext(system_prompt_parts=["safe summary"]),
            pydantic_ai=PydanticAIRunContext(
                user_prompt=latest_user_message.text,
                conversation_id=str(conversation.id),
                instructions="safe summary",
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
        raise NotImplementedError

    async def record_compaction_result(
        self,
        *,
        conversation: Conversation,
        request: ConversationCompactionRequest,
        result: ConversationCompactionResult,
        trace_id: str | None = None,
    ) -> ConversationSummary:
        raise NotImplementedError


@dataclass(slots=True)
class RecordingAgent:
    requests: list[AgentRequest] = field(default_factory=list)

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            text=f"user={request.user_id};conversation={request.conversation_id}",
            metadata={"agent_metadata": "safe"},
            trace_id=request.trace_id,
        )


@dataclass(slots=True)
class RecordingDeliveryAdapter:
    channel: str = "telegram"
    events: list[OutboundEvent] = field(default_factory=list)

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        self.events.append(event)
        return DeliveryResult(
            event_id=event.event_id,
            channel=event.channel,
            status=DeliveryStatus.SENT,
            external_message_ids=[f"sent-{event.external_chat_id}"],
        )


def conversation(*, user_id: UUID, chat_id: str) -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=user_id,
        channel="telegram",
        conversation_key=f"telegram:private:{chat_id}",
        external_chat_id=chat_id,
    )


def inbound_event(
    *,
    user_id: UUID,
    chat_id: str,
    message_id: str,
    text: str,
) -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id=f"external-{user_id}",
        external_chat_id=chat_id,
        external_message_id=message_id,
        external_update_id=f"update-{message_id}",
        idempotency_key=f"telegram:{chat_id}:{message_id}",
        user_id=user_id,
        text=text,
        thread_id="future-thread",
        reply_to_message_id="future-reply",
        channel_metadata={
            "username": "mutable_handle",
            "first_name": "Mutable",
            "raw_update": {"secret": "transport-shape"},
        },
        metadata={"raw_payload": {"secret": "do-not-pass-to-agent"}},
        trace_id=f"trace-{message_id}",
    )


def inbound_worker(
    *,
    inbound_queue: AsyncioInboundQueue,
    outbound_queue: AsyncioOutboundQueue,
    conversations_by_chat_id: dict[str, Conversation],
    memory: RecordingMemoryService,
    agent: RecordingAgent,
    lock_manager: AsyncioConversationLockManager,
) -> InboundWorker:
    return InboundWorker(
        inbound_queue=inbound_queue,
        outbound_queue=outbound_queue,
        conversation_resolver=MappingConversationResolver(conversations_by_chat_id),
        memory_service=cast(ConversationMemoryService, memory),
        agent_boundary=agent,
        lock_manager=lock_manager,
        retry_policy=AgentRetryPolicy(max_attempts=1),
    )


async def test_agent_boundary_gets_clean_request_without_raw_transport_metadata() -> None:
    user_id = uuid4()
    resolved_conversation = conversation(user_id=user_id, chat_id="111")
    inbound_queue = AsyncioInboundQueue()
    outbound_queue = AsyncioOutboundQueue()
    memory = RecordingMemoryService()
    agent = RecordingAgent()
    worker = inbound_worker(
        inbound_queue=inbound_queue,
        outbound_queue=outbound_queue,
        conversations_by_chat_id={"111": resolved_conversation},
        memory=memory,
        agent=agent,
        lock_manager=AsyncioConversationLockManager(),
    )
    event = inbound_event(user_id=user_id, chat_id="111", message_id="1", text="hello")

    await worker.process_event(event)

    assert len(agent.requests) == 1
    request = agent.requests[0]
    assert request.user_id == user_id
    assert request.conversation_id == resolved_conversation.id
    assert request.metadata == {
        "idempotency_key": "telegram:111:1",
        "external_message_id": "1",
        "conversation_id": str(resolved_conversation.id),
    }
    assert "username" not in request.metadata
    assert "first_name" not in request.metadata
    assert "external_chat_id" not in request.metadata
    assert "external_user_id" not in request.metadata
    assert "external_update_id" not in request.metadata
    assert "raw_update" not in request.metadata
    assert "raw_payload" not in request.metadata
    assert request.pydantic_ai is not None
    assert request.pydantic_ai.user_prompt == "hello"
    assert request.pydantic_ai.instructions == "safe summary"

    outbound = await outbound_queue.consume()
    try:
        assert outbound.user_id == user_id
        assert outbound.conversation_id == resolved_conversation.id
        assert outbound.external_chat_id == "111"
        assert outbound.thread_id == "future-thread"
        assert outbound.channel_metadata == {}
        assert outbound.metadata == {"agent_metadata": "safe"}
        assert not hasattr(outbound, "external_user_id")
        assert not hasattr(outbound, "external_update_id")
    finally:
        await outbound_queue.acknowledge()


async def test_in_memory_spine_keeps_delivery_targets_separate_for_parallel_users() -> None:
    first_user_id = uuid4()
    second_user_id = uuid4()
    first_conversation = conversation(user_id=first_user_id, chat_id="111")
    second_conversation = conversation(user_id=second_user_id, chat_id="222")
    inbound_queue = AsyncioInboundQueue()
    outbound_queue = AsyncioOutboundQueue()
    memory = RecordingMemoryService()
    agent = RecordingAgent()
    conversations_by_chat_id = {
        "111": first_conversation,
        "222": second_conversation,
    }
    inbound_lock_manager = AsyncioConversationLockManager()
    inbound_workers = [
        inbound_worker(
            inbound_queue=inbound_queue,
            outbound_queue=outbound_queue,
            conversations_by_chat_id=conversations_by_chat_id,
            memory=memory,
            agent=agent,
            lock_manager=inbound_lock_manager,
        )
        for _ in range(2)
    ]
    delivery_adapter = RecordingDeliveryAdapter()
    registry = InMemoryChannelAdapterRegistry()
    registry.register(delivery_adapter)
    delivery_lock_manager = AsyncioConversationLockManager()
    delivery_workers = [
        DeliveryWorker(
            outbound_queue=outbound_queue,
            channel_adapters=registry,
            lock_manager=delivery_lock_manager,
        )
        for _ in range(2)
    ]

    await inbound_queue.publish(
        inbound_event(user_id=first_user_id, chat_id="111", message_id="1", text="first")
    )
    await inbound_queue.publish(
        inbound_event(user_id=second_user_id, chat_id="222", message_id="2", text="second")
    )

    await asyncio.gather(*(worker.process_next() for worker in inbound_workers))
    await asyncio.wait_for(inbound_queue.join(), timeout=0.1)
    await asyncio.gather(*(worker.process_next() for worker in delivery_workers))
    await asyncio.wait_for(outbound_queue.join(), timeout=0.1)

    delivered_by_chat_id = {event.external_chat_id: event for event in delivery_adapter.events}
    assert set(delivered_by_chat_id) == {"111", "222"}
    assert delivered_by_chat_id["111"].user_id == first_user_id
    assert delivered_by_chat_id["111"].conversation_id == first_conversation.id
    assert delivered_by_chat_id["222"].user_id == second_user_id
    assert delivered_by_chat_id["222"].conversation_id == second_conversation.id
    assert {message.user_id for message in memory.user_messages} == {
        first_user_id,
        second_user_id,
    }
    assert {message.conversation_id for message in memory.assistant_messages} == {
        first_conversation.id,
        second_conversation.id,
    }
    assert {request.user_id for request in agent.requests} == {
        first_user_id,
        second_user_id,
    }
    assert inbound_lock_manager.tracked_lock_count == 0
    assert delivery_lock_manager.tracked_lock_count == 0
