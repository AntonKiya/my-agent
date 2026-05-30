import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from agent_service.agents import AgentContext, AgentRequest, AgentResponse, PydanticAIRunContext
from agent_service.channels import InboundEvent
from agent_service.conversations import AsyncioConversationLockManager, Conversation
from agent_service.inbound import AgentRetryPolicy, InboundWorker
from agent_service.memory import (
    ConversationCompactionDecision,
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationSummary,
    PreparedConversationContext,
)
from agent_service.messaging import AsyncioInboundQueue, AsyncioOutboundQueue


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
            agent_context=AgentContext(),
            pydantic_ai=PydanticAIRunContext(
                user_prompt=latest_user_message.text,
                conversation_id=str(conversation.id),
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
class IdentityAgent:
    requests: list[AgentRequest] = field(default_factory=list)

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            text=f"user={request.user_id};conversation={request.conversation_id};text={request.text}",
            trace_id=request.trace_id,
        )


def conversation(*, user_id: UUID, chat_id: str) -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=user_id,
        channel="telegram",
        conversation_key=f"telegram:private:{chat_id}",
        external_chat_id=chat_id,
    )


def inbound_event(*, user_id: UUID, chat_id: str, text: str) -> InboundEvent:
    return InboundEvent(
        channel="telegram",
        external_user_id=f"external-{user_id}",
        external_chat_id=chat_id,
        external_message_id=f"message-{text}",
        idempotency_key=f"telegram:{chat_id}:message-{text}",
        user_id=user_id,
        text=text,
        trace_id=f"trace-{text}",
    )


async def test_processing_spine_keeps_users_and_conversations_separate() -> None:
    first_user_id = uuid4()
    second_user_id = uuid4()
    first_conversation = conversation(user_id=first_user_id, chat_id="111")
    second_conversation = conversation(user_id=second_user_id, chat_id="222")
    inbound_queue = AsyncioInboundQueue()
    outbound_queue = AsyncioOutboundQueue()
    resolver = MappingConversationResolver(
        conversations_by_chat_id={
            "111": first_conversation,
            "222": second_conversation,
        }
    )
    memory = RecordingMemoryService()
    agent = IdentityAgent()
    lock_manager = AsyncioConversationLockManager()

    workers = [
        InboundWorker(
            inbound_queue=inbound_queue,
            outbound_queue=outbound_queue,
            conversation_resolver=resolver,
            memory_service=memory,
            agent_boundary=agent,
            lock_manager=lock_manager,
            retry_policy=AgentRetryPolicy(max_attempts=1),
        )
        for _ in range(2)
    ]

    await inbound_queue.publish(inbound_event(user_id=first_user_id, chat_id="111", text="first"))
    await inbound_queue.publish(inbound_event(user_id=second_user_id, chat_id="222", text="second"))

    await asyncio.gather(*(worker.process_next() for worker in workers))

    first_outbound = await outbound_queue.consume()
    second_outbound = await outbound_queue.consume()
    outbounds_by_user_id = {
        first_outbound.user_id: first_outbound,
        second_outbound.user_id: second_outbound,
    }

    assert outbounds_by_user_id[first_user_id].conversation_id == first_conversation.id
    assert outbounds_by_user_id[second_user_id].conversation_id == second_conversation.id
    assert str(first_user_id) in (outbounds_by_user_id[first_user_id].text or "")
    assert str(second_user_id) in (outbounds_by_user_id[second_user_id].text or "")
    assert {message.user_id for message in memory.user_messages} == {first_user_id, second_user_id}
    assert {message.conversation_id for message in memory.assistant_messages} == {
        first_conversation.id,
        second_conversation.id,
    }
    assert {request.user_id for request in agent.requests} == {first_user_id, second_user_id}
    assert {request.conversation_id for request in agent.requests} == {
        first_conversation.id,
        second_conversation.id,
    }
    assert lock_manager.tracked_lock_count == 0
