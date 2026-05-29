from typing import Protocol, runtime_checkable
from uuid import UUID

from agent_service.agents import AgentResponse
from agent_service.channels import InboundEvent
from agent_service.conversations import Conversation
from agent_service.memory.models import ConversationMemoryMessage, PreparedConversationContext


@runtime_checkable
class ConversationMemoryService(Protocol):
    async def record_user_message(
        self,
        *,
        conversation: Conversation,
        event: InboundEvent,
    ) -> ConversationMemoryMessage:
        """Persist the inbound user message before the agent run."""
        ...

    async def prepare_agent_context(
        self,
        *,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> PreparedConversationContext:
        """Build the hot working context used by the agent boundary."""
        ...

    async def record_assistant_message(
        self,
        *,
        conversation: Conversation,
        response: AgentResponse,
        trace_id: str | None = None,
        outbound_event_id: UUID | None = None,
    ) -> ConversationMemoryMessage:
        """Persist the assistant message after a successful agent response."""
        ...
