from typing import Protocol, runtime_checkable
from uuid import UUID

from agent_service.agents import AgentResponse
from agent_service.channels import InboundEvent
from agent_service.conversations import Conversation
from agent_service.memory.models import (
    ConversationCompactionDecision,
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationContextSnapshot,
    ConversationMemoryMessage,
    ConversationSummary,
    PreparedConversationContext,
)


@runtime_checkable
class ConversationMemoryStore(Protocol):
    async def append_message(
        self,
        *,
        message: ConversationMemoryMessage,
    ) -> ConversationMemoryMessage:
        """Append one message and assign the next per-conversation sequence."""
        ...

    async def list_recent_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int,
    ) -> list[ConversationMemoryMessage]:
        """Load recent messages for one conversation in ascending sequence order."""
        ...

    async def list_messages_after_sequence(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        after_sequence: int,
        limit: int,
    ) -> list[ConversationMemoryMessage]:
        """Load recent messages after a compacted sequence in ascending sequence order."""
        ...

    async def current_message_sequence(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> int:
        """Return the latest persisted per-conversation message sequence."""
        ...


@runtime_checkable
class ConversationContextSnapshotStore(Protocol):
    async def get_snapshot(
        self,
        *,
        conversation_id: UUID,
    ) -> ConversationContextSnapshot | None:
        """Load a hot working context snapshot by conversation id."""
        ...

    async def save_snapshot(
        self,
        *,
        snapshot: ConversationContextSnapshot,
    ) -> None:
        """Persist a hot working context snapshot with backend-specific TTL."""
        ...

    async def delete_snapshot(
        self,
        *,
        conversation_id: UUID,
    ) -> None:
        """Delete a hot working context snapshot when it must be rebuilt."""
        ...


@runtime_checkable
class ConversationCompactionStore(Protocol):
    async def append_summary(
        self,
        *,
        summary: ConversationSummary,
    ) -> ConversationSummary:
        """Persist one immutable conversation summary or compaction failure state."""
        ...

    async def get_latest_completed_summary(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> ConversationSummary | None:
        """Load the latest completed summary for one conversation and user."""
        ...

    async def get_completed_summary_by_sequence(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        to_sequence: int,
    ) -> ConversationSummary | None:
        """Load one completed summary by its idempotency sequence boundary."""
        ...


@runtime_checkable
class ConversationCompactor(Protocol):
    async def compact(
        self,
        *,
        request: ConversationCompactionRequest,
    ) -> ConversationCompactionResult:
        """Compact old user/assistant conversation messages into a summary."""
        ...


@runtime_checkable
class ConversationCompactionPolicyProtocol(Protocol):
    def decide(
        self,
        *,
        snapshot: ConversationContextSnapshot,
        additional_input_tokens: int = 0,
    ) -> ConversationCompactionDecision:
        """Decide whether a snapshot should be compacted."""
        ...


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

    async def prepare_compaction_request(
        self,
        *,
        conversation: Conversation,
        compact_through_sequence: int | None = None,
    ) -> ConversationCompactionRequest:
        """Build a safe request for an external compactor from current memory state."""
        ...

    async def evaluate_compaction(
        self,
        *,
        conversation: Conversation,
        policy: ConversationCompactionPolicyProtocol,
    ) -> ConversationCompactionDecision:
        """Evaluate whether current memory state should be compacted."""
        ...

    async def record_compaction_result(
        self,
        *,
        conversation: Conversation,
        request: ConversationCompactionRequest,
        result: ConversationCompactionResult,
        trace_id: str | None = None,
    ) -> ConversationSummary:
        """Persist compaction output and apply it to hot memory state."""
        ...
