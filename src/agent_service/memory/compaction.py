from dataclasses import dataclass

from agent_service.memory.interfaces import ConversationCompactor
from agent_service.memory.models import (
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationContextSnapshot,
    ConversationMemoryMessage,
    ConversationMemoryRole,
)

COMPACTABLE_ROLES = frozenset(
    {
        ConversationMemoryRole.USER,
        ConversationMemoryRole.ASSISTANT,
    }
)


@dataclass(slots=True)
class NoopConversationCompactor(ConversationCompactor):
    """Compaction boundary placeholder that never changes conversation state."""

    async def compact(
        self,
        *,
        request: ConversationCompactionRequest,
    ) -> ConversationCompactionResult:
        return ConversationCompactionResult(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            summary=request.previous_summary,
            last_compacted_sequence=request.last_compacted_sequence,
            token_count=0,
            metadata={"noop": True, **request.metadata},
        )


def compactable_messages_from_snapshot(
    snapshot: ConversationContextSnapshot,
) -> list[ConversationMemoryMessage]:
    last_compacted_sequence = snapshot.last_compacted_sequence or 0
    return [
        message
        for message in snapshot.recent_messages
        if message.role in COMPACTABLE_ROLES
        and message.sequence is not None
        and message.sequence > last_compacted_sequence
    ]


def compaction_request_from_snapshot(
    snapshot: ConversationContextSnapshot,
) -> ConversationCompactionRequest:
    return ConversationCompactionRequest(
        conversation_id=snapshot.conversation_id,
        user_id=snapshot.user_id,
        previous_summary=snapshot.summary,
        messages=compactable_messages_from_snapshot(snapshot),
        last_compacted_sequence=snapshot.last_compacted_sequence,
        metadata={
            "snapshot_version": snapshot.version,
            "last_seen_sequence": snapshot.last_seen_sequence,
            "last_seen_message_id": (
                str(snapshot.last_seen_message_id)
                if snapshot.last_seen_message_id is not None
                else None
            ),
        },
    )
