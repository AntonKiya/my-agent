from dataclasses import dataclass

from agent_service.memory.interfaces import ConversationCompactor
from agent_service.memory.models import (
    ConversationCompactionDecision,
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


@dataclass(frozen=True, slots=True)
class ConversationCompactionPolicy:
    enabled: bool = False
    context_window_tokens: int = 196_600
    reserved_output_tokens: int = 16_384
    trigger_fraction: float = 0.80
    recent_tail_fraction: float = 0.30

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be greater than zero")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must be greater than or equal to zero")
        if self.reserved_output_tokens >= self.context_window_tokens:
            raise ValueError("reserved_output_tokens must be less than context_window_tokens")
        if not 0 < self.trigger_fraction < 1:
            raise ValueError("trigger_fraction must be between zero and one")
        if not 0 < self.recent_tail_fraction < 1:
            raise ValueError("recent_tail_fraction must be between zero and one")
        if self.recent_tail_fraction >= self.trigger_fraction:
            raise ValueError("recent_tail_fraction must be less than trigger_fraction")

    @property
    def usable_input_budget_tokens(self) -> int:
        return self.context_window_tokens - self.reserved_output_tokens

    @property
    def trigger_tokens(self) -> int:
        return int(self.usable_input_budget_tokens * self.trigger_fraction)

    @property
    def recent_tail_budget_tokens(self) -> int:
        return int(self.usable_input_budget_tokens * self.recent_tail_fraction)

    def decide(
        self,
        *,
        snapshot: ConversationContextSnapshot,
        additional_input_tokens: int = 0,
    ) -> ConversationCompactionDecision:
        if additional_input_tokens < 0:
            raise ValueError("additional_input_tokens must be greater than or equal to zero")

        estimated_input_tokens = snapshot.token_count + additional_input_tokens
        base = {
            "estimated_input_tokens": estimated_input_tokens,
            "usable_input_budget_tokens": self.usable_input_budget_tokens,
            "trigger_tokens": self.trigger_tokens,
            "recent_tail_budget_tokens": self.recent_tail_budget_tokens,
        }
        if not self.enabled:
            return ConversationCompactionDecision(
                should_compact=False,
                reason="disabled",
                **base,
            )
        if estimated_input_tokens < self.trigger_tokens:
            return ConversationCompactionDecision(
                should_compact=False,
                reason="below_trigger",
                **base,
            )

        compactable_messages = compactable_messages_from_snapshot(snapshot)
        if len(compactable_messages) < 2:
            return ConversationCompactionDecision(
                should_compact=False,
                reason="not_enough_compactable_messages",
                compactable_token_count=_messages_token_count(compactable_messages),
                **base,
            )

        retained_tail = _retained_tail_by_token_budget(
            compactable_messages,
            token_budget=self.recent_tail_budget_tokens,
        )
        compact_prefix = compactable_messages[: len(compactable_messages) - len(retained_tail)]
        if not compact_prefix:
            return ConversationCompactionDecision(
                should_compact=False,
                reason="nothing_before_recent_tail",
                compactable_token_count=_messages_token_count(compactable_messages),
                retained_tail_token_count=_messages_token_count(retained_tail),
                **base,
            )

        last_compacted = compact_prefix[-1]
        first_retained = retained_tail[0] if retained_tail else None
        return ConversationCompactionDecision(
            should_compact=True,
            reason="trigger_reached",
            compact_through_sequence=last_compacted.sequence,
            keep_from_sequence=first_retained.sequence if first_retained is not None else None,
            compactable_token_count=_messages_token_count(compact_prefix),
            retained_tail_token_count=_messages_token_count(retained_tail),
            **base,
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
    *,
    compact_through_sequence: int | None = None,
) -> ConversationCompactionRequest:
    messages = compactable_messages_from_snapshot(snapshot)
    if compact_through_sequence is not None:
        messages = [
            message
            for message in messages
            if message.sequence is not None and message.sequence <= compact_through_sequence
        ]
    return ConversationCompactionRequest(
        conversation_id=snapshot.conversation_id,
        user_id=snapshot.user_id,
        previous_summary=snapshot.summary,
        messages=messages,
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


def _retained_tail_by_token_budget(
    messages: list[ConversationMemoryMessage],
    *,
    token_budget: int,
) -> list[ConversationMemoryMessage]:
    retained: list[ConversationMemoryMessage] = []
    total = 0
    for message in reversed(messages):
        token_count = message.token_count or 0
        if retained and total + token_count > token_budget:
            break
        retained.append(message)
        total += token_count
    retained.reverse()
    return retained


def _messages_token_count(messages: list[ConversationMemoryMessage]) -> int:
    return sum(message.token_count or 0 for message in messages)
