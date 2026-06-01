import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from agent_service.agents import (
    AgentContext,
    AgentContextMessage,
    AgentContextRole,
    AgentResponse,
    PydanticAIRunContext,
)
from agent_service.channels import InboundEvent
from agent_service.conversations import Conversation
from agent_service.memory.compaction import compaction_request_from_snapshot
from agent_service.memory.interfaces import (
    ConversationCompactionPolicyProtocol,
    ConversationCompactionStore,
    ConversationContextSnapshotStore,
    ConversationMemoryService,
    ConversationMemoryStore,
)
from agent_service.memory.models import (
    ConversationCompactionDecision,
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationContextSnapshot,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationSummary,
    PreparedConversationContext,
)
from agent_service.observability.events import TRACE_CONTEXT_METADATA_KEY
from agent_service.memory.pydantic_ai import pydantic_ai_history_from_memory

logger = logging.getLogger(__name__)

DEFAULT_RECENT_MESSAGE_LIMIT = 100
CONTEXT_SNAPSHOT_VERSION = 1


class ConversationMemoryServiceError(Exception):
    """Raised when conversation memory cannot be prepared safely."""


@dataclass(slots=True)
class DefaultConversationMemoryService(ConversationMemoryService):
    memory_store: ConversationMemoryStore
    snapshot_store: ConversationContextSnapshotStore | None = None
    recent_message_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT
    compaction_store: ConversationCompactionStore | None = None

    def __post_init__(self) -> None:
        if self.recent_message_limit < 1:
            raise ValueError("recent_message_limit must be greater than zero")

    async def record_user_message(
        self,
        *,
        conversation: Conversation,
        event: InboundEvent,
    ) -> ConversationMemoryMessage:
        if event.user_id is None:
            raise ConversationMemoryServiceError("Inbound event must be resolved to a user")
        if event.user_id != conversation.user_id:
            raise ConversationMemoryServiceError("Inbound event user does not own conversation")

        message = await self.memory_store.append_message(
            message=ConversationMemoryMessage(
                conversation_id=conversation.id,
                user_id=conversation.user_id,
                role=ConversationMemoryRole.USER,
                text=event.text,
                attachments=list(event.attachments),
                inbound_event_id=event.event_id,
                trace_id=event.trace_id,
                metadata={
                    "channel": event.channel,
                    "idempotency_key": event.idempotency_key,
                    "external_user_id": event.external_user_id,
                    "external_chat_id": event.external_chat_id,
                    "external_message_id": event.external_message_id,
                    "external_update_id": event.external_update_id,
                    "message_type": event.message_type.value,
                    "thread_id": event.thread_id,
                    "reply_to_message_id": event.reply_to_message_id,
                    "channel_metadata": event.channel_metadata,
                    "metadata": {
                        key: value
                        for key, value in event.metadata.items()
                        if key != TRACE_CONTEXT_METADATA_KEY
                    },
                },
                created_at=event.received_at,
            )
        )
        await self._extend_snapshot_if_fresh(message)
        return message

    async def prepare_agent_context(
        self,
        *,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> PreparedConversationContext:
        self._validate_latest_message(conversation, latest_user_message)
        current_sequence = await self.memory_store.current_message_sequence(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
        )
        snapshot = await self._load_fresh_snapshot(
            conversation=conversation,
            current_sequence=current_sequence,
            latest_user_message=latest_user_message,
        )
        snapshot_source = "redis"
        if snapshot is None:
            snapshot = await self._rebuild_snapshot(
                conversation=conversation,
                current_sequence=current_sequence,
            )
            snapshot_source = "postgres"

        history_messages = [
            message for message in snapshot.recent_messages if message.id != latest_user_message.id
        ]
        summary_parts = [snapshot.summary] if snapshot.summary else []
        agent_context = AgentContext(
            system_prompt_parts=summary_parts,
            recent_messages=[
                _agent_context_message(message) for message in snapshot.recent_messages
            ],
            metadata={
                "snapshot_version": snapshot.version,
                "last_seen_sequence": snapshot.last_seen_sequence,
                "last_seen_message_id": _str_or_none(snapshot.last_seen_message_id),
                "current_sequence": current_sequence,
            },
        )
        pydantic_ai = PydanticAIRunContext(
            user_prompt=latest_user_message.text,
            message_history=pydantic_ai_history_from_memory(history_messages),
            conversation_id=str(conversation.id),
            instructions=snapshot.summary,
            metadata={
                "snapshot_version": snapshot.version,
                "last_seen_sequence": snapshot.last_seen_sequence,
                "current_sequence": current_sequence,
            },
        )
        return PreparedConversationContext(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            latest_user_message_id=latest_user_message.id,
            agent_context=agent_context,
            pydantic_ai=pydantic_ai,
            snapshot=snapshot,
            metadata={
                "snapshot_source": snapshot_source,
                "current_sequence": current_sequence,
            },
        )

    async def record_assistant_message(
        self,
        *,
        conversation: Conversation,
        response: AgentResponse,
        trace_id: str | None = None,
        outbound_event_id: UUID | None = None,
    ) -> ConversationMemoryMessage:
        metadata: dict[str, Any] = dict(response.metadata)
        if response.usage is not None:
            metadata["usage"] = response.usage.model_dump(mode="json")
        if response.tool_info is not None:
            metadata["tool_info"] = [
                tool_info.model_dump(mode="json") for tool_info in response.tool_info
            ]

        message = await self.memory_store.append_message(
            message=ConversationMemoryMessage(
                conversation_id=conversation.id,
                user_id=conversation.user_id,
                role=ConversationMemoryRole.ASSISTANT,
                text=response.text,
                outbound_event_id=outbound_event_id,
                trace_id=trace_id or response.trace_id,
                token_count=response.usage.output_tokens if response.usage is not None else None,
                metadata=metadata,
            )
        )
        await self._extend_snapshot_if_fresh(message)
        return message

    async def prepare_compaction_request(
        self,
        *,
        conversation: Conversation,
        compact_through_sequence: int | None = None,
    ) -> ConversationCompactionRequest:
        current_sequence = await self.memory_store.current_message_sequence(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
        )
        snapshot = await self._load_fresh_snapshot_for_sequence(
            conversation=conversation,
            current_sequence=current_sequence,
        )
        if snapshot is None:
            snapshot = await self._rebuild_snapshot(
                conversation=conversation,
                current_sequence=current_sequence,
            )
        return compaction_request_from_snapshot(
            snapshot,
            compact_through_sequence=compact_through_sequence,
        )

    async def evaluate_compaction(
        self,
        *,
        conversation: Conversation,
        policy: ConversationCompactionPolicyProtocol,
    ) -> ConversationCompactionDecision:
        current_sequence = await self.memory_store.current_message_sequence(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
        )
        snapshot = await self._load_fresh_snapshot_for_sequence(
            conversation=conversation,
            current_sequence=current_sequence,
        )
        if snapshot is None:
            snapshot = await self._rebuild_snapshot(
                conversation=conversation,
                current_sequence=current_sequence,
            )
        return policy.decide(snapshot=snapshot)

    async def record_compaction_result(
        self,
        *,
        conversation: Conversation,
        request: ConversationCompactionRequest,
        result: ConversationCompactionResult,
        trace_id: str | None = None,
    ) -> ConversationSummary:
        if self.compaction_store is None:
            raise ConversationMemoryServiceError("Conversation compaction store is not configured")
        _validate_compaction_ownership(
            conversation=conversation,
            request=request,
            result=result,
        )
        if result.last_compacted_sequence is None:
            raise ConversationMemoryServiceError(
                "Compaction result must include compacted sequence"
            )
        existing = await self.compaction_store.get_completed_summary_by_sequence(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            to_sequence=result.last_compacted_sequence,
        )
        if existing is not None:
            await self._apply_summary_to_snapshot(existing)
            return existing
        summary = _summary_from_compaction_result(
            conversation=conversation,
            request=request,
            result=result,
            trace_id=trace_id,
        )
        stored = await self.compaction_store.append_summary(summary=summary)
        await self._apply_summary_to_snapshot(stored)
        return stored

    async def _load_fresh_snapshot(
        self,
        *,
        conversation: Conversation,
        current_sequence: int,
        latest_user_message: ConversationMemoryMessage,
    ) -> ConversationContextSnapshot | None:
        snapshot = await self._get_snapshot_or_none(conversation.id)
        if snapshot is None:
            return None
        if not self._snapshot_is_fresh(
            snapshot,
            conversation=conversation,
            current_sequence=current_sequence,
            latest_user_message=latest_user_message,
        ):
            await self._delete_snapshot(conversation.id)
            return None
        return snapshot

    async def _rebuild_snapshot(
        self,
        *,
        conversation: Conversation,
        current_sequence: int,
    ) -> ConversationContextSnapshot:
        latest_summary = await self._get_latest_completed_summary(conversation)
        if latest_summary is None:
            messages = await self.memory_store.list_recent_messages(
                conversation_id=conversation.id,
                limit=self.recent_message_limit,
            )
        else:
            if latest_summary.to_sequence > current_sequence:
                raise ConversationMemoryServiceError(
                    "Latest conversation summary is ahead of message sequence"
                )
            messages = await self.memory_store.list_messages_after_sequence(
                conversation_id=conversation.id,
                user_id=conversation.user_id,
                after_sequence=latest_summary.to_sequence,
                limit=self.recent_message_limit,
            )
        snapshot = _snapshot_from_messages(
            conversation=conversation,
            messages=messages,
            current_sequence=current_sequence,
            summary=latest_summary,
        )
        await self._save_snapshot(snapshot)
        return snapshot

    async def _extend_snapshot_if_fresh(self, message: ConversationMemoryMessage) -> None:
        if message.sequence is None:
            raise ConversationMemoryServiceError("Stored memory message must have a sequence")
        snapshot = await self._get_snapshot_or_none(message.conversation_id)
        if snapshot is None:
            return
        previous_sequence = message.sequence - 1
        if (
            snapshot.user_id != message.user_id
            or snapshot.last_seen_sequence is None
            or snapshot.last_seen_sequence != previous_sequence
        ):
            await self._delete_snapshot(message.conversation_id)
            return

        recent_messages = [*snapshot.recent_messages, message][-self.recent_message_limit :]
        updated = snapshot.model_copy(
            update={
                "recent_messages": recent_messages,
                "last_seen_message_id": message.id,
                "last_seen_sequence": message.sequence,
                "token_count": _snapshot_summary_token_count(snapshot)
                + _token_count(recent_messages),
                "updated_at": _utc_now(),
            }
        )
        await self._save_snapshot(updated)

    def _snapshot_is_fresh(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        conversation: Conversation,
        current_sequence: int,
        latest_user_message: ConversationMemoryMessage,
    ) -> bool:
        if snapshot.conversation_id != conversation.id:
            return False
        if snapshot.user_id != conversation.user_id:
            return False
        if snapshot.version != CONTEXT_SNAPSHOT_VERSION:
            return False
        if snapshot.last_seen_sequence != current_sequence:
            return False
        if latest_user_message.sequence is None:
            return False
        if latest_user_message.sequence > current_sequence:
            return False
        return any(message.id == latest_user_message.id for message in snapshot.recent_messages)

    async def _load_fresh_snapshot_for_sequence(
        self,
        *,
        conversation: Conversation,
        current_sequence: int,
    ) -> ConversationContextSnapshot | None:
        snapshot = await self._get_snapshot_or_none(conversation.id)
        if snapshot is None:
            return None
        if not self._snapshot_matches_sequence(
            snapshot,
            conversation=conversation,
            current_sequence=current_sequence,
        ):
            await self._delete_snapshot(conversation.id)
            return None
        return snapshot

    def _snapshot_matches_sequence(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        conversation: Conversation,
        current_sequence: int,
    ) -> bool:
        if snapshot.conversation_id != conversation.id:
            return False
        if snapshot.user_id != conversation.user_id:
            return False
        if snapshot.version != CONTEXT_SNAPSHOT_VERSION:
            return False
        return snapshot.last_seen_sequence == current_sequence

    async def _apply_summary_to_snapshot(self, summary: ConversationSummary) -> None:
        snapshot = await self._get_snapshot_or_none(summary.conversation_id)
        if snapshot is None:
            return
        if snapshot.user_id != summary.user_id:
            await self._delete_snapshot(summary.conversation_id)
            return

        recent_messages = [
            message
            for message in snapshot.recent_messages
            if message.sequence is not None and message.sequence > summary.to_sequence
        ]
        updated = snapshot.model_copy(
            update={
                "summary": summary.summary,
                "recent_messages": recent_messages,
                "last_compacted_message_id": summary.last_compacted_message_id,
                "last_compacted_sequence": summary.to_sequence,
                "token_count": summary.output_token_count + _token_count(recent_messages),
                "metadata": {
                    **snapshot.metadata,
                    "latest_summary_id": str(summary.id),
                    "latest_summary_created_at": summary.created_at.isoformat(),
                    "latest_summary_token_count": summary.output_token_count,
                },
                "updated_at": _utc_now(),
            }
        )
        await self._save_snapshot(updated)

    async def _get_latest_completed_summary(
        self,
        conversation: Conversation,
    ) -> ConversationSummary | None:
        if self.compaction_store is None:
            return None
        summary = await self.compaction_store.get_latest_completed_summary(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
        )
        if summary is None:
            return None
        if summary.conversation_id != conversation.id or summary.user_id != conversation.user_id:
            raise ConversationMemoryServiceError(
                "Latest conversation summary belongs to another conversation or user"
            )
        return summary

    def _validate_latest_message(
        self,
        conversation: Conversation,
        latest_user_message: ConversationMemoryMessage,
    ) -> None:
        if latest_user_message.conversation_id != conversation.id:
            raise ConversationMemoryServiceError("Latest message belongs to another conversation")
        if latest_user_message.user_id != conversation.user_id:
            raise ConversationMemoryServiceError("Latest message belongs to another user")
        if latest_user_message.role is not ConversationMemoryRole.USER:
            raise ConversationMemoryServiceError("Latest message must be a user message")
        if latest_user_message.sequence is None:
            raise ConversationMemoryServiceError("Latest message must have a sequence")

    async def _get_snapshot_or_none(
        self,
        conversation_id: UUID,
    ) -> ConversationContextSnapshot | None:
        if self.snapshot_store is None:
            return None
        try:
            return await self.snapshot_store.get_snapshot(conversation_id=conversation_id)
        except Exception:
            logger.warning(
                "Context snapshot is invalid and will be rebuilt",
                exc_info=True,
                extra={
                    "event": "conversation_context_snapshot_invalid",
                    "conversation_id": str(conversation_id),
                },
            )
            await self._delete_snapshot(conversation_id)
            return None

    async def _save_snapshot(self, snapshot: ConversationContextSnapshot) -> None:
        if self.snapshot_store is None:
            return
        await self.snapshot_store.save_snapshot(snapshot=snapshot)

    async def _delete_snapshot(self, conversation_id: UUID) -> None:
        if self.snapshot_store is None:
            return
        await self.snapshot_store.delete_snapshot(conversation_id=conversation_id)


def _snapshot_from_messages(
    *,
    conversation: Conversation,
    messages: list[ConversationMemoryMessage],
    current_sequence: int,
    summary: ConversationSummary | None = None,
) -> ConversationContextSnapshot:
    last_message = messages[-1] if messages else None
    return ConversationContextSnapshot(
        conversation_id=conversation.id,
        user_id=conversation.user_id,
        summary=summary.summary if summary is not None else None,
        recent_messages=messages,
        last_compacted_message_id=(
            summary.last_compacted_message_id if summary is not None else None
        ),
        last_seen_message_id=(
            last_message.id
            if last_message is not None
            else summary.last_compacted_message_id
            if summary is not None
            else None
        ),
        last_compacted_sequence=summary.to_sequence if summary is not None else None,
        last_seen_sequence=current_sequence,
        version=CONTEXT_SNAPSHOT_VERSION,
        token_count=(summary.output_token_count if summary is not None else 0)
        + _token_count(messages),
        metadata=(
            {
                "latest_summary_id": str(summary.id),
                "latest_summary_created_at": summary.created_at.isoformat(),
                "latest_summary_token_count": summary.output_token_count,
            }
            if summary is not None
            else {}
        ),
    )


def _summary_from_compaction_result(
    *,
    conversation: Conversation,
    request: ConversationCompactionRequest,
    result: ConversationCompactionResult,
    trace_id: str | None,
) -> ConversationSummary:
    _validate_compaction_ownership(
        conversation=conversation,
        request=request,
        result=result,
    )
    if result.last_compacted_sequence is None:
        raise ConversationMemoryServiceError("Compaction result must include compacted sequence")
    compacted_messages = [
        message
        for message in request.messages
        if message.sequence is not None and message.sequence <= result.last_compacted_sequence
    ]
    if not compacted_messages:
        raise ConversationMemoryServiceError("Compaction result did not compact any messages")
    last_message = compacted_messages[-1]
    if result.last_compacted_message_id != last_message.id:
        raise ConversationMemoryServiceError("Compaction result message id does not match sequence")

    compacted_message_ids = result.compacted_message_ids or [
        message.id for message in compacted_messages
    ]
    model = result.metadata.get("model")
    return ConversationSummary(
        conversation_id=conversation.id,
        user_id=conversation.user_id,
        from_sequence=compacted_messages[0].sequence or 1,
        to_sequence=result.last_compacted_sequence,
        previous_summary=request.previous_summary,
        summary=result.summary,
        compacted_message_ids=compacted_message_ids,
        last_compacted_message_id=result.last_compacted_message_id,
        input_token_count=_token_count(compacted_messages),
        output_token_count=result.token_count,
        model=model if isinstance(model, str) else None,
        trace_id=trace_id,
        metadata={
            "request": request.metadata,
            "result": result.metadata,
        },
        created_at=result.created_at,
    )


def _validate_compaction_ownership(
    *,
    conversation: Conversation,
    request: ConversationCompactionRequest,
    result: ConversationCompactionResult,
) -> None:
    if request.conversation_id != conversation.id or result.conversation_id != conversation.id:
        raise ConversationMemoryServiceError("Compaction state belongs to another conversation")
    if request.user_id != conversation.user_id or result.user_id != conversation.user_id:
        raise ConversationMemoryServiceError("Compaction state belongs to another user")


def _agent_context_message(message: ConversationMemoryMessage) -> AgentContextMessage:
    return AgentContextMessage(
        role=_agent_context_role(message.role),
        text=_message_text(message),
        message_id=message.id,
        metadata={
            "sequence": message.sequence,
            "tool_name": message.tool_name,
            "tool_call_id": message.tool_call_id,
            "trace_id": message.trace_id,
            **message.metadata,
        },
        created_at=message.created_at,
    )


def _agent_context_role(role: ConversationMemoryRole) -> AgentContextRole:
    if role is ConversationMemoryRole.USER:
        return AgentContextRole.USER
    if role is ConversationMemoryRole.ASSISTANT:
        return AgentContextRole.ASSISTANT
    if role is ConversationMemoryRole.TOOL_CALL:
        return AgentContextRole.TOOL_CALL
    return AgentContextRole.TOOL_RESULT


def _message_text(message: ConversationMemoryMessage) -> str:
    if message.text:
        return message.text
    if message.role is ConversationMemoryRole.TOOL_CALL:
        return f"Tool call: {message.tool_name or 'unknown'}"
    if message.role is ConversationMemoryRole.TOOL_RESULT:
        return f"Tool result: {message.tool_name or 'unknown'}"
    if message.attachments:
        return "[attachments]"
    return "[empty message]"


def _token_count(messages: list[ConversationMemoryMessage]) -> int:
    return sum(message.token_count or 0 for message in messages)


def _snapshot_summary_token_count(snapshot: ConversationContextSnapshot) -> int:
    value = snapshot.metadata.get("latest_summary_token_count")
    if isinstance(value, int):
        return value
    return max(snapshot.token_count - _token_count(snapshot.recent_messages), 0)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _str_or_none(value: UUID | None) -> str | None:
    if value is None:
        return None
    return str(value)
