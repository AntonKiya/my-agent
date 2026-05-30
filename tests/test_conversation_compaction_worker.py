import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from agent_service.conversations import AsyncioConversationLockManager, Conversation
from agent_service.memory import (
    ConversationCompactionDecision,
    ConversationCompactionJob,
    ConversationCompactionPolicy,
    ConversationCompactionRequest,
    ConversationCompactionResult,
    ConversationCompactionWorker,
    ConversationMemoryMessage,
    ConversationMemoryRole,
    ConversationMemoryService,
    ConversationSummary,
)
from agent_service.messaging import AsyncioCompactionQueue
from agent_service.observability.tracing import get_trace_id, reset_trace_id, set_trace_id


@dataclass(slots=True)
class RecordingMemoryService:
    requests: list[ConversationCompactionRequest] = field(default_factory=list)
    results: list[ConversationCompactionResult] = field(default_factory=list)
    compact_through_sequences: list[int | None] = field(default_factory=list)

    async def prepare_compaction_request(
        self,
        *,
        conversation: Conversation,
        compact_through_sequence: int | None = None,
    ) -> ConversationCompactionRequest:
        self.compact_through_sequences.append(compact_through_sequence)
        message = ConversationMemoryMessage(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            sequence=compact_through_sequence or 1,
            role=ConversationMemoryRole.USER,
            text="old context",
            token_count=12,
        )
        request = ConversationCompactionRequest(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            messages=[message],
            metadata={"source": "test"},
        )
        self.requests.append(request)
        return request

    async def record_compaction_result(
        self,
        *,
        conversation: Conversation,
        request: ConversationCompactionRequest,
        result: ConversationCompactionResult,
        trace_id: str | None = None,
    ) -> ConversationSummary:
        self.results.append(result)
        return ConversationSummary(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            from_sequence=request.messages[0].sequence or 1,
            to_sequence=result.last_compacted_sequence or 1,
            summary=result.summary,
            compacted_message_ids=result.compacted_message_ids,
            last_compacted_message_id=result.last_compacted_message_id,
            output_token_count=result.token_count,
            trace_id=trace_id,
            created_at=result.created_at,
        )

    async def evaluate_compaction(
        self,
        *,
        conversation: Conversation,
        policy: ConversationCompactionPolicy,
    ) -> ConversationCompactionDecision:
        raise NotImplementedError

    async def record_user_message(self, **kwargs: object) -> ConversationMemoryMessage:
        raise NotImplementedError

    async def prepare_agent_context(self, **kwargs: object) -> object:
        raise NotImplementedError

    async def record_assistant_message(self, **kwargs: object) -> ConversationMemoryMessage:
        raise NotImplementedError


@dataclass(slots=True)
class RecordingCompactor:
    requests: list[ConversationCompactionRequest] = field(default_factory=list)
    trace_ids: list[str | None] = field(default_factory=list)

    async def compact(
        self,
        *,
        request: ConversationCompactionRequest,
    ) -> ConversationCompactionResult:
        self.requests.append(request)
        self.trace_ids.append(get_trace_id())
        message = request.messages[-1]
        if message.sequence is None:
            raise ValueError("test request requires sequence")
        return ConversationCompactionResult(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            summary="compressed old context",
            compacted_message_ids=[message.id],
            last_compacted_message_id=message.id,
            last_compacted_sequence=message.sequence,
            token_count=5,
            metadata={"model": "test"},
            created_at=datetime(2026, 5, 30, 13, 0, tzinfo=UTC),
        )


def conversation() -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=uuid4(),
        channel="telegram",
        conversation_key="telegram:private:123",
        external_chat_id="123",
    )


def compaction_job(
    *,
    resolved_conversation: Conversation,
    compact_through_sequence: int = 7,
) -> ConversationCompactionJob:
    return ConversationCompactionJob(
        conversation=resolved_conversation,
        compact_through_sequence=compact_through_sequence,
        reason="trigger_reached",
        trace_id="trace-1",
    )


async def test_compaction_worker_compacts_one_job_under_conversation_lock() -> None:
    resolved_conversation = conversation()
    queue = AsyncioCompactionQueue()
    memory = RecordingMemoryService()
    compactor = RecordingCompactor()
    lock_manager = AsyncioConversationLockManager()
    worker = ConversationCompactionWorker(
        compaction_queue=queue,
        memory_service=cast(ConversationMemoryService, memory),
        compactor=compactor,
        lock_manager=lock_manager,
    )
    job = compaction_job(resolved_conversation=resolved_conversation)

    await queue.publish(job)
    await worker.process_next()

    assert memory.compact_through_sequences == [7]
    assert len(compactor.requests) == 1
    assert len(memory.results) == 1
    assert memory.results[0].summary == "compressed old context"
    assert compactor.trace_ids == ["trace-1"]
    assert lock_manager.tracked_lock_count == 0


async def test_compaction_worker_skips_empty_requests() -> None:
    resolved_conversation = conversation()
    queue = AsyncioCompactionQueue()

    class EmptyMemoryService(RecordingMemoryService):
        async def prepare_compaction_request(
            self,
            *,
            conversation: Conversation,
            compact_through_sequence: int | None = None,
        ) -> ConversationCompactionRequest:
            return ConversationCompactionRequest(
                conversation_id=conversation.id,
                user_id=conversation.user_id,
            )

    memory = EmptyMemoryService()
    compactor = RecordingCompactor()
    worker = ConversationCompactionWorker(
        compaction_queue=queue,
        memory_service=cast(ConversationMemoryService, memory),
        compactor=compactor,
        lock_manager=AsyncioConversationLockManager(),
    )

    await queue.publish(compaction_job(resolved_conversation=resolved_conversation))
    await worker.process_next()

    assert compactor.requests == []
    assert memory.results == []


async def test_compaction_worker_restores_trace_context_and_logs_safe_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agent_service.memory.worker")
    resolved_conversation = conversation()
    queue = AsyncioCompactionQueue()
    memory = RecordingMemoryService()
    compactor = RecordingCompactor()
    worker = ConversationCompactionWorker(
        compaction_queue=queue,
        memory_service=cast(ConversationMemoryService, memory),
        compactor=compactor,
        lock_manager=AsyncioConversationLockManager(),
    )
    outer_token = set_trace_id("outer-trace")
    try:
        await queue.publish(compaction_job(resolved_conversation=resolved_conversation))
        await worker.process_next()

        assert get_trace_id() == "outer-trace"
    finally:
        reset_trace_id(outer_token)

    completed = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "conversation_compaction_completed"
    ]
    assert len(completed) == 1
    assert completed[0].__dict__["conversation_id"] == str(resolved_conversation.id)
    assert completed[0].__dict__["user_id"] == str(resolved_conversation.user_id)
    assert not hasattr(completed[0], "summary")
    assert not hasattr(completed[0], "prompt")
    assert not hasattr(completed[0], "text")
