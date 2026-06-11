import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent_service.conversations import ConversationLockManager
from agent_service.memory.interfaces import ConversationCompactor, ConversationMemoryService
from agent_service.memory.models import ConversationCompactionJob, ConversationCompactionResult
from agent_service.memory.tokens import estimate_messages_tokens, usage_token_count
from agent_service.messaging.interfaces import CompactionQueue
from agent_service.observability.events import (
    attached_trace_context,
    business_span,
    elapsed_ms,
    log_event,
    log_exception,
    start_timer,
)
from agent_service.observability.tracing import reset_trace_id, set_trace_id

logger = logging.getLogger(__name__)

SleepCallable = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class ConversationCompactionWorker:
    compaction_queue: CompactionQueue
    memory_service: ConversationMemoryService
    compactor: ConversationCompactor
    lock_manager: ConversationLockManager
    error_backoff_seconds: float = 0.1
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.error_backoff_seconds < 0:
            raise ValueError(
                "Compaction worker error backoff must be greater than or equal to zero"
            )

    async def run_forever(self) -> None:
        while True:
            try:
                await self.process_next()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Conversation compaction worker iteration failed",
                    extra={"event": "conversation_compaction_worker_iteration_failed"},
                )
                if self.error_backoff_seconds > 0:
                    await self.sleep(self.error_backoff_seconds)

    async def process_next(self) -> None:
        job = await self.compaction_queue.consume()
        with attached_trace_context(job.metadata):
            log_event(
                logger,
                logging.INFO,
                "Conversation compaction job dequeued",
                event="conversation_compaction_dequeued",
                queue_name="compaction",
                conversation_id=str(job.conversation.id),
                user_id=str(job.conversation.user_id),
                compaction_job_id=str(job.event_id),
                compact_through_sequence=job.compact_through_sequence,
                reason=job.reason,
                queue_size=self.compaction_queue.stats.size,
                queue_maxsize=self.compaction_queue.stats.maxsize,
            )
        try:
            await self.process_job(job)
        finally:
            await self.compaction_queue.acknowledge()

    async def process_job(self, job: ConversationCompactionJob) -> None:
        with attached_trace_context(job.metadata):
            token = set_trace_id(job.trace_id) if job.trace_id is not None else None
            started_at = start_timer()
            try:
                with business_span(
                    "Process conversation compaction",
                    event="conversation_compaction_processing",
                    conversation_id=str(job.conversation.id),
                    user_id=str(job.conversation.user_id),
                    compaction_job_id=str(job.event_id),
                    compact_through_sequence=job.compact_through_sequence,
                    reason=job.reason,
                ):
                    await self._process_job_with_trace(job)
            except Exception:
                log_exception(
                    logger,
                    "Conversation compaction job failed",
                    event="conversation_compaction_failed",
                    conversation_id=str(job.conversation.id),
                    user_id=str(job.conversation.user_id),
                    compaction_job_id=str(job.event_id),
                    compact_through_sequence=job.compact_through_sequence,
                    reason=job.reason,
                    duration_ms=elapsed_ms(started_at),
                )
                raise
            finally:
                if token is not None:
                    reset_trace_id(token)

    async def _process_job_with_trace(self, job: ConversationCompactionJob) -> None:
        async with self.lock_manager.acquire(job.conversation.id):
            with business_span(
                "Prepare compaction request",
                event="conversation_compaction_request_preparation",
                conversation_id=str(job.conversation.id),
                user_id=str(job.conversation.user_id),
                compaction_job_id=str(job.event_id),
            ):
                request = await self.memory_service.prepare_compaction_request(
                    conversation=job.conversation,
                    compact_through_sequence=job.compact_through_sequence,
                )
            if not request.messages:
                log_event(
                    logger,
                    logging.INFO,
                    "Conversation compaction skipped because no messages are eligible",
                    event="conversation_compaction_skipped",
                    conversation_id=str(job.conversation.id),
                    user_id=str(job.conversation.user_id),
                    compaction_job_id=str(job.event_id),
                    reason="empty_request",
                    compact_through_sequence=job.compact_through_sequence,
                )
                return

            started_at = start_timer()
            with business_span(
                "Run conversation compactor",
                event="conversation_compaction_agent_run",
                conversation_id=str(job.conversation.id),
                user_id=str(job.conversation.user_id),
                compaction_job_id=str(job.event_id),
            ):
                result = await self.compactor.compact(request=request)
            compactor_duration_ms = elapsed_ms(started_at)
            with business_span(
                "Record compaction result",
                event="conversation_compaction_result_recording",
                conversation_id=str(job.conversation.id),
                user_id=str(job.conversation.user_id),
                compaction_job_id=str(job.event_id),
            ):
                await self.memory_service.record_compaction_result(
                    conversation=job.conversation,
                    request=request,
                    result=result,
                    trace_id=job.trace_id,
                )
            log_event(
                logger,
                logging.INFO,
                "Conversation compaction completed",
                event="conversation_compaction_completed",
                conversation_id=str(job.conversation.id),
                user_id=str(job.conversation.user_id),
                compaction_job_id=str(job.event_id),
                compact_through_sequence=job.compact_through_sequence,
                request_message_count=len(request.messages),
                input_token_count=_compaction_usage_input_token_count(result, job=job),
                estimated_input_token_count=estimate_messages_tokens(request.messages),
                output_token_count=result.token_count,
                compactor_duration_ms=compactor_duration_ms,
            )


def _compaction_usage_input_token_count(
    result: ConversationCompactionResult,
    *,
    job: ConversationCompactionJob,
) -> int | None:
    usage = result.metadata.get("run_usage")
    if not isinstance(usage, dict):
        _log_compaction_usage_missing(result, job=job, missing_reason="usage_metadata_missing")
        return None
    input_token_count = usage_token_count(usage, "input_tokens")
    if input_token_count is None:
        _log_compaction_usage_missing(result, job=job, missing_reason="input_tokens_missing")
    return input_token_count


def _log_compaction_usage_missing(
    result: ConversationCompactionResult,
    *,
    job: ConversationCompactionJob,
    missing_reason: str,
) -> None:
    log_event(
        logger,
        logging.WARNING,
        "Conversation compaction input usage is missing",
        event="conversation_compaction_input_usage_missing",
        conversation_id=str(job.conversation.id),
        user_id=str(job.conversation.user_id),
        compaction_job_id=str(job.event_id),
        compact_through_sequence=job.compact_through_sequence,
        missing_reason=missing_reason,
        output_token_count=result.token_count,
    )
