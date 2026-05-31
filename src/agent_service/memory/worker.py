import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent_service.conversations import ConversationLockManager
from agent_service.memory.interfaces import ConversationCompactor, ConversationMemoryService
from agent_service.memory.models import ConversationCompactionJob
from agent_service.messaging.interfaces import CompactionQueue
from agent_service.observability.events import (
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
        try:
            await self.process_job(job)
        finally:
            await self.compaction_queue.acknowledge()

    async def process_job(self, job: ConversationCompactionJob) -> None:
        token = set_trace_id(job.trace_id) if job.trace_id is not None else None
        started_at = start_timer()
        try:
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
            result = await self.compactor.compact(request=request)
            compactor_duration_ms = elapsed_ms(started_at)
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
                input_token_count=sum(message.token_count or 0 for message in request.messages),
                output_token_count=result.token_count,
                compactor_duration_ms=compactor_duration_ms,
            )
