import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

from agent_service.agents import AgentBoundary, AgentRequest, AgentResponse
from agent_service.channels import InboundEvent, InboundEventStatus, OutboundEvent
from agent_service.conversations import (
    Conversation,
    ConversationLockManager,
    ConversationResolverProtocol,
)
from agent_service.inbound.errors import UnresolvedInboundEventError
from agent_service.memory import (
    ConversationCompactionJob,
    ConversationCompactionPolicyProtocol,
    ConversationMemoryService,
    PreparedConversationContext,
)
from agent_service.messaging import CompactionQueue, InboundQueue, OutboundQueue
from agent_service.observability.events import (
    elapsed_ms,
    log_event,
    log_exception,
    start_timer,
)
from agent_service.observability.tracing import create_trace_id, reset_trace_id, set_trace_id

logger = logging.getLogger(__name__)

SleepCallable = Callable[[float], Awaitable[None]]

DEFAULT_AGENT_RETRY_BACKOFF_SECONDS = (1.0, 5.0, 15.0)
DEFAULT_FALLBACK_TEXT = "Sorry, I could not process that message right now. Please try again later."


@dataclass(frozen=True, slots=True)
class AgentRetryPolicy:
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = DEFAULT_AGENT_RETRY_BACKOFF_SECONDS

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("Agent retry max_attempts must be at least one")
        if any(delay < 0 for delay in self.backoff_seconds):
            raise ValueError("Agent retry backoff delays must be greater than or equal to zero")

    def delay_for_attempt(self, attempt_number: int) -> float:
        if not self.backoff_seconds:
            return 0
        index = min(attempt_number - 1, len(self.backoff_seconds) - 1)
        return self.backoff_seconds[index]


@dataclass(slots=True)
class InboundWorker:
    inbound_queue: InboundQueue
    outbound_queue: OutboundQueue
    conversation_resolver: ConversationResolverProtocol
    memory_service: ConversationMemoryService
    agent_boundary: AgentBoundary
    lock_manager: ConversationLockManager
    retry_policy: AgentRetryPolicy = field(default_factory=AgentRetryPolicy)
    fallback_text: str = DEFAULT_FALLBACK_TEXT
    error_backoff_seconds: float = 0.1
    compaction_queue: CompactionQueue | None = None
    compaction_policy: ConversationCompactionPolicyProtocol | None = None
    compaction_publish_timeout_seconds: float = 0.1
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.error_backoff_seconds < 0:
            raise ValueError("Inbound worker error backoff must be greater than or equal to zero")
        if self.compaction_publish_timeout_seconds < 0:
            raise ValueError(
                "Compaction publish timeout must be greater than or equal to zero"
            )

    async def run_forever(self) -> None:
        while True:
            try:
                await self.process_next()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Inbound worker iteration failed",
                    extra={"event": "inbound_worker_iteration_failed"},
                )
                if self.error_backoff_seconds > 0:
                    await self.sleep(self.error_backoff_seconds)

    async def process_next(self) -> None:
        event = await self.inbound_queue.consume()
        await self.process_event(event)

    async def process_event(self, event: InboundEvent) -> None:
        trace_id = event.trace_id or create_trace_id()
        token = set_trace_id(trace_id)
        event.trace_id = trace_id
        event.status = InboundEventStatus.PROCESSING
        started_at = start_timer()
        try:
            await self._process_event_with_trace(event)
        except Exception:
            log_exception(
                logger,
                "Inbound event processing failed",
                event="inbound_event_processing_failed",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                status=event.status.value,
                duration_ms=elapsed_ms(started_at),
            )
            raise
        else:
            log_event(
                logger,
                logging.INFO,
                "Inbound event processed",
                event="inbound_event_processed",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                status=event.status.value,
                duration_ms=elapsed_ms(started_at),
            )
        finally:
            reset_trace_id(token)

    async def _process_event_with_trace(self, event: InboundEvent) -> None:
        if event.user_id is None:
            event.status = InboundEventStatus.DEAD_LETTER
            raise UnresolvedInboundEventError("Inbound worker requires event.user_id")

        conversation = await self.conversation_resolver.resolve(event)
        async with self.lock_manager.acquire(conversation.id):
            lock_started_at = start_timer()
            log_event(
                logger,
                logging.DEBUG,
                "Conversation lock acquired for inbound event",
                event="conversation_lock_acquired",
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
                inbound_event_id=str(event.event_id),
            )
            user_message = await self.memory_service.record_user_message(
                conversation=conversation,
                event=event,
            )
            prepared_context = await self.memory_service.prepare_agent_context(
                conversation=conversation,
                latest_user_message=user_message,
            )
            try:
                response = await self._run_agent_with_retry(
                    request=self._agent_request(
                        event=event,
                        conversation_id=conversation.id,
                        prepared_context=prepared_context,
                    )
                )
            except Exception:
                logger.exception(
                    "Agent call failed after retries",
                    extra={
                        "event": "inbound_agent_dead_letter",
                        "inbound_event_id": str(event.event_id),
                        "conversation_id": str(conversation.id),
                        "user_id": str(event.user_id),
                    },
                )
                await self._publish_fallback_event(event, conversation_id=conversation.id)
                event.status = InboundEventStatus.FALLBACK_SENT
                return

            outbound_event = self._outbound_event(
                event=event,
                conversation_id=conversation.id,
                response=response,
            )
            await self.memory_service.record_assistant_message(
                conversation=conversation,
                response=response,
                trace_id=response.trace_id or event.trace_id,
                outbound_event_id=outbound_event.event_id,
            )
            await self.outbound_queue.publish(outbound_event)
            log_event(
                logger,
                logging.INFO,
                "Outbound event published",
                event="outbound_event_published",
                inbound_event_id=str(event.event_id),
                outbound_event_id=str(outbound_event.event_id),
                conversation_id=str(conversation.id),
                user_id=str(event.user_id),
                channel=outbound_event.channel,
                queue_size=self.outbound_queue.stats.size,
                queue_maxsize=self.outbound_queue.stats.maxsize,
            )
            event.status = InboundEventStatus.COMPLETED
            await self._schedule_compaction_if_needed(event=event, conversation=conversation)
            log_event(
                logger,
                logging.DEBUG,
                "Conversation lock released for inbound event",
                event="conversation_lock_released",
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
                inbound_event_id=str(event.event_id),
                duration_ms=elapsed_ms(lock_started_at),
            )

    async def _run_agent_with_retry(self, *, request: AgentRequest) -> AgentResponse:
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            started_at = start_timer()
            try:
                response = await self.agent_boundary.run(request)
                log_event(
                    logger,
                    logging.INFO,
                    "Agent run completed",
                    event="agent_run_completed",
                    inbound_event_id=str(request.inbound_event_id),
                    conversation_id=str(request.conversation_id),
                    user_id=str(request.user_id),
                    channel=request.channel,
                    attempt=attempt_number,
                    duration_ms=elapsed_ms(started_at),
                )
                return response
            except Exception as exc:
                if attempt_number >= self.retry_policy.max_attempts:
                    raise
                request.metadata["retry_attempt"] = attempt_number
                delay_seconds = self.retry_policy.delay_for_attempt(attempt_number)
                log_event(
                    logger,
                    logging.WARNING,
                    "Agent run attempt failed and will be retried",
                    event="agent_run_retry_scheduled",
                    inbound_event_id=str(request.inbound_event_id),
                    conversation_id=str(request.conversation_id),
                    user_id=str(request.user_id),
                    channel=request.channel,
                    attempt=attempt_number,
                    error_type=type(exc).__name__,
                    delay_seconds=delay_seconds,
                    duration_ms=elapsed_ms(started_at),
                )
                await self.sleep(delay_seconds)
        raise RuntimeError("Agent retry loop exited without a response")

    def _agent_request(
        self,
        *,
        event: InboundEvent,
        conversation_id: UUID,
        prepared_context: PreparedConversationContext,
    ) -> AgentRequest:
        if event.user_id is None:
            raise UnresolvedInboundEventError("Inbound worker requires event.user_id")
        return AgentRequest(
            user_id=event.user_id,
            conversation_id=prepared_context.conversation_id,
            inbound_event_id=event.event_id,
            channel=event.channel,
            text=event.text,
            attachments=list(event.attachments),
            context=prepared_context.agent_context,
            pydantic_ai=prepared_context.pydantic_ai,
            metadata={
                "idempotency_key": event.idempotency_key,
                "external_message_id": event.external_message_id,
                "conversation_id": str(conversation_id),
            },
            trace_id=event.trace_id,
        )

    def _outbound_event(
        self,
        *,
        event: InboundEvent,
        conversation_id: UUID,
        response: AgentResponse,
    ) -> OutboundEvent:
        if event.user_id is None:
            raise UnresolvedInboundEventError("Inbound worker requires event.user_id")
        return OutboundEvent(
            channel=event.channel,
            user_id=event.user_id,
            conversation_id=conversation_id,
            external_chat_id=event.external_chat_id,
            text=response.text,
            thread_id=event.thread_id,
            channel_metadata={},
            metadata=response.metadata,
            trace_id=response.trace_id or event.trace_id,
        )

    async def _publish_fallback_event(
        self,
        event: InboundEvent,
        *,
        conversation_id: UUID,
    ) -> None:
        if event.user_id is None:
            raise UnresolvedInboundEventError("Inbound worker requires event.user_id")
        await self.outbound_queue.publish(
            outbound_event := OutboundEvent(
                channel=event.channel,
                user_id=event.user_id,
                conversation_id=conversation_id,
                external_chat_id=event.external_chat_id,
                text=self.fallback_text,
                thread_id=event.thread_id,
                metadata={"fallback": True, "failed_inbound_event_id": str(event.event_id)},
                trace_id=event.trace_id,
            )
        )
        log_event(
            logger,
            logging.WARNING,
            "Fallback outbound event published",
            event="fallback_outbound_event_published",
            inbound_event_id=str(event.event_id),
            outbound_event_id=str(outbound_event.event_id),
            conversation_id=str(conversation_id),
            user_id=str(event.user_id),
            channel=event.channel,
            queue_size=self.outbound_queue.stats.size,
            queue_maxsize=self.outbound_queue.stats.maxsize,
        )

    async def _schedule_compaction_if_needed(
        self,
        *,
        event: InboundEvent,
        conversation: Conversation,
    ) -> None:
        if self.compaction_queue is None or self.compaction_policy is None:
            return

        decision = await self.memory_service.evaluate_compaction(
            conversation=conversation,
            policy=self.compaction_policy,
        )
        if not decision.should_compact or decision.compact_through_sequence is None:
            return

        job = ConversationCompactionJob(
            conversation=conversation,
            compact_through_sequence=decision.compact_through_sequence,
            reason=decision.reason,
            trace_id=event.trace_id,
            metadata={
                "inbound_event_id": str(event.event_id),
                "estimated_input_tokens": decision.estimated_input_tokens,
                "trigger_tokens": decision.trigger_tokens,
                "recent_tail_budget_tokens": decision.recent_tail_budget_tokens,
                "keep_from_sequence": decision.keep_from_sequence,
                "compactable_token_count": decision.compactable_token_count,
                "retained_tail_token_count": decision.retained_tail_token_count,
            },
        )
        try:
            if self.compaction_publish_timeout_seconds == 0:
                await self.compaction_queue.publish(job)
            else:
                await asyncio.wait_for(
                    self.compaction_queue.publish(job),
                    timeout=self.compaction_publish_timeout_seconds,
                )
            log_event(
                logger,
                logging.INFO,
                "Conversation compaction scheduled",
                event="conversation_compaction_scheduled",
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
                inbound_event_id=str(event.event_id),
                compaction_job_id=str(job.event_id),
                compact_through_sequence=job.compact_through_sequence,
                reason=job.reason,
                queue_size=self.compaction_queue.stats.size,
                queue_maxsize=self.compaction_queue.stats.maxsize,
                estimated_input_tokens=decision.estimated_input_tokens,
                trigger_tokens=decision.trigger_tokens,
                recent_tail_budget_tokens=decision.recent_tail_budget_tokens,
            )
        except TimeoutError:
            logger.warning(
                "Conversation compaction scheduling timed out",
                extra={
                    "event": "conversation_compaction_schedule_timeout",
                    "conversation_id": str(conversation.id),
                    "user_id": str(conversation.user_id),
                    "inbound_event_id": str(event.event_id),
                },
            )
