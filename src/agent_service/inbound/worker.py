import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from agent_service.agents import AgentBoundary, AgentRequest, AgentResponse
from agent_service.channels import InboundEvent, InboundEventStatus
from agent_service.conversations import (
    Conversation,
    ConversationLockManager,
    ConversationResolverProtocol,
)
from agent_service.delivery.models import DeliveryResult, DeliveryStatus
from agent_service.inbound.errors import OutboundOverloadedError, UnresolvedInboundEventError
from agent_service.inbound.idempotency import InboundIdempotencyStore
from agent_service.inbound.preprocessing import (
    ContentProcessingError,
    InboundContentPreprocessor,
    event_needs_content_preprocessing,
)
from agent_service.memory import (
    ConversationCompactionJob,
    ConversationCompactionPolicyProtocol,
    ConversationMemoryService,
    PreparedConversationContext,
)
from agent_service.messaging.interfaces import CompactionQueue, InboundQueue
from agent_service.observability.events import (
    attached_trace_context,
    business_span,
    elapsed_ms,
    log_event,
    log_exception,
    start_timer,
    store_current_trace_context,
)
from agent_service.observability.tracing import create_trace_id, reset_trace_id, set_trace_id
from agent_service.outbound import OutboundEvent, OutboundQueue

logger = logging.getLogger(__name__)

SleepCallable = Callable[[float], Awaitable[None]]

DEFAULT_AGENT_RETRY_BACKOFF_SECONDS = (1.0, 5.0, 15.0)
DEFAULT_FALLBACK_TEXT = "Sorry, I could not process that message right now. Please try again later."


class ThinkingIndicatorSender(Protocol):
    async def send_thinking_indicator(self, event: InboundEvent) -> DeliveryResult:
        """Best-effort transport-specific signal that processing has started."""
        ...


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
    idempotency_store: InboundIdempotencyStore | None = None
    retry_policy: AgentRetryPolicy = field(default_factory=AgentRetryPolicy)
    fallback_text: str = DEFAULT_FALLBACK_TEXT
    error_backoff_seconds: float = 0.1
    outbound_publish_timeout_seconds: float = 5.0
    thinking_indicator_sender: ThinkingIndicatorSender | None = None
    thinking_indicator_timeout_seconds: float = 1.0
    content_preprocessor: InboundContentPreprocessor | None = None
    compaction_queue: CompactionQueue | None = None
    compaction_policy: ConversationCompactionPolicyProtocol | None = None
    compaction_publish_timeout_seconds: float = 0.1
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.error_backoff_seconds < 0:
            raise ValueError("Inbound worker error backoff must be greater than or equal to zero")
        if self.outbound_publish_timeout_seconds < 0:
            raise ValueError("Outbound publish timeout must be greater than or equal to zero")
        if self.thinking_indicator_timeout_seconds <= 0:
            raise ValueError("Thinking indicator timeout must be greater than zero")
        if self.compaction_publish_timeout_seconds < 0:
            raise ValueError("Compaction publish timeout must be greater than or equal to zero")

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
        with attached_trace_context(event.metadata):
            log_event(
                logger,
                logging.INFO,
                "Inbound event dequeued",
                event="inbound_event_dequeued",
                queue_name="inbound",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                queue_size=self.inbound_queue.stats.size,
                queue_maxsize=self.inbound_queue.stats.maxsize,
            )
        try:
            await self.process_event(event)
        finally:
            await self.inbound_queue.acknowledge()

    async def process_event(self, event: InboundEvent) -> None:
        with attached_trace_context(event.metadata):
            trace_id = event.trace_id or create_trace_id()
            token = set_trace_id(trace_id)
            event.trace_id = trace_id
            event.status = InboundEventStatus.PROCESSING
            await self._mark_idempotency_status(event)
            started_at = start_timer()
            try:
                with business_span(
                    "Process inbound event",
                    event="inbound_event_processing",
                    inbound_event_id=str(event.event_id),
                    channel=event.channel,
                    user_id=str(event.user_id) if event.user_id is not None else None,
                ):
                    await self._process_event_with_trace(event)
            except Exception:
                if event.status is InboundEventStatus.PROCESSING:
                    event.status = InboundEventStatus.FAILED_RETRYABLE
                    with suppress(Exception):
                        await self._mark_idempotency_status(
                            event,
                            failure_reason="inbound event processing failed",
                        )
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
            await self._mark_idempotency_status(
                event,
                failure_reason="event.user_id is missing",
            )
            raise UnresolvedInboundEventError("Inbound worker requires event.user_id")

        with business_span(
            "Resolve conversation",
            event="conversation_resolution",
            inbound_event_id=str(event.event_id),
            channel=event.channel,
            user_id=str(event.user_id),
        ):
            conversation = await self.conversation_resolver.resolve(event)
            log_event(
                logger,
                logging.INFO,
                "Conversation resolved",
                event="conversation_resolved",
                inbound_event_id=str(event.event_id),
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
                channel=conversation.channel,
                conversation_type=conversation.type.value,
            )
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
            with business_span(
                "Preprocess inbound content",
                event="inbound_content_preprocessing",
                inbound_event_id=str(event.event_id),
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
            ):
                content_preprocessed = await self._preprocess_content(
                    event,
                    conversation=conversation,
                )
                if not content_preprocessed:
                    return
            with business_span(
                "Record user message",
                event="memory_user_message_recording",
                inbound_event_id=str(event.event_id),
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
            ):
                user_message = await self.memory_service.record_user_message(
                    conversation=conversation,
                    event=event,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "User message recorded",
                    event="memory_user_message_recorded",
                    inbound_event_id=str(event.event_id),
                    conversation_id=str(conversation.id),
                    user_id=str(conversation.user_id),
                    message_id=str(user_message.id),
                    sequence=user_message.sequence,
                )
            with business_span(
                "Prepare agent context",
                event="agent_context_preparation",
                inbound_event_id=str(event.event_id),
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
            ):
                prepared_context = await self.memory_service.prepare_agent_context(
                    conversation=conversation,
                    latest_user_message=user_message,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "Agent context prepared",
                    event="agent_context_prepared",
                    inbound_event_id=str(event.event_id),
                    conversation_id=str(conversation.id),
                    user_id=str(conversation.user_id),
                    snapshot_source=prepared_context.metadata.get("snapshot_source"),
                    snapshot_version=prepared_context.metadata.get("snapshot_version"),
                )
            await self._send_thinking_indicator(event)
            try:
                response = await self._run_agent_with_retry(
                    request=self._agent_request(
                        event=event,
                        conversation_id=conversation.id,
                        prepared_context=prepared_context,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "Agent call failed after retries",
                    extra={
                        "event": "inbound_agent_dead_letter",
                        "inbound_event_id": str(event.event_id),
                        "conversation_id": str(conversation.id),
                        "user_id": str(event.user_id),
                        **_agent_error_log_fields(exc),
                    },
                )
                await self._publish_fallback_event(event, conversation_id=conversation.id)
                event.status = InboundEventStatus.FALLBACK_SENT
                await self._mark_idempotency_status(
                    event,
                    failure_reason="agent call failed after retries",
                )
                return

            outbound_event = self._outbound_event(
                event=event,
                conversation_id=conversation.id,
                response=response,
            )
            # Publish before persisting the assistant message: if the outbound
            # queue is saturated and the publish times out, no "delivered"
            # message is left in memory, keeping a retry safe and idempotent.
            await self._publish_outbound(
                outbound_event,
                inbound_event=event,
                conversation_id=conversation.id,
            )
            with business_span(
                "Record assistant message",
                event="memory_assistant_message_recording",
                inbound_event_id=str(event.event_id),
                outbound_event_id=str(outbound_event.event_id),
                conversation_id=str(conversation.id),
                user_id=str(conversation.user_id),
            ):
                assistant_message = await self.memory_service.record_assistant_message(
                    conversation=conversation,
                    response=response,
                    trace_id=response.trace_id or event.trace_id,
                    outbound_event_id=outbound_event.event_id,
                )
                log_event(
                    logger,
                    logging.INFO,
                    "Assistant message recorded",
                    event="memory_assistant_message_recorded",
                    inbound_event_id=str(event.event_id),
                    outbound_event_id=str(outbound_event.event_id),
                    conversation_id=str(conversation.id),
                    user_id=str(conversation.user_id),
                    message_id=str(assistant_message.id),
                    sequence=assistant_message.sequence,
                )
            event.status = InboundEventStatus.COMPLETED
            await self._mark_idempotency_status(event)
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
        with business_span(
            "Run agent",
            event="agent_run",
            inbound_event_id=str(request.inbound_event_id),
            conversation_id=str(request.conversation_id),
            user_id=str(request.user_id),
            channel=request.channel,
        ):
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
                        usage_input_tokens=(
                            response.usage.input_tokens if response.usage is not None else None
                        ),
                        usage_output_tokens=(
                            response.usage.output_tokens if response.usage is not None else None
                        ),
                        usage_total_tokens=(
                            response.usage.total_tokens if response.usage is not None else None
                        ),
                        usage_requests=(
                            response.usage.metadata.get("requests")
                            if response.usage is not None
                            else None
                        ),
                        usage_tool_calls=(
                            response.usage.metadata.get("tool_calls")
                            if response.usage is not None
                            else None
                        ),
                        **_pydantic_ai_new_message_counts(response),
                    )
                    return response
                except Exception as exc:
                    if attempt_number >= self.retry_policy.max_attempts:
                        log_exception(
                            logger,
                            "Agent run failed after final attempt",
                            event="agent_run_failed",
                            inbound_event_id=str(request.inbound_event_id),
                            conversation_id=str(request.conversation_id),
                            user_id=str(request.user_id),
                            channel=request.channel,
                            attempt=attempt_number,
                            duration_ms=elapsed_ms(started_at),
                            **_agent_error_log_fields(exc),
                        )
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
                        delay_seconds=delay_seconds,
                        duration_ms=elapsed_ms(started_at),
                        **_agent_error_log_fields(exc),
                    )
                    await self.sleep(delay_seconds)
        raise RuntimeError("Agent retry loop exited without a response")

    async def _send_thinking_indicator(self, event: InboundEvent) -> None:
        if self.thinking_indicator_sender is None:
            return
        try:
            result = await asyncio.wait_for(
                self.thinking_indicator_sender.send_thinking_indicator(event),
                timeout=self.thinking_indicator_timeout_seconds,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.DEBUG,
                "Thinking indicator send skipped",
                event="thinking_indicator_send_skipped",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                error_type=type(exc).__name__,
            )
            return
        if result.status is not DeliveryStatus.SENT:
            log_event(
                logger,
                logging.DEBUG,
                "Thinking indicator send skipped",
                event="thinking_indicator_send_skipped",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                status=result.status.value,
                error_code=result.error_code,
                retry_after_seconds=result.retry_after_seconds,
            )

    async def _preprocess_content(self, event: InboundEvent, *, conversation: Conversation) -> bool:
        if self.content_preprocessor is None:
            if not event_needs_content_preprocessing(event):
                return True
            await self._publish_content_processing_fallback(
                event,
                conversation=conversation,
                failure_reason="content preprocessor is not configured",
                error_code="content_preprocessor_not_configured",
            )
            return False

        try:
            await self.content_preprocessor.process(event)
        except ContentProcessingError as exc:
            logger.warning(
                "Inbound content preprocessing failed before agent run",
                extra={
                    "event": "inbound_content_preprocessing_failed",
                    "inbound_event_id": str(event.event_id),
                    "conversation_id": str(conversation.id),
                    "user_id": str(conversation.user_id),
                    "channel": event.channel,
                    "message_type": event.message_type.value,
                    "retryable": exc.retryable,
                    "error_code": exc.error_code,
                    "error_type": type(exc).__name__,
                },
            )
            await self._publish_content_processing_fallback(
                event,
                conversation=conversation,
                failure_reason="content preprocessing failed",
                error_code=exc.error_code,
            )
            return False
        return True

    async def _publish_content_processing_fallback(
        self,
        event: InboundEvent,
        *,
        conversation: Conversation,
        failure_reason: str,
        error_code: str | None,
    ) -> None:
        await self._publish_fallback_event(event, conversation_id=conversation.id)
        event.status = InboundEventStatus.FALLBACK_SENT
        event.metadata["content_processing"] = {
            "status": "failed",
            "error_code": error_code,
        }
        await self._mark_idempotency_status(event, failure_reason=failure_reason)

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

    async def _publish_outbound(
        self,
        outbound_event: OutboundEvent,
        *,
        inbound_event: InboundEvent,
        conversation_id: UUID,
    ) -> None:
        if self.outbound_publish_timeout_seconds == 0:
            store_current_trace_context(outbound_event.metadata)
            await self.outbound_queue.publish(outbound_event)
            self._log_outbound_published(
                outbound_event,
                inbound_event=inbound_event,
                conversation_id=conversation_id,
            )
            return
        try:
            store_current_trace_context(outbound_event.metadata)
            await asyncio.wait_for(
                self.outbound_queue.publish(outbound_event),
                timeout=self.outbound_publish_timeout_seconds,
            )
            self._log_outbound_published(
                outbound_event,
                inbound_event=inbound_event,
                conversation_id=conversation_id,
            )
        except TimeoutError as exc:
            log_event(
                logger,
                logging.WARNING,
                "Outbound queue publish timed out",
                event="outbound_queue_overloaded",
                inbound_event_id=str(inbound_event.event_id),
                outbound_event_id=str(outbound_event.event_id),
                conversation_id=str(conversation_id),
                user_id=(str(inbound_event.user_id) if inbound_event.user_id is not None else None),
                channel=outbound_event.channel,
                queue_size=self.outbound_queue.stats.size,
                queue_maxsize=self.outbound_queue.stats.maxsize,
                publish_timeout_seconds=self.outbound_publish_timeout_seconds,
            )
            raise OutboundOverloadedError("Outbound queue publish timed out") from exc

    def _log_outbound_published(
        self,
        outbound_event: OutboundEvent,
        *,
        inbound_event: InboundEvent,
        conversation_id: UUID,
    ) -> None:
        log_event(
            logger,
            logging.INFO,
            "Outbound event published",
            event="outbound_event_published",
            queue_name="outbound",
            inbound_event_id=str(inbound_event.event_id),
            outbound_event_id=str(outbound_event.event_id),
            conversation_id=str(conversation_id),
            user_id=str(inbound_event.user_id) if inbound_event.user_id is not None else None,
            channel=outbound_event.channel,
            queue_size=self.outbound_queue.stats.size,
            queue_maxsize=self.outbound_queue.stats.maxsize,
        )

    async def _publish_fallback_event(
        self,
        event: InboundEvent,
        *,
        conversation_id: UUID,
    ) -> None:
        if event.user_id is None:
            raise UnresolvedInboundEventError("Inbound worker requires event.user_id")
        outbound_event = OutboundEvent(
            channel=event.channel,
            user_id=event.user_id,
            conversation_id=conversation_id,
            external_chat_id=event.external_chat_id,
            text=self.fallback_text,
            thread_id=event.thread_id,
            metadata={"fallback": True, "failed_inbound_event_id": str(event.event_id)},
            trace_id=event.trace_id,
        )
        await self._publish_outbound(
            outbound_event,
            inbound_event=event,
            conversation_id=conversation_id,
        )
        log_event(
            logger,
            logging.WARNING,
            "Fallback outbound event published",
            event="fallback_outbound_event_published",
            queue_name="outbound",
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
            store_current_trace_context(job.metadata)
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
                queue_name="compaction",
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

    async def _mark_idempotency_status(
        self,
        event: InboundEvent,
        *,
        failure_reason: str | None = None,
    ) -> None:
        if self.idempotency_store is None:
            return
        await self.idempotency_store.mark_status(
            event_id=event.event_id,
            status=event.status,
            failure_reason=failure_reason,
        )


def _pydantic_ai_new_message_counts(response: AgentResponse) -> dict[str, int]:
    message_count = len(response.pydantic_ai_new_messages)
    part_count = 0
    tool_call_count = 0
    tool_result_count = 0
    for message in response.pydantic_ai_new_messages:
        for part in message.parts:
            part_count += 1
            if isinstance(part, ToolCallPart):
                tool_call_count += 1
            elif isinstance(part, ToolReturnPart):
                tool_result_count += 1
    return {
        "pydantic_ai_new_message_count": message_count,
        "pydantic_ai_new_part_count": part_count,
        "pydantic_ai_tool_call_count": tool_call_count,
        "pydantic_ai_tool_result_count": tool_result_count,
    }


def _agent_error_log_fields(exc: BaseException) -> dict[str, object]:
    fields: dict[str, object] = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    for attr, field_name in (
        ("status_code", "error_status_code"),
        ("model_name", "error_model_name"),
        ("body", "error_body"),
    ):
        value = getattr(exc, attr, None)
        if value is not None:
            fields[field_name] = value
            if attr == "body":
                fields["error_body_type"] = type(value).__name__

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            fields["error_response_status_code"] = status_code
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            fields["error_response_text"] = _truncate_log_text(text)
        headers = getattr(response, "headers", None)
        if headers is not None:
            for header_name in (
                "x-request-id",
                "x-openrouter-request-id",
                "cf-ray",
            ):
                header_value = headers.get(header_name)
                if header_value:
                    fields[f"error_response_header_{header_name.replace('-', '_')}"] = header_value

    cause = exc.__cause__
    if cause is not None:
        fields["error_cause_type"] = type(cause).__name__
        fields["error_cause_message"] = str(cause)

    return fields


def _truncate_log_text(value: str, *, max_length: int = 8000) -> str:
    if len(value) <= max_length:
        return value
    omitted = len(value) - max_length
    return f"{value[:max_length]}...[truncated {omitted} chars]"
