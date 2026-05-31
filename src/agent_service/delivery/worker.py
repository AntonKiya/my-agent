import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent_service.channels.errors import ChannelAdapterNotFoundError
from agent_service.conversations import ConversationLockManager
from agent_service.delivery.interfaces import DeliveryAdapterRegistry
from agent_service.delivery.models import DeliveryResult, DeliveryStatus
from agent_service.observability.events import (
    elapsed_ms,
    log_event,
    log_exception,
    start_timer,
)
from agent_service.observability.tracing import create_trace_id, reset_trace_id, set_trace_id
from agent_service.outbound import OutboundEvent, OutboundQueue

logger = logging.getLogger(__name__)

SleepCallable = Callable[[float], Awaitable[None]]

DEFAULT_DELIVERY_RETRY_BACKOFF_SECONDS = (1.0, 5.0, 15.0)


@dataclass(frozen=True, slots=True)
class DeliveryRetryPolicy:
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = DEFAULT_DELIVERY_RETRY_BACKOFF_SECONDS

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("Delivery retry max_attempts must be at least one")
        if any(delay < 0 for delay in self.backoff_seconds):
            raise ValueError("Delivery retry backoff delays must be greater than or equal to zero")

    def delay_for_attempt(self, attempt_number: int, result: DeliveryResult) -> float:
        if result.retry_after_seconds is not None:
            return result.retry_after_seconds
        if not self.backoff_seconds:
            return 0
        index = min(attempt_number - 1, len(self.backoff_seconds) - 1)
        return self.backoff_seconds[index]


@dataclass(slots=True)
class DeliveryWorker:
    outbound_queue: OutboundQueue
    channel_adapters: DeliveryAdapterRegistry
    lock_manager: ConversationLockManager
    retry_policy: DeliveryRetryPolicy = field(default_factory=DeliveryRetryPolicy)
    error_backoff_seconds: float = 0.1
    sleep: SleepCallable = field(default=asyncio.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.error_backoff_seconds < 0:
            raise ValueError("Delivery worker error backoff must be greater than or equal to zero")

    async def run_forever(self) -> None:
        while True:
            try:
                await self.process_next()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Delivery worker iteration failed",
                    extra={"event": "delivery_worker_iteration_failed"},
                )
                if self.error_backoff_seconds > 0:
                    await self.sleep(self.error_backoff_seconds)

    async def process_next(self) -> None:
        event = await self.outbound_queue.consume()
        try:
            await self.process_event(event)
        finally:
            await self.outbound_queue.acknowledge()

    async def process_event(self, event: OutboundEvent) -> DeliveryResult:
        trace_id = event.trace_id or create_trace_id()
        token = set_trace_id(trace_id)
        event.trace_id = trace_id
        started_at = start_timer()
        try:
            async with self.lock_manager.acquire(event.conversation_id):
                result = await self._send_with_retry(event)
        except Exception:
            log_exception(
                logger,
                "Delivery event processing failed",
                event="delivery_event_processing_failed",
                outbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id),
                conversation_id=str(event.conversation_id),
                status=event.status.value,
                duration_ms=elapsed_ms(started_at),
            )
            raise
        else:
            log_event(
                logger,
                logging.INFO,
                "Delivery event processed",
                event="delivery_event_processed",
                outbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id),
                conversation_id=str(event.conversation_id),
                status=result.status.value,
                duration_ms=elapsed_ms(started_at),
            )
            return result
        finally:
            reset_trace_id(token)

    async def _send_with_retry(self, event: OutboundEvent) -> DeliveryResult:
        last_result: DeliveryResult | None = None
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            started_at = start_timer()
            event.status = DeliveryStatus.SENDING
            result = await self._send_once(event)
            if (
                result.status is DeliveryStatus.FAILED_RETRYABLE
                and attempt_number >= self.retry_policy.max_attempts
            ):
                result = self._retry_exhausted_result(result)

            event.status = result.status
            last_result = result

            log_event(
                logger,
                logging.INFO if result.status is DeliveryStatus.SENT else logging.WARNING,
                "Delivery attempt completed",
                event="delivery_attempt_completed",
                outbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id),
                conversation_id=str(event.conversation_id),
                attempt=attempt_number,
                status=result.status.value,
                error_code=result.error_code,
                duration_ms=elapsed_ms(started_at),
            )

            if result.status is DeliveryStatus.SENT:
                return result
            if result.status is DeliveryStatus.DEAD_LETTER:
                return result

            delay_seconds = self.retry_policy.delay_for_attempt(attempt_number, result)
            log_event(
                logger,
                logging.WARNING,
                "Delivery retry scheduled",
                event="delivery_retry_scheduled",
                outbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id),
                conversation_id=str(event.conversation_id),
                attempt=attempt_number,
                delay_seconds=delay_seconds,
                error_code=result.error_code,
            )
            await self.sleep(delay_seconds)

        if last_result is not None:
            return last_result
        raise RuntimeError("Delivery retry loop exited without a result")

    async def _send_once(self, event: OutboundEvent) -> DeliveryResult:
        try:
            adapter = self.channel_adapters.get(event.channel)
        except ChannelAdapterNotFoundError as exc:
            return DeliveryResult(
                event_id=event.event_id,
                channel=event.channel,
                status=DeliveryStatus.DEAD_LETTER,
                error_code="adapter_not_found",
                error_message=str(exc),
            )

        try:
            return await adapter.send(event)
        except Exception as exc:
            return DeliveryResult(
                event_id=event.event_id,
                channel=event.channel,
                status=DeliveryStatus.FAILED_RETRYABLE,
                error_code="adapter_send_exception",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__},
            )

    def _retry_exhausted_result(self, result: DeliveryResult) -> DeliveryResult:
        return DeliveryResult(
            event_id=result.event_id,
            channel=result.channel,
            status=DeliveryStatus.DEAD_LETTER,
            external_message_ids=list(result.external_message_ids),
            error_code=result.error_code or "delivery_retry_exhausted",
            error_message=result.error_message or "Delivery retry attempts exhausted",
            retry_after_seconds=result.retry_after_seconds,
            metadata={**result.metadata, "retry_exhausted": True},
        )
