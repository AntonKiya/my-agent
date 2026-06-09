import asyncio
import logging
from typing import Protocol, runtime_checkable

from agent_service.channels.models import InboundEvent
from agent_service.inbound.idempotency import InboundIdempotencyStore
from agent_service.inbound.media_groups import (
    MediaGroupAddStatus,
    MediaGroupBufferError,
    RedisInboundMediaGroupAggregator,
    media_group_key,
)
from agent_service.messaging.interfaces import InboundQueue
from agent_service.observability.events import (
    business_span,
    elapsed_ms,
    log_event,
    start_timer,
    store_current_trace_context,
)
from agent_service.users import UserResolutionError, UserResolutionResult, UserResolutionStatus

from .models import InboundIntakeResult, InboundIntakeStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class InboundUserResolver(Protocol):
    async def resolve(self, event: InboundEvent) -> UserResolutionResult:
        """Resolve the internal user for a normalized inbound event."""
        ...


@runtime_checkable
class InboundIntake(Protocol):
    async def accept(self, event: InboundEvent) -> InboundIntakeResult:
        """Accept a normalized event and publish it only after user resolution."""
        ...


class InboundIntakeService(InboundIntake):
    def __init__(
        self,
        *,
        user_resolver: InboundUserResolver,
        inbound_queue: InboundQueue,
        idempotency_store: InboundIdempotencyStore | None = None,
        media_group_aggregator: RedisInboundMediaGroupAggregator | None = None,
        publish_timeout_seconds: float | None = None,
    ) -> None:
        if publish_timeout_seconds is not None and publish_timeout_seconds <= 0:
            raise ValueError("Inbound publish timeout must be greater than zero")
        self._user_resolver = user_resolver
        self._inbound_queue = inbound_queue
        self._idempotency_store = idempotency_store
        self._media_group_aggregator = media_group_aggregator
        self._publish_timeout_seconds = publish_timeout_seconds

    async def accept(self, event: InboundEvent) -> InboundIntakeResult:
        started_at = start_timer()
        with business_span(
            "Accept inbound event",
            event="inbound_intake_accept",
            channel=event.channel,
            inbound_event_id=str(event.event_id),
        ):
            resolution = await self._user_resolver.resolve(event)
            log_event(
                logger,
                logging.INFO,
                "Inbound user resolution completed",
                event="inbound_user_resolution_completed",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_resolution_status=resolution.status.value,
                user_id=str(resolution.user.id) if resolution.user is not None else None,
                duration_ms=elapsed_ms(started_at),
            )

            if resolution.status is UserResolutionStatus.RESOLVED:
                if resolution.event is None:
                    raise UserResolutionError("Resolved user result did not include an event")
                media_group_result = await self._buffer_media_group_event_if_needed(
                    resolution.event,
                    started_at=started_at,
                )
                if media_group_result is not None:
                    return media_group_result
                if self._idempotency_store is not None:
                    claim = await self._idempotency_store.claim(resolution.event)
                    if not claim.claimed:
                        log_event(
                            logger,
                            logging.INFO,
                            "Duplicate inbound event suppressed",
                            event="inbound_event_duplicate_suppressed",
                            inbound_event_id=str(resolution.event.event_id),
                            existing_inbound_event_id=(
                                str(claim.existing_event_id)
                                if claim.existing_event_id is not None
                                else None
                            ),
                            existing_status=(
                                claim.existing_status.value
                                if claim.existing_status is not None
                                else None
                            ),
                            channel=resolution.event.channel,
                            user_id=str(resolution.event.user_id),
                            duration_ms=elapsed_ms(started_at),
                        )
                        return InboundIntakeResult(
                            status=InboundIntakeStatus.DUPLICATE,
                            published=False,
                            user_resolution_status=resolution.status,
                            reason="duplicate inbound event",
                        )
                    log_event(
                        logger,
                        logging.INFO,
                        "Inbound idempotency claim acquired",
                        event="inbound_idempotency_claimed",
                        inbound_event_id=str(resolution.event.event_id),
                        channel=resolution.event.channel,
                        user_id=str(resolution.event.user_id),
                    )
                published = await self._publish_resolved_event(resolution.event)
                queue_stats = self._inbound_queue.stats
                if not published:
                    if self._idempotency_store is not None:
                        await self._idempotency_store.release_claim(
                            event_id=resolution.event.event_id,
                        )
                    logger.warning(
                        "Inbound queue publish timed out",
                        extra={
                            "event": "inbound_queue_overloaded",
                            "queue_name": "inbound",
                            "inbound_event_id": str(resolution.event.event_id),
                            "user_id": str(resolution.event.user_id),
                            "queue_size": queue_stats.size,
                            "queue_maxsize": queue_stats.maxsize,
                            "publish_timeout_seconds": self._publish_timeout_seconds,
                            "duration_ms": elapsed_ms(started_at),
                        },
                    )
                    return InboundIntakeResult(
                        status=InboundIntakeStatus.OVERLOADED,
                        published=False,
                        user_resolution_status=resolution.status,
                        reason="inbound queue is overloaded",
                        queue_size=queue_stats.size,
                        queue_maxsize=queue_stats.maxsize,
                    )
                log_event(
                    logger,
                    logging.INFO,
                    "Inbound event published",
                    event="inbound_event_published",
                    queue_name="inbound",
                    inbound_event_id=str(resolution.event.event_id),
                    channel=resolution.event.channel,
                    user_id=str(resolution.event.user_id),
                    queue_size=queue_stats.size,
                    queue_maxsize=queue_stats.maxsize,
                    duration_ms=elapsed_ms(started_at),
                )
                return InboundIntakeResult(
                    status=InboundIntakeStatus.PUBLISHED,
                    published=True,
                    user_resolution_status=resolution.status,
                    queue_size=queue_stats.size,
                    queue_maxsize=queue_stats.maxsize,
                )

            log_event(
                logger,
                logging.INFO,
                "Inbound event rejected",
                event="inbound_event_rejected",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_resolution_status=resolution.status.value,
                reason=resolution.reason,
                duration_ms=elapsed_ms(started_at),
            )
            return InboundIntakeResult(
                status=InboundIntakeStatus.REJECTED,
                published=False,
                user_resolution_status=resolution.status,
                reason=resolution.reason,
            )

    async def _buffer_media_group_event_if_needed(
        self,
        event: InboundEvent,
        *,
        started_at: float,
    ) -> InboundIntakeResult | None:
        try:
            group_key = media_group_key(event)
        except MediaGroupBufferError:
            group_key = None
        if group_key is None:
            return None
        if self._media_group_aggregator is None:
            log_event(
                logger,
                logging.WARNING,
                "Inbound media group aggregation unavailable",
                event="inbound_media_group_aggregation_unavailable",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                media_group_id=event.channel_metadata.get("media_group_id"),
            )
            return None

        try:
            result = await self._media_group_aggregator.add(event)
        except MediaGroupBufferError as exc:
            log_event(
                logger,
                logging.WARNING,
                "Inbound media group buffering failed",
                event="inbound_media_group_buffering_failed",
                inbound_event_id=str(event.event_id),
                channel=event.channel,
                user_id=str(event.user_id) if event.user_id is not None else None,
                media_group_id=event.channel_metadata.get("media_group_id"),
                group_key=group_key,
                error_type=type(exc).__name__,
                duration_ms=elapsed_ms(started_at),
            )
            return None

        if result.status is MediaGroupAddStatus.BUFFERED:
            return InboundIntakeResult(
                status=InboundIntakeStatus.BUFFERED,
                published=False,
                user_resolution_status=UserResolutionStatus.RESOLVED,
                reason="inbound media group item buffered",
            )

        log_event(
            logger,
            logging.INFO,
            "Inbound media group item duplicate suppressed",
            event="inbound_media_group_item_duplicate_suppressed",
            inbound_event_id=str(event.event_id),
            channel=event.channel,
            user_id=str(event.user_id) if event.user_id is not None else None,
            media_group_id=event.channel_metadata.get("media_group_id"),
            group_key=group_key,
            item_status=result.status.value,
            duration_ms=elapsed_ms(started_at),
        )
        return InboundIntakeResult(
            status=InboundIntakeStatus.DUPLICATE,
            published=False,
            user_resolution_status=UserResolutionStatus.RESOLVED,
            reason="duplicate inbound media group item",
        )

    async def _publish_resolved_event(self, event: InboundEvent) -> bool:
        store_current_trace_context(event.metadata)
        if self._publish_timeout_seconds is None:
            await self._inbound_queue.publish(event)
            return True
        try:
            await asyncio.wait_for(
                self._inbound_queue.publish(event),
                timeout=self._publish_timeout_seconds,
            )
        except TimeoutError:
            return False
        return True
