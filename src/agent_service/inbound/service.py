from typing import Protocol, runtime_checkable

from agent_service.channels.models import InboundEvent
from agent_service.messaging import InboundQueue
from agent_service.users import UserResolutionError, UserResolutionResult, UserResolutionStatus

from .models import InboundIntakeResult, InboundIntakeStatus


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
    ) -> None:
        self._user_resolver = user_resolver
        self._inbound_queue = inbound_queue

    async def accept(self, event: InboundEvent) -> InboundIntakeResult:
        resolution = await self._user_resolver.resolve(event)

        if resolution.status is UserResolutionStatus.RESOLVED:
            if resolution.event is None:
                raise UserResolutionError("Resolved user result did not include an event")
            await self._inbound_queue.publish(resolution.event)
            return InboundIntakeResult(
                status=InboundIntakeStatus.PUBLISHED,
                published=True,
                user_resolution_status=resolution.status,
            )

        return InboundIntakeResult(
            status=InboundIntakeStatus.REJECTED,
            published=False,
            user_resolution_status=resolution.status,
            reason=resolution.reason,
        )
