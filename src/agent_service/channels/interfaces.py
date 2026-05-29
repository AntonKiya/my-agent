from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from agent_service.channels.models import ChannelName, DeliveryResult, InboundEvent, OutboundEvent

RawInboundPayloadT = TypeVar("RawInboundPayloadT", contravariant=True)


@runtime_checkable
class ChannelInboundNormalizer(Protocol[RawInboundPayloadT]):
    channel: ChannelName

    async def normalize(self, payload: RawInboundPayloadT) -> InboundEvent | None:
        """Convert a transport-specific update into a channel-agnostic inbound event."""
        ...


@runtime_checkable
class ChannelAdapter(Protocol):
    channel: ChannelName

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        """Deliver a channel-agnostic outbound event through the concrete transport."""
        ...


@runtime_checkable
class ChannelAdapterRegistry(Protocol):
    def register(self, adapter: ChannelAdapter) -> None:
        """Register an adapter for its channel name."""
        ...

    def get(self, channel: ChannelName) -> ChannelAdapter:
        """Resolve an adapter by channel name."""
        ...


RawMappingInboundNormalizer = ChannelInboundNormalizer[Mapping[str, Any]]
