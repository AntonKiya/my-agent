from typing import Protocol, runtime_checkable

from agent_service.delivery.models import DeliveryResult
from agent_service.outbound import OutboundEvent


@runtime_checkable
class DeliveryAdapter(Protocol):
    channel: str

    async def send(self, event: OutboundEvent) -> DeliveryResult:
        """Deliver an outbound event through a concrete channel adapter."""
        ...


@runtime_checkable
class DeliveryAdapterRegistry(Protocol):
    def get(self, channel: str) -> DeliveryAdapter:
        """Resolve a delivery-capable adapter by channel name."""
        ...
