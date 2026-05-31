from typing import Protocol, runtime_checkable

from agent_service.messaging.base import EventQueue
from agent_service.outbound.models import OutboundEvent


@runtime_checkable
class OutboundQueue(EventQueue[OutboundEvent], Protocol):
    pass
