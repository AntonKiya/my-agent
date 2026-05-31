from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent_service.messaging.base import EventQueue

if TYPE_CHECKING:
    from agent_service.channels.models import InboundEvent
    from agent_service.memory.models import ConversationCompactionJob
else:
    InboundEvent = object
    ConversationCompactionJob = object


@runtime_checkable
class InboundQueue(EventQueue[InboundEvent], Protocol):
    pass


@runtime_checkable
class CompactionQueue(EventQueue[ConversationCompactionJob], Protocol):
    pass
