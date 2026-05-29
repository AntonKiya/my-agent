class InboundWorkerError(Exception):
    """Base class for inbound worker errors."""


class UnresolvedInboundEventError(InboundWorkerError):
    """Raised when a worker receives an event without an internal user id."""
