class InboundWorkerError(Exception):
    """Base class for inbound worker errors."""


class UnresolvedInboundEventError(InboundWorkerError):
    """Raised when a worker receives an event without an internal user id."""


class OutboundOverloadedError(InboundWorkerError):
    """Raised when publishing to the outbound queue exceeds the publish timeout.

    Treated as a retryable failure so the conversation lock is released and the
    assistant message is not persisted as delivered when the outbound queue is
    saturated (for example, when delivery workers are wedged).
    """
