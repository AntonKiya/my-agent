class ConversationError(Exception):
    """Base class for conversation domain errors."""


class ConversationOwnershipError(ConversationError):
    """Raised when a conversation lookup would mix data between users."""


class ConversationResolutionError(ConversationError):
    """Raised when an inbound event cannot be mapped to a conversation."""


class UnsupportedConversationChannelError(ConversationResolutionError):
    """Raised when no conversation key rules exist for an inbound channel."""


class ConversationLockError(ConversationError):
    """Base class for conversation lock errors."""


class ConversationLockTimeoutError(ConversationLockError):
    """Raised when a conversation lock cannot be acquired before timeout."""
