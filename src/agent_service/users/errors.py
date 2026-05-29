class UserError(Exception):
    """Base exception for user-domain failures."""


class UserResolutionError(UserError):
    """Raised when a user cannot be resolved safely."""
