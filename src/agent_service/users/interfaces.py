from typing import Protocol, runtime_checkable

from agent_service.users.models import (
    ChannelIdentityLookup,
    ObservedChannelIdentity,
    UserWithIdentity,
)


@runtime_checkable
class UserStore(Protocol):
    async def get_by_channel_identity(
        self,
        *,
        lookup: ChannelIdentityLookup,
    ) -> UserWithIdentity | None:
        """Load a user by the stable transport identity, never by username."""
        ...

    async def get_or_create_active_user_with_identity(
        self,
        *,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        """Load or atomically create an active user for the observed channel identity."""
        ...

    async def update_identity_seen(
        self,
        *,
        user: UserWithIdentity,
        identity: ObservedChannelIdentity,
    ) -> UserWithIdentity:
        """Persist last-seen identity metadata without changing who the user is."""
        ...
