from typing import Any

from agent_service.channels.models import InboundEvent
from agent_service.users.errors import UserResolutionError
from agent_service.users.interfaces import UserStore
from agent_service.users.models import (
    ObservedChannelIdentity,
    UserMetadata,
    UserResolutionResult,
    UserResolutionStatus,
    UserStatus,
)

USERNAME_METADATA_KEY = "username"


class UserResolver:
    def __init__(self, store: UserStore) -> None:
        self._store = store

    async def resolve(self, event: InboundEvent) -> UserResolutionResult:
        observed_identity = observed_channel_identity_from_event(event)
        user_with_identity = await self._store.get_or_create_active_user_with_identity(
            identity=observed_identity,
        )

        if event.user_id is not None and event.user_id != user_with_identity.user.id:
            raise UserResolutionError("Inbound event already has a different user_id")

        match user_with_identity.user.status:
            case UserStatus.ACTIVE:
                resolved_event = event.model_copy(
                    update={"user_id": user_with_identity.user.id},
                )
                return UserResolutionResult(
                    status=UserResolutionStatus.RESOLVED,
                    user=user_with_identity.user,
                    identity=user_with_identity.identity,
                    event=resolved_event,
                )
            case UserStatus.BLOCKED:
                return UserResolutionResult(
                    status=UserResolutionStatus.BLOCKED,
                    user=user_with_identity.user,
                    identity=user_with_identity.identity,
                    reason="user is blocked",
                )
            case UserStatus.PENDING:
                return UserResolutionResult(
                    status=UserResolutionStatus.PENDING,
                    user=user_with_identity.user,
                    identity=user_with_identity.identity,
                    reason="user is pending",
                )
        raise UserResolutionError(f"Unsupported user status: {user_with_identity.user.status}")


def observed_channel_identity_from_event(event: InboundEvent) -> ObservedChannelIdentity:
    username = _optional_str(event.channel_metadata.get(USERNAME_METADATA_KEY))
    return ObservedChannelIdentity(
        channel=event.channel,
        external_user_id=event.external_user_id,
        external_chat_id=event.external_chat_id,
        username=username,
        metadata=_identity_metadata(event.channel_metadata),
        observed_at=event.received_at,
    )


def _identity_metadata(channel_metadata: dict[str, Any]) -> UserMetadata:
    return {
        key: value
        for key, value in channel_metadata.items()
        if key != USERNAME_METADATA_KEY and value is not None
    }


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
