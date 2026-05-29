from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_service.channels.models import ChannelName, InboundEvent

UserMetadata = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(UTC)


class UserStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    PENDING = "pending"


class UserResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    BLOCKED = "blocked"
    PENDING = "pending"


class UserModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class User(UserModel):
    id: UUID = Field(default_factory=uuid4)
    status: UserStatus = UserStatus.ACTIVE
    metadata: UserMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ChannelIdentity(UserModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    channel: ChannelName = Field(min_length=1)
    external_user_id: str = Field(min_length=1)
    external_chat_id: str | None = None
    username: str | None = None
    metadata: UserMetadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)


class ChannelIdentityLookup(UserModel):
    channel: ChannelName = Field(min_length=1)
    external_user_id: str = Field(min_length=1)


class ObservedChannelIdentity(UserModel):
    channel: ChannelName = Field(min_length=1)
    external_user_id: str = Field(min_length=1)
    external_chat_id: str | None = None
    username: str | None = None
    metadata: UserMetadata = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)

    def lookup(self) -> ChannelIdentityLookup:
        return ChannelIdentityLookup(
            channel=self.channel,
            external_user_id=self.external_user_id,
        )


class UserWithIdentity(UserModel):
    user: User
    identity: ChannelIdentity

    @model_validator(mode="after")
    def identity_must_belong_to_user(self) -> Self:
        if self.identity.user_id != self.user.id:
            raise ValueError("Channel identity user_id must match user id")
        return self


class UserResolutionResult(UserModel):
    status: UserResolutionStatus
    user: User | None = None
    identity: ChannelIdentity | None = None
    event: InboundEvent | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def resolved_result_requires_user_and_identity(self) -> Self:
        if self.status is UserResolutionStatus.RESOLVED:
            if self.user is None or self.identity is None or self.event is None:
                raise ValueError("Resolved user result requires user, identity, and event")
            if self.event.user_id != self.user.id:
                raise ValueError("Resolved user event user_id must match user id")
        return self
