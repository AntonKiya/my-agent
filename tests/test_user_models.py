from datetime import UTC
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_service.channels import InboundEvent
from agent_service.users import (
    ChannelIdentity,
    ObservedChannelIdentity,
    User,
    UserResolutionResult,
    UserResolutionStatus,
    UserStatus,
    UserWithIdentity,
)


def test_user_defaults_to_active_with_isolated_metadata() -> None:
    first = User()
    second = User()

    first.metadata["role"] = "admin"

    assert first.status is UserStatus.ACTIVE
    assert first.created_at.tzinfo is UTC
    assert second.metadata == {}


def test_channel_identity_uses_external_user_id_as_stable_identity() -> None:
    user = User()

    identity = ChannelIdentity(
        user_id=user.id,
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        username="handle",
        metadata={"first_name": "Anton"},
    )

    assert identity.user_id == user.id
    assert identity.channel == "telegram"
    assert identity.external_user_id == "67890"
    assert identity.username == "handle"
    assert identity.metadata["first_name"] == "Anton"
    assert identity.last_seen_at.tzinfo is UTC


def test_observed_channel_identity_builds_lookup_without_username() -> None:
    observed = ObservedChannelIdentity(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        username="handle",
        metadata={"first_name": "Anton"},
    )

    lookup = observed.lookup()

    assert lookup.channel == "telegram"
    assert lookup.external_user_id == "67890"
    assert "username" not in lookup.model_dump()
    assert observed.observed_at.tzinfo is UTC


def test_user_with_identity_rejects_mismatched_user_id() -> None:
    user = User()
    identity = ChannelIdentity(
        user_id=uuid4(),
        channel="telegram",
        external_user_id="67890",
    )

    with pytest.raises(ValidationError):
        UserWithIdentity(user=user, identity=identity)


def test_resolved_user_result_requires_user_and_identity() -> None:
    with pytest.raises(ValidationError):
        UserResolutionResult(status=UserResolutionStatus.RESOLVED)


def test_resolved_user_result_requires_event_user_id_to_match_user() -> None:
    user = User()
    identity = ChannelIdentity(
        user_id=user.id,
        channel="telegram",
        external_user_id="67890",
    )
    event = InboundEvent(
        channel="telegram",
        external_user_id="67890",
        external_chat_id="12345",
        idempotency_key="telegram:12345:42",
        user_id=uuid4(),
    )

    with pytest.raises(ValidationError):
        UserResolutionResult(
            status=UserResolutionStatus.RESOLVED,
            user=user,
            identity=identity,
            event=event,
        )


def test_blocked_resolution_result_can_explain_rejection() -> None:
    result = UserResolutionResult(
        status=UserResolutionStatus.BLOCKED,
        reason="user is blocked",
    )

    assert result.status is UserResolutionStatus.BLOCKED
    assert result.user is None
    assert result.identity is None


def test_user_model_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        User.model_validate({"status": "active", "telegram_user_id": "67890"})
