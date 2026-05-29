from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from agent_service.users.models import UserResolutionStatus


class InboundIntakeStatus(StrEnum):
    PUBLISHED = "published"
    REJECTED = "rejected"


class InboundIntakeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: InboundIntakeStatus
    published: bool
    user_resolution_status: UserResolutionStatus
    reason: str | None = None
