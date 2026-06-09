from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from agent_service.users.models import UserResolutionStatus


class InboundIntakeStatus(StrEnum):
    PUBLISHED = "published"
    BUFFERED = "buffered"
    REJECTED = "rejected"
    OVERLOADED = "overloaded"
    DUPLICATE = "duplicate"


class InboundIntakeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: InboundIntakeStatus
    published: bool
    user_resolution_status: UserResolutionStatus
    reason: str | None = None
    queue_size: int | None = None
    queue_maxsize: int | None = None
