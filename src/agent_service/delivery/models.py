from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

DeliveryMetadata = dict[str, Any]
DeliveryChannelName = str


class DeliveryStatus(StrEnum):
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    FAILED_RETRYABLE = "failed_retryable"
    DEAD_LETTER = "dead_letter"


TERMINAL_DELIVERY_STATUSES = frozenset(
    {
        DeliveryStatus.SENT,
        DeliveryStatus.DEAD_LETTER,
    }
)
RETRYABLE_DELIVERY_STATUSES = frozenset({DeliveryStatus.FAILED_RETRYABLE})
ACTIVE_DELIVERY_STATUSES = frozenset(
    {
        DeliveryStatus.QUEUED,
        DeliveryStatus.SENDING,
    }
)
RESULT_DELIVERY_STATUSES = frozenset(
    {
        DeliveryStatus.SENT,
        DeliveryStatus.FAILED_RETRYABLE,
        DeliveryStatus.DEAD_LETTER,
    }
)


class DeliveryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeliveryResult(DeliveryModel):
    event_id: UUID
    channel: DeliveryChannelName = Field(min_length=1)
    status: DeliveryStatus
    external_message_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = Field(default=None, gt=0)
    metadata: DeliveryMetadata = Field(default_factory=dict)
    delivered_at: datetime | None = None

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_DELIVERY_STATUSES

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_DELIVERY_STATUSES

    @model_validator(mode="after")
    def result_status_must_be_adapter_outcome(self) -> "DeliveryResult":
        if self.status not in RESULT_DELIVERY_STATUSES:
            raise ValueError(
                "Delivery result status must be sent, failed_retryable, or dead_letter"
            )
        return self
