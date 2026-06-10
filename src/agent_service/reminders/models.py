from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_service.channels.models import ChannelName

Weekday = Literal["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
WEEKDAYS: tuple[Weekday, ...] = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")


def utc_now() -> datetime:
    return datetime.now(UTC)


class ReminderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReminderStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    DELETED = "deleted"


class ReminderEventStatus(StrEnum):
    QUEUED = "queued"
    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class NotificationOutboxStatus(StrEnum):
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class OnceSchedule(ReminderModel):
    type: Literal["once"] = "once"
    local_datetime: datetime

    @field_validator("local_datetime")
    @classmethod
    def local_datetime_must_be_naive(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("local_datetime must not include timezone info")
        return value.replace(second=0, microsecond=0)


class WeeklySchedule(ReminderModel):
    type: Literal["weekly"] = "weekly"
    days_of_week: list[Weekday] = Field(min_length=1)
    times: list[time] = Field(min_length=1)
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("times")
    @classmethod
    def weekly_times_must_be_minute_precision(cls, value: list[time]) -> list[time]:
        return _unique_times(value)

    @field_validator("days_of_week")
    @classmethod
    def weekdays_must_be_unique(cls, value: list[Weekday]) -> list[Weekday]:
        seen: set[Weekday] = set()
        unique: list[Weekday] = []
        for day in value:
            if day in seen:
                continue
            seen.add(day)
            unique.append(day)
        return unique

    @model_validator(mode="after")
    def end_date_must_not_precede_start_date(self) -> "WeeklySchedule":
        if self.start_date is not None and self.end_date is not None:
            if self.end_date < self.start_date:
                raise ValueError("end_date must be on or after start_date")
        return self


class IntervalWindowSchedule(ReminderModel):
    type: Literal["interval_window"] = "interval_window"
    interval_minutes: int = Field(gt=0)
    days_of_week: list[Weekday] | None = None
    window_start: time
    window_end: time
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("window_start", "window_end")
    @classmethod
    def window_times_must_be_minute_precision(cls, value: time) -> time:
        return _minute_precision_time(value)

    @field_validator("days_of_week")
    @classmethod
    def optional_weekdays_must_be_unique(
        cls,
        value: list[Weekday] | None,
    ) -> list[Weekday] | None:
        if value is None:
            return None
        seen: set[Weekday] = set()
        unique: list[Weekday] = []
        for day in value:
            if day in seen:
                continue
            seen.add(day)
            unique.append(day)
        return unique

    @model_validator(mode="after")
    def window_and_dates_must_be_valid(self) -> "IntervalWindowSchedule":
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be later than window_start")
        if self.start_date is not None and self.end_date is not None:
            if self.end_date < self.start_date:
                raise ValueError("end_date must be on or after start_date")
        return self


ReminderSchedule = Annotated[
    OnceSchedule | WeeklySchedule | IntervalWindowSchedule,
    Field(discriminator="type"),
]


class Reminder(ReminderModel):
    id: UUID = Field(default_factory=uuid4)
    group_id: UUID | None = None
    user_id: UUID
    channel: ChannelName = Field(min_length=1)
    external_chat_id: str = Field(min_length=1)
    thread_id: str | None = None
    source_conversation_id: UUID | None = None
    source_inbound_event_id: UUID | None = None
    status: ReminderStatus = ReminderStatus.ACTIVE
    timezone: str = Field(min_length=1)
    message: str = Field(min_length=1)
    schedule: ReminderSchedule
    source_text: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    next_fire_at_utc: datetime | None = None
    last_fire_at_utc: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReminderEvent(ReminderModel):
    id: UUID = Field(default_factory=uuid4)
    reminder_id: UUID
    scheduled_for_utc: datetime
    status: ReminderEventStatus
    idempotency_key: str = Field(min_length=1)
    outbound_event_id: UUID | None = None
    created_at: datetime = Field(default_factory=utc_now)
    sent_at: datetime | None = None
    error: str | None = None


class NotificationOutboxItem(ReminderModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    source_conversation_id: UUID
    reminder_event_id: UUID | None = None
    channel: ChannelName = Field(min_length=1)
    external_chat_id: str = Field(min_length=1)
    thread_id: str | None = None
    payload: dict[str, object]
    status: NotificationOutboxStatus = NotificationOutboxStatus.PENDING
    retry_count: int = Field(default=0, ge=0)
    available_at: datetime = Field(default_factory=utc_now)
    locked_until: datetime | None = None
    locked_by: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReminderPreview(ReminderModel):
    next_fire_at_utc: datetime | None
    next_fire_local: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


def _unique_times(values: list[time]) -> list[time]:
    seen: set[time] = set()
    unique: list[time] = []
    for value in values:
        normalized = _minute_precision_time(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return sorted(unique)


def _minute_precision_time(value: time) -> time:
    if value.tzinfo is not None:
        raise ValueError("time values must not include timezone info")
    return value.replace(second=0, microsecond=0)
