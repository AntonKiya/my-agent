from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from uuid import UUID


class QuotaMetric(StrEnum):
    AGENT_TURN = "agent_turn"


class QuotaPeriod(StrEnum):
    DAY = "day"


@dataclass(frozen=True, slots=True)
class QuotaReservationRequest:
    user_id: UUID
    metric: QuotaMetric = QuotaMetric.AGENT_TURN
    period: QuotaPeriod = QuotaPeriod.DAY
    requested_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class QuotaReservationResult:
    allowed: bool
    user_id: UUID
    metric: QuotaMetric
    period: QuotaPeriod
    period_start: datetime
    period_end: datetime
    used_count: int
    limit_count: int

    @property
    def remaining_count(self) -> int:
        return max(self.limit_count - self.used_count, 0)


def quota_period_bounds(
    *,
    period: QuotaPeriod,
    at: datetime | None = None,
) -> tuple[datetime, datetime]:
    timestamp = quota_timestamp_utc(at)

    if period is QuotaPeriod.DAY:
        period_start = datetime.combine(timestamp.date(), time.min, tzinfo=UTC)
        return period_start, period_start + timedelta(days=1)

    raise ValueError(f"Unsupported quota period: {period}")


def quota_timestamp_utc(at: datetime | None = None) -> datetime:
    timestamp = at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)
