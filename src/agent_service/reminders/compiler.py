from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_service.reminders.models import (
    IntervalWindowSchedule,
    OnceSchedule,
    ReminderSchedule,
    Weekday,
    WeeklySchedule,
)

WEEKDAY_TO_INDEX: dict[Weekday, int] = {
    "MO": 0,
    "TU": 1,
    "WE": 2,
    "TH": 3,
    "FR": 4,
    "SA": 5,
    "SU": 6,
}
MAX_LOOKAHEAD_DAYS = 366 * 5


class ReminderScheduleError(ValueError):
    """Raised when a reminder schedule cannot be compiled."""


def validate_timezone(timezone: str) -> ZoneInfo:
    clean_timezone = timezone.strip()
    if not clean_timezone:
        raise ReminderScheduleError("timezone is required")
    try:
        return ZoneInfo(clean_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ReminderScheduleError("timezone must be a valid IANA timezone") from exc


def next_fire_at_utc(
    schedule: ReminderSchedule,
    *,
    timezone: str,
    after_utc: datetime,
) -> datetime | None:
    tz = validate_timezone(timezone)
    after = _aware_utc(after_utc)
    if isinstance(schedule, OnceSchedule):
        return _once_next_fire(schedule, tz=tz, after_utc=after)
    if isinstance(schedule, WeeklySchedule):
        return _weekly_next_fire(schedule, tz=tz, after_utc=after)
    if isinstance(schedule, IntervalWindowSchedule):
        return _interval_window_next_fire(schedule, tz=tz, after_utc=after)
    raise ReminderScheduleError(f"unsupported schedule type: {type(schedule).__name__}")


def preview_next_fire_times(
    schedule: ReminderSchedule,
    *,
    timezone: str,
    after_utc: datetime,
    limit: int = 3,
) -> tuple[datetime, ...]:
    if limit < 1:
        return ()
    values: list[datetime] = []
    cursor = _aware_utc(after_utc)
    for _ in range(limit):
        value = next_fire_at_utc(schedule, timezone=timezone, after_utc=cursor)
        if value is None:
            break
        values.append(value)
        cursor = value + timedelta(seconds=1)
    return tuple(values)


def format_local_fire_times(values: tuple[datetime, ...], *, timezone: str) -> list[str]:
    tz = validate_timezone(timezone)
    return [value.astimezone(tz).strftime("%Y-%m-%d %H:%M") for value in values]


def _once_next_fire(
    schedule: OnceSchedule,
    *,
    tz: ZoneInfo,
    after_utc: datetime,
) -> datetime | None:
    value = _local_to_utc(schedule.local_datetime, tz)
    if value is None or value <= after_utc:
        return None
    return value


def _weekly_next_fire(
    schedule: WeeklySchedule,
    *,
    tz: ZoneInfo,
    after_utc: datetime,
) -> datetime | None:
    after_local = after_utc.astimezone(tz)
    start = max(after_local.date(), schedule.start_date or after_local.date())
    day_indexes = {WEEKDAY_TO_INDEX[day] for day in schedule.days_of_week}
    for candidate_date in _date_range(start, schedule.end_date):
        if candidate_date.weekday() not in day_indexes:
            continue
        for candidate_time in schedule.times:
            candidate = _candidate_utc(candidate_date, candidate_time, tz)
            if candidate is not None and candidate > after_utc:
                return candidate
    return None


def _interval_window_next_fire(
    schedule: IntervalWindowSchedule,
    *,
    tz: ZoneInfo,
    after_utc: datetime,
) -> datetime | None:
    after_local = after_utc.astimezone(tz)
    start = max(after_local.date(), schedule.start_date or after_local.date())
    day_indexes = (
        {WEEKDAY_TO_INDEX[day] for day in schedule.days_of_week}
        if schedule.days_of_week is not None
        else None
    )
    for candidate_date in _date_range(start, schedule.end_date):
        if day_indexes is not None and candidate_date.weekday() not in day_indexes:
            continue
        for candidate_time in _window_times(schedule):
            candidate = _candidate_utc(candidate_date, candidate_time, tz)
            if candidate is not None and candidate > after_utc:
                return candidate
    return None


def _window_times(schedule: IntervalWindowSchedule) -> tuple[time, ...]:
    values: list[time] = []
    cursor = datetime.combine(date(2000, 1, 1), schedule.window_start)
    end = datetime.combine(date(2000, 1, 1), schedule.window_end)
    step = timedelta(minutes=schedule.interval_minutes)
    while cursor <= end:
        values.append(cursor.time())
        cursor += step
    return tuple(values)


def _date_range(start: date, end: date | None) -> tuple[date, ...]:
    max_end = start + timedelta(days=MAX_LOOKAHEAD_DAYS)
    effective_end = min(end, max_end) if end is not None else max_end
    if effective_end < start:
        return ()
    days = (effective_end - start).days
    return tuple(start + timedelta(days=offset) for offset in range(days + 1))


def _candidate_utc(candidate_date: date, candidate_time: time, tz: ZoneInfo) -> datetime | None:
    return _local_to_utc(datetime.combine(candidate_date, candidate_time), tz)


def _local_to_utc(local_datetime: datetime, tz: ZoneInfo) -> datetime | None:
    if local_datetime.tzinfo is not None:
        raise ReminderScheduleError("local datetimes must not include timezone info")
    # Try both folds. If both are valid during fall-back, use the first occurrence.
    for fold in (0, 1):
        aware = local_datetime.replace(tzinfo=tz, fold=fold)
        utc_value = aware.astimezone(UTC)
        roundtrip = utc_value.astimezone(tz).replace(tzinfo=None)
        if roundtrip == local_datetime:
            return utc_value
    return None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
