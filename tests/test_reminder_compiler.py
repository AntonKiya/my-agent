from datetime import UTC, datetime, time

from agent_service.reminders import (
    IntervalWindowSchedule,
    OnceSchedule,
    WeeklySchedule,
    next_fire_at_utc,
    preview_next_fire_times,
)


def test_once_schedule_uses_user_timezone() -> None:
    schedule = OnceSchedule(local_datetime=datetime(2026, 6, 10, 18, 0))

    result = next_fire_at_utc(
        schedule,
        timezone="Europe/Moscow",
        after_utc=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
    )

    assert result == datetime(2026, 6, 10, 15, 0, tzinfo=UTC)


def test_weekly_schedule_returns_same_day_future_time() -> None:
    schedule = WeeklySchedule(days_of_week=["TU", "TH", "SA"], times=[time(14, 0)])

    result = next_fire_at_utc(
        schedule,
        timezone="Europe/Moscow",
        after_utc=datetime(2026, 6, 9, 10, 0, tzinfo=UTC),
    )

    assert result == datetime(2026, 6, 9, 11, 0, tzinfo=UTC)


def test_interval_window_skips_elapsed_times_inside_window() -> None:
    schedule = IntervalWindowSchedule(
        interval_minutes=120,
        window_start=time(9, 0),
        window_end=time(18, 0),
    )

    result = next_fire_at_utc(
        schedule,
        timezone="Europe/Moscow",
        after_utc=datetime(2026, 6, 9, 8, 30, tzinfo=UTC),
    )

    assert result == datetime(2026, 6, 9, 10, 0, tzinfo=UTC)


def test_nonexistent_dst_local_time_is_skipped() -> None:
    schedule = OnceSchedule(local_datetime=datetime(2026, 3, 29, 2, 30))

    result = next_fire_at_utc(
        schedule,
        timezone="Europe/Berlin",
        after_utc=datetime(2026, 3, 28, 0, 0, tzinfo=UTC),
    )

    assert result is None


def test_ambiguous_dst_local_time_fires_once_at_first_occurrence() -> None:
    schedule = OnceSchedule(local_datetime=datetime(2026, 10, 25, 2, 30))

    result = next_fire_at_utc(
        schedule,
        timezone="Europe/Berlin",
        after_utc=datetime(2026, 10, 24, 0, 0, tzinfo=UTC),
    )

    assert result == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)


def test_preview_returns_multiple_future_times_without_backfill() -> None:
    schedule = WeeklySchedule(days_of_week=["MO", "WE"], times=[time(10, 0)])

    result = preview_next_fire_times(
        schedule,
        timezone="Europe/Moscow",
        after_utc=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
        limit=3,
    )

    assert result == (
        datetime(2026, 6, 10, 7, 0, tzinfo=UTC),
        datetime(2026, 6, 15, 7, 0, tzinfo=UTC),
        datetime(2026, 6, 17, 7, 0, tzinfo=UTC),
    )
