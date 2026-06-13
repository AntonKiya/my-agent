from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from agent_service.quotas import (
    PostgresQuotaService,
    QuotaConfigurationError,
    QuotaMetric,
    QuotaPeriod,
    QuotaReservationRequest,
)
from agent_service.quotas.postgres import RESERVE_QUOTA_SQL


@dataclass(slots=True)
class FakeConnection:
    fetch_results: list[Mapping[str, object] | None] = field(default_factory=list)
    fetch_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        self.fetch_calls.append((query, args))
        if not self.fetch_results:
            return None
        return self.fetch_results.pop(0)


@dataclass(slots=True)
class FakeAcquire:
    connection: FakeConnection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


@dataclass(slots=True)
class FakePool:
    connection: FakeConnection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


async def test_postgres_quota_service_allows_reservation_with_default_policy() -> None:
    user_id = uuid4()
    requested_at = datetime(2026, 6, 13, 15, 30, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            {
                "limit_count": 100,
                "used_count": 1,
                "allowed": True,
            }
        ]
    )
    service = PostgresQuotaService(FakePool(connection))

    result = await service.reserve(
        QuotaReservationRequest(user_id=user_id, requested_at=requested_at)
    )

    assert result.allowed
    assert result.user_id == user_id
    assert result.metric is QuotaMetric.AGENT_TURN
    assert result.period is QuotaPeriod.DAY
    assert result.period_start == datetime(2026, 6, 13, tzinfo=UTC)
    assert result.period_end == datetime(2026, 6, 14, tzinfo=UTC)
    assert result.used_count == 1
    assert result.limit_count == 100
    assert result.remaining_count == 99
    assert connection.fetch_calls == [
        (
            RESERVE_QUOTA_SQL,
            (
                user_id,
                "agent_turn",
                "day",
                datetime(2026, 6, 13, tzinfo=UTC),
                requested_at,
            ),
        )
    ]


async def test_postgres_quota_service_denies_when_daily_limit_is_exhausted() -> None:
    user_id = uuid4()
    requested_at = datetime(2026, 6, 13, 23, 59, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            {
                "limit_count": 100,
                "used_count": 100,
                "allowed": False,
            }
        ]
    )
    service = PostgresQuotaService(FakePool(connection))

    result = await service.reserve(
        QuotaReservationRequest(user_id=user_id, requested_at=requested_at)
    )

    assert not result.allowed
    assert result.used_count == 100
    assert result.limit_count == 100
    assert result.remaining_count == 0


async def test_postgres_quota_service_uses_user_override_result() -> None:
    user_id = uuid4()
    connection = FakeConnection(
        fetch_results=[
            {
                "limit_count": 1000,
                "used_count": 101,
                "allowed": True,
            }
        ]
    )
    service = PostgresQuotaService(FakePool(connection))

    result = await service.reserve(
        QuotaReservationRequest(
            user_id=user_id,
            requested_at=datetime(2026, 6, 13, tzinfo=UTC),
        )
    )

    assert result.allowed
    assert result.used_count == 101
    assert result.limit_count == 1000
    assert result.remaining_count == 899


async def test_postgres_quota_service_starts_new_counter_on_next_utc_day() -> None:
    user_id = uuid4()
    connection = FakeConnection(
        fetch_results=[
            {
                "limit_count": 100,
                "used_count": 1,
                "allowed": True,
            }
        ]
    )
    service = PostgresQuotaService(FakePool(connection))

    result = await service.reserve(
        QuotaReservationRequest(
            user_id=user_id,
            requested_at=datetime(2026, 6, 14, 0, 0, tzinfo=UTC),
        )
    )

    assert result.period_start == datetime(2026, 6, 14, tzinfo=UTC)
    assert result.period_end == datetime(2026, 6, 15, tzinfo=UTC)
    assert result.used_count == 1
    assert connection.fetch_calls[0][1][3] == datetime(2026, 6, 14, tzinfo=UTC)


async def test_postgres_quota_service_normalizes_requested_at_to_utc_before_sql() -> None:
    user_id = uuid4()
    requested_at = datetime(2026, 6, 14, 2, 30, tzinfo=timezone(timedelta(hours=3)))
    expected_requested_at = datetime(2026, 6, 13, 23, 30, tzinfo=UTC)
    connection = FakeConnection(
        fetch_results=[
            {
                "limit_count": 100,
                "used_count": 10,
                "allowed": True,
            }
        ]
    )
    service = PostgresQuotaService(FakePool(connection))

    result = await service.reserve(
        QuotaReservationRequest(user_id=user_id, requested_at=requested_at)
    )

    assert result.period_start == datetime(2026, 6, 13, tzinfo=UTC)
    assert connection.fetch_calls[0][1][4] == expected_requested_at


async def test_postgres_quota_service_treats_naive_requested_at_as_utc() -> None:
    user_id = uuid4()
    connection = FakeConnection(
        fetch_results=[
            {
                "limit_count": 100,
                "used_count": 10,
                "allowed": True,
            }
        ]
    )
    service = PostgresQuotaService(FakePool(connection))

    await service.reserve(
        QuotaReservationRequest(
            user_id=user_id,
            requested_at=datetime(2026, 6, 13, 12, 0),
        )
    )

    assert connection.fetch_calls[0][1][4] == datetime(2026, 6, 13, 12, 0, tzinfo=UTC)


async def test_postgres_quota_service_rejects_missing_policy() -> None:
    connection = FakeConnection(
        fetch_results=[
            {
                "limit_count": None,
                "used_count": 0,
                "allowed": False,
            }
        ]
    )
    service = PostgresQuotaService(FakePool(connection))

    with pytest.raises(QuotaConfigurationError):
        await service.reserve(QuotaReservationRequest(user_id=uuid4()))
