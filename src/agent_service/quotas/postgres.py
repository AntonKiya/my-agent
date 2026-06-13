from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from agent_service.quotas.interfaces import QuotaService
from agent_service.quotas.models import (
    QuotaReservationRequest,
    QuotaReservationResult,
    quota_period_bounds,
    quota_timestamp_utc,
)

RESERVE_QUOTA_SQL = """
WITH effective_limit AS (
    SELECT COALESCE(
        (
            SELECT limit_count
            FROM user_quota_overrides
            WHERE user_id = $1
              AND metric = $2
              AND period = $3
              AND enabled
        ),
        (
            SELECT limit_count
            FROM usage_quota_policies
            WHERE metric = $2
              AND period = $3
              AND enabled
            ORDER BY created_at ASC
            LIMIT 1
        )
    ) AS limit_count
),
inserted AS (
    INSERT INTO usage_quota_counters (
        user_id,
        metric,
        period,
        period_start,
        used_count,
        created_at,
        updated_at
    )
    SELECT $1, $2, $3, $4, 1, $5, $5
    FROM effective_limit
    WHERE limit_count IS NOT NULL
      AND limit_count > 0
    ON CONFLICT (user_id, metric, period, period_start) DO NOTHING
    RETURNING used_count
),
updated AS (
    UPDATE usage_quota_counters
    SET used_count = usage_quota_counters.used_count + 1,
        updated_at = $5
    FROM effective_limit
    WHERE usage_quota_counters.user_id = $1
      AND usage_quota_counters.metric = $2
      AND usage_quota_counters.period = $3
      AND usage_quota_counters.period_start = $4
      AND NOT EXISTS (SELECT 1 FROM inserted)
      AND effective_limit.limit_count IS NOT NULL
      AND usage_quota_counters.used_count < effective_limit.limit_count
    RETURNING usage_quota_counters.used_count
),
current_counter AS (
    SELECT used_count
    FROM usage_quota_counters
    WHERE user_id = $1
      AND metric = $2
      AND period = $3
      AND period_start = $4
)
SELECT
    effective_limit.limit_count,
    COALESCE(
        (SELECT used_count FROM inserted),
        (SELECT used_count FROM updated),
        (SELECT used_count FROM current_counter),
        0
    ) AS used_count,
    (
        EXISTS (SELECT 1 FROM inserted)
        OR EXISTS (SELECT 1 FROM updated)
    ) AS allowed
FROM effective_limit
"""


class QuotaConfigurationError(RuntimeError):
    """Raised when an enabled quota metric has no configured policy."""


class PostgresConnection(Protocol):
    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        """Fetch one row from Postgres."""
        ...


class PostgresPool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire a Postgres connection from a pool."""
        ...


class PostgresQuotaService(QuotaService):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def reserve(self, request: QuotaReservationRequest) -> QuotaReservationResult:
        requested_at = quota_timestamp_utc(request.requested_at)
        period_start, period_end = quota_period_bounds(
            period=request.period,
            at=requested_at,
        )
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                RESERVE_QUOTA_SQL,
                request.user_id,
                request.metric.value,
                request.period.value,
                period_start,
                requested_at,
            )
        if row is None:
            raise QuotaConfigurationError(
                f"Quota policy is not configured for {request.metric.value}/{request.period.value}"
            )

        limit_count = row["limit_count"]
        if not isinstance(limit_count, int):
            raise QuotaConfigurationError(
                f"Quota policy is not configured for {request.metric.value}/{request.period.value}"
            )

        used_count = row["used_count"]
        allowed = row["allowed"]
        if not isinstance(used_count, int) or not isinstance(allowed, bool):
            raise TypeError("Quota reservation row has invalid shape")

        return QuotaReservationResult(
            allowed=allowed,
            user_id=request.user_id,
            metric=request.metric,
            period=request.period,
            period_start=period_start,
            period_end=period_end,
            used_count=used_count,
            limit_count=limit_count,
        )
