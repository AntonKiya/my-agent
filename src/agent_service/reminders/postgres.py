import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from pydantic import TypeAdapter, ValidationError

from agent_service.reminders.compiler import ReminderScheduleError, next_fire_at_utc
from agent_service.reminders.interfaces import NotificationOutboxStore, ReminderStore
from agent_service.reminders.models import (
    NotificationOutboxItem,
    NotificationOutboxStatus,
    OnceSchedule,
    Reminder,
    ReminderEventStatus,
    ReminderSchedule,
    ReminderStatus,
)

SCHEDULE_ADAPTER: TypeAdapter[ReminderSchedule] = TypeAdapter(ReminderSchedule)

INSERT_REMINDER_SQL = """
INSERT INTO reminders (
    id,
    group_id,
    user_id,
    channel,
    external_chat_id,
    thread_id,
    source_conversation_id,
    source_inbound_event_id,
    status,
    timezone,
    message,
    schedule_json,
    source_text,
    assumptions_json,
    next_fire_at_utc,
    last_fire_at_utc,
    created_at,
    updated_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13, $14::jsonb,
    $15, $16, $17, $18
)
"""

COUNT_ACTIVE_REMINDERS_SQL = """
SELECT count(*) AS reminder_count
FROM reminders
WHERE user_id = $1
  AND status = 'active'
"""

COUNT_SOURCE_EVENT_REMINDERS_SQL = """
SELECT count(*) AS reminder_count
FROM reminders
WHERE source_inbound_event_id = $1
"""

LIST_USER_REMINDERS_SQL = """
SELECT *
FROM reminders
WHERE user_id = $1
  AND status = ANY($2::text[])
ORDER BY created_at DESC
LIMIT $3
"""

MARK_REMINDER_DELETED_SQL = """
UPDATE reminders
SET status = 'deleted',
    updated_at = $3
WHERE id = $1
  AND user_id = $2
  AND status != 'deleted'
"""

DUE_REMINDERS_SQL = """
SELECT *
FROM reminders
WHERE status = 'active'
  AND next_fire_at_utc IS NOT NULL
  AND next_fire_at_utc <= $1
ORDER BY next_fire_at_utc
LIMIT $2
FOR UPDATE SKIP LOCKED
"""

INSERT_REMINDER_EVENT_SQL = """
INSERT INTO reminder_events (
    id,
    reminder_id,
    scheduled_for_utc,
    status,
    idempotency_key,
    outbound_event_id,
    created_at,
    sent_at,
    error
) VALUES ($1, $2, $3, $4, $5, $6, $7, NULL, $8)
ON CONFLICT (reminder_id, scheduled_for_utc) DO NOTHING
RETURNING id
"""

INSERT_OUTBOX_SQL = """
INSERT INTO notification_outbox (
    id,
    user_id,
    source_conversation_id,
    reminder_event_id,
    channel,
    external_chat_id,
    thread_id,
    payload_json,
    status,
    retry_count,
    available_at,
    locked_until,
    locked_by,
    created_at,
    updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, 'pending', 0, $9, NULL, NULL, $10, $11)
"""

UPDATE_REMINDER_AFTER_FIRE_SQL = """
UPDATE reminders
SET status = $2,
    next_fire_at_utc = $3,
    last_fire_at_utc = $4,
    updated_at = $5
WHERE id = $1
"""

COMPLETE_INVALID_REMINDER_SQL = """
UPDATE reminders
SET status = 'completed',
    next_fire_at_utc = NULL,
    updated_at = $2
WHERE id = $1
"""

CLAIM_OUTBOX_SQL = """
UPDATE notification_outbox
SET status = 'sending',
    locked_by = $1,
    locked_until = $2,
    updated_at = $3
WHERE id IN (
    SELECT id
    FROM notification_outbox
    WHERE (
        status = 'pending'
        OR (status = 'sending' AND locked_until IS NOT NULL AND locked_until <= $3)
    )
      AND available_at <= $3
      AND (locked_until IS NULL OR locked_until <= $3)
    ORDER BY available_at ASC
    LIMIT $4
    FOR UPDATE SKIP LOCKED
)
RETURNING *
"""

MARK_OUTBOX_SENT_SQL = """
UPDATE notification_outbox
SET status = 'sent',
    locked_by = NULL,
    locked_until = NULL,
    updated_at = $2
WHERE id = $1
"""

MARK_OUTBOX_RETRY_SQL = """
UPDATE notification_outbox
SET status = 'pending',
    retry_count = retry_count + 1,
    available_at = $2,
    locked_by = NULL,
    locked_until = NULL,
    updated_at = $3
WHERE id = $1
"""

MARK_OUTBOX_DEAD_LETTER_SQL = """
UPDATE notification_outbox
SET status = 'dead_letter',
    locked_by = NULL,
    locked_until = NULL,
    updated_at = $2
WHERE id = $1
"""

UPDATE_REMINDER_EVENT_STATUS_SQL = """
UPDATE reminder_events
SET status = $2,
    sent_at = COALESCE($3, sent_at),
    error = COALESCE($4, error)
WHERE id = $1
"""


class PostgresConnection(Protocol):
    async def fetch(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        """Fetch rows from Postgres."""
        ...

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        """Fetch one row from Postgres."""
        ...

    async def execute(self, query: str, *args: object) -> str:
        """Execute a SQL command against Postgres."""
        ...

    def transaction(self) -> AbstractAsyncContextManager[object]:
        """Open a transaction on this connection."""
        ...


class PostgresPool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire a Postgres connection from a pool."""
        ...


class PostgresReminderStore(ReminderStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create(self, reminder: Reminder) -> Reminder:
        async with self._pool.acquire() as connection:
            await connection.execute(
                INSERT_REMINDER_SQL,
                reminder.id,
                reminder.group_id,
                reminder.user_id,
                reminder.channel,
                reminder.external_chat_id,
                reminder.thread_id,
                reminder.source_conversation_id,
                reminder.source_inbound_event_id,
                reminder.status.value,
                reminder.timezone,
                reminder.message,
                _jsonb(_schedule_json(reminder.schedule)),
                reminder.source_text,
                _jsonb(reminder.assumptions),
                reminder.next_fire_at_utc,
                reminder.last_fire_at_utc,
                reminder.created_at,
                reminder.updated_at,
            )
        return reminder

    async def list_for_user(
        self,
        *,
        user_id: UUID,
        statuses: Sequence[ReminderStatus] | None = None,
        limit: int = 20,
    ) -> tuple[Reminder, ...]:
        requested_statuses = statuses or (ReminderStatus.ACTIVE, ReminderStatus.PAUSED)
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                LIST_USER_REMINDERS_SQL,
                user_id,
                [status.value for status in requested_statuses],
                limit,
            )
        return tuple(_reminder_from_row(row) for row in rows)

    async def mark_deleted(self, *, reminder_id: UUID, user_id: UUID) -> bool:
        async with self._pool.acquire() as connection:
            result = await connection.execute(
                MARK_REMINDER_DELETED_SQL,
                reminder_id,
                user_id,
                datetime.now(UTC),
            )
        return _affected_rows(result) > 0

    async def count_active_for_user(self, *, user_id: UUID) -> int:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(COUNT_ACTIVE_REMINDERS_SQL, user_id)
        return _row_count(row)

    async def count_created_for_source_event(self, *, source_inbound_event_id: UUID) -> int:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                COUNT_SOURCE_EVENT_REMINDERS_SQL,
                source_inbound_event_id,
            )
        return _row_count(row)

    async def process_due_reminders(
        self,
        *,
        now: datetime,
        batch_size: int,
    ) -> int:
        now_utc = _aware_utc(now)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                rows = await connection.fetch(DUE_REMINDERS_SQL, now_utc, batch_size)
                for row in rows:
                    try:
                        reminder = _reminder_from_row(row)
                        await _process_due_reminder(connection, reminder=reminder, now=now_utc)
                    except (
                        ReminderScheduleError,
                        TypeError,
                        ValueError,
                        ValidationError,
                    ):
                        await connection.execute(
                            COMPLETE_INVALID_REMINDER_SQL,
                            _uuid(row["id"]),
                            now_utc,
                        )
        return len(rows)


class PostgresNotificationOutboxStore(NotificationOutboxStore):
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def claim_due(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
        limit: int,
    ) -> tuple[NotificationOutboxItem, ...]:
        now_utc = _aware_utc(now)
        lease_until = now_utc + timedelta(seconds=lease_seconds)
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                CLAIM_OUTBOX_SQL,
                worker_id,
                lease_until,
                now_utc,
                limit,
            )
        return tuple(_outbox_from_row(row) for row in rows)

    async def mark_sent(
        self,
        *,
        item_id: UUID,
        reminder_event_id: UUID | None,
        sent_at: datetime,
    ) -> None:
        sent_at_utc = _aware_utc(sent_at)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(MARK_OUTBOX_SENT_SQL, item_id, sent_at_utc)
                if reminder_event_id is not None:
                    await connection.execute(
                        UPDATE_REMINDER_EVENT_STATUS_SQL,
                        reminder_event_id,
                        ReminderEventStatus.SENT.value,
                        sent_at_utc,
                        None,
                    )

    async def mark_retry(
        self,
        *,
        item_id: UUID,
        available_at: datetime,
        error: str | None = None,
    ) -> None:
        del error
        async with self._pool.acquire() as connection:
            await connection.execute(
                MARK_OUTBOX_RETRY_SQL,
                item_id,
                _aware_utc(available_at),
                datetime.now(UTC),
            )

    async def mark_dead_letter(
        self,
        *,
        item_id: UUID,
        reminder_event_id: UUID | None,
        error: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(MARK_OUTBOX_DEAD_LETTER_SQL, item_id, now)
                if reminder_event_id is not None:
                    await connection.execute(
                        UPDATE_REMINDER_EVENT_STATUS_SQL,
                        reminder_event_id,
                        ReminderEventStatus.FAILED.value,
                        None,
                        error,
                    )

    async def mark_reminder_event_status(
        self,
        *,
        reminder_event_id: UUID,
        status: ReminderEventStatus,
        error: str | None = None,
        sent_at: datetime | None = None,
    ) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(
                UPDATE_REMINDER_EVENT_STATUS_SQL,
                reminder_event_id,
                status.value,
                _aware_utc(sent_at) if sent_at is not None else None,
                error,
            )


async def _process_due_reminder(
    connection: PostgresConnection,
    *,
    reminder: Reminder,
    now: datetime,
) -> None:
    scheduled_for = reminder.next_fire_at_utc
    if scheduled_for is None:
        return

    status, error = _event_status_for_misfire(
        reminder=reminder,
        scheduled_for=scheduled_for,
        now=now,
    )
    event_id = uuid4()
    outbound_event_id = uuid4() if status is ReminderEventStatus.QUEUED else None
    idempotency_key = f"{reminder.id}:{scheduled_for.isoformat()}"
    inserted_event = await connection.fetchrow(
        INSERT_REMINDER_EVENT_SQL,
        event_id,
        reminder.id,
        scheduled_for,
        status.value,
        idempotency_key,
        outbound_event_id,
        now,
        error,
    )
    event_created = inserted_event is not None
    if (
        event_created
        and status is ReminderEventStatus.QUEUED
        and outbound_event_id is not None
    ):
        if reminder.source_conversation_id is None:
            await connection.execute(
                UPDATE_REMINDER_EVENT_STATUS_SQL,
                event_id,
                ReminderEventStatus.FAILED.value,
                None,
                "reminder has no source_conversation_id for delivery",
            )
        else:
            await connection.execute(
                INSERT_OUTBOX_SQL,
                outbound_event_id,
                reminder.user_id,
                reminder.source_conversation_id,
                event_id,
                reminder.channel,
                reminder.external_chat_id,
                reminder.thread_id,
                _jsonb(_outbound_payload(reminder)),
                now,
                now,
                now,
            )

    next_fire = _next_after_due(reminder=reminder, now=now)
    next_status = ReminderStatus.COMPLETED if next_fire is None else ReminderStatus.ACTIVE
    await connection.execute(
        UPDATE_REMINDER_AFTER_FIRE_SQL,
        reminder.id,
        next_status.value,
        next_fire,
        scheduled_for,
        now,
    )


def _event_status_for_misfire(
    *,
    reminder: Reminder,
    scheduled_for: datetime,
    now: datetime,
) -> tuple[ReminderEventStatus, str | None]:
    lateness = now - scheduled_for
    if lateness <= timedelta(0):
        return ReminderEventStatus.QUEUED, None
    if isinstance(reminder.schedule, OnceSchedule):
        if lateness <= timedelta(hours=24):
            return ReminderEventStatus.QUEUED, None
        return ReminderEventStatus.EXPIRED, "one-shot reminder expired after 24h misfire grace"
    if reminder.schedule.type == "interval_window":
        if lateness <= timedelta(minutes=30):
            return ReminderEventStatus.QUEUED, None
        return ReminderEventStatus.SKIPPED, "interval reminder skipped after 30m misfire grace"
    if lateness <= timedelta(hours=2):
        return ReminderEventStatus.QUEUED, None
    return ReminderEventStatus.SKIPPED, "recurring reminder coalesced after 2h misfire grace"


def _next_after_due(*, reminder: Reminder, now: datetime) -> datetime | None:
    if isinstance(reminder.schedule, OnceSchedule):
        return None
    return next_fire_at_utc(reminder.schedule, timezone=reminder.timezone, after_utc=now)


def _outbound_payload(reminder: Reminder) -> dict[str, object]:
    return {
        "text": f"Напоминание: {reminder.message}",
        "metadata": {
            "reminder_id": str(reminder.id),
            "source": "reminder",
        },
    }


def _reminder_from_row(row: Mapping[str, object]) -> Reminder:
    return Reminder(
        id=_uuid(row["id"]),
        group_id=_optional_uuid(row.get("group_id")),
        user_id=_uuid(row["user_id"]),
        channel=str(row["channel"]),
        external_chat_id=str(row["external_chat_id"]),
        thread_id=_optional_str(row.get("thread_id")),
        source_conversation_id=_optional_uuid(row.get("source_conversation_id")),
        source_inbound_event_id=_optional_uuid(row.get("source_inbound_event_id")),
        status=ReminderStatus(str(row["status"])),
        timezone=str(row["timezone"]),
        message=str(row["message"]),
        schedule=_schedule_from_json(row["schedule_json"]),
        source_text=_optional_str(row.get("source_text")),
        assumptions=_list_str(row.get("assumptions_json")),
        next_fire_at_utc=_optional_datetime(row.get("next_fire_at_utc")),
        last_fire_at_utc=_optional_datetime(row.get("last_fire_at_utc")),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _outbox_from_row(row: Mapping[str, object]) -> NotificationOutboxItem:
    return NotificationOutboxItem(
        id=_uuid(row["id"]),
        user_id=_uuid(row["user_id"]),
        source_conversation_id=_uuid(row["source_conversation_id"]),
        reminder_event_id=_optional_uuid(row.get("reminder_event_id")),
        channel=str(row["channel"]),
        external_chat_id=str(row["external_chat_id"]),
        thread_id=_optional_str(row.get("thread_id")),
        payload=_dict_json(row["payload_json"]),
        status=NotificationOutboxStatus(str(row["status"])),
        retry_count=_int(row["retry_count"]),
        available_at=_datetime(row["available_at"]),
        locked_until=_optional_datetime(row.get("locked_until")),
        locked_by=_optional_str(row.get("locked_by")),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _schedule_json(schedule: ReminderSchedule) -> dict[str, object]:
    return cast(dict[str, object], SCHEDULE_ADAPTER.dump_python(schedule, mode="json"))


def _schedule_from_json(value: object) -> ReminderSchedule:
    return SCHEDULE_ADAPTER.validate_python(_dict_json(value))


def _dict_json(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise TypeError("expected JSON object")
    return parsed


def _list_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


def _row_count(row: Mapping[str, object] | None) -> int:
    if row is None:
        return 0
    value = row.get("reminder_count")
    if isinstance(value, int):
        return value
    return int(str(value))


def _jsonb(value: object) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _optional_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    return _uuid(value)


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("expected datetime")
    return _aware_utc(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    return int(str(value))


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _affected_rows(result: str) -> int:
    try:
        return int(result.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return 0
