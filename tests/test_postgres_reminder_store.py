from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from uuid import uuid4

from agent_service.reminders import Reminder, WeeklySchedule
from agent_service.reminders.postgres import (
    COMPLETE_INVALID_REMINDER_SQL,
    INSERT_OUTBOX_SQL,
    UPDATE_REMINDER_AFTER_FIRE_SQL,
    PostgresReminderStore,
    _process_due_reminder,
)


@dataclass(slots=True)
class FakeConnection:
    fetch_results: list[list[Mapping[str, object]]] = field(default_factory=list)
    fetchrow_results: list[Mapping[str, object] | None] = field(default_factory=list)
    fetch_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    fetchrow_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    execute_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        self.fetch_calls.append((query, args))
        if self.fetch_results:
            return self.fetch_results.pop(0)
        return []

    async def fetchrow(self, query: str, *args: object) -> Mapping[str, object] | None:
        self.fetchrow_calls.append((query, args))
        if self.fetchrow_results:
            return self.fetchrow_results.pop(0)
        return {"id": args[0]}

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        return "OK"

    def transaction(self) -> AbstractAsyncContextManager[object]:
        return FakeTransaction()


@dataclass(slots=True)
class FakeTransaction:
    async def __aenter__(self) -> object:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        return None


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


def reminder(*, source_conversation: bool = True) -> Reminder:
    return Reminder(
        user_id=uuid4(),
        channel="telegram",
        external_chat_id="123",
        source_conversation_id=uuid4() if source_conversation else None,
        timezone="Europe/Moscow",
        message="выпить витамины",
        schedule=WeeklySchedule(days_of_week=["WE"], times=[time(10, 0)]),
        next_fire_at_utc=datetime(2026, 6, 10, 7, 0, tzinfo=UTC),
        created_at=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
    )


async def test_process_due_reminder_advances_when_event_already_exists() -> None:
    connection = FakeConnection(fetchrow_results=[None])

    await _process_due_reminder(
        connection,
        reminder=reminder(),
        now=datetime(2026, 6, 10, 7, 1, tzinfo=UTC),
    )

    assert all(call[0] != INSERT_OUTBOX_SQL for call in connection.execute_calls)
    assert any(call[0] == UPDATE_REMINDER_AFTER_FIRE_SQL for call in connection.execute_calls)


async def test_process_due_reminder_without_conversation_does_not_stay_due_forever() -> None:
    connection = FakeConnection()

    await _process_due_reminder(
        connection,
        reminder=reminder(source_conversation=False),
        now=datetime(2026, 6, 10, 7, 1, tzinfo=UTC),
    )

    assert all(call[0] != INSERT_OUTBOX_SQL for call in connection.execute_calls)
    assert any(call[0] == UPDATE_REMINDER_AFTER_FIRE_SQL for call in connection.execute_calls)


async def test_process_due_reminders_completes_invalid_stored_schedule() -> None:
    reminder_id = uuid4()
    connection = FakeConnection(
        fetch_results=[
            [
                {
                    "id": reminder_id,
                    "user_id": uuid4(),
                    "channel": "telegram",
                    "external_chat_id": "123",
                    "status": "active",
                    "timezone": "Invalid/Zone",
                    "message": "bad",
                    "schedule_json": {"type": "weekly", "days_of_week": ["WE"], "times": ["10:00"]},
                    "next_fire_at_utc": datetime(2026, 6, 10, 7, 0, tzinfo=UTC),
                    "created_at": datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
                    "updated_at": datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
                }
            ]
        ]
    )
    store = PostgresReminderStore(FakePool(connection))

    processed = await store.process_due_reminders(
        now=datetime(2026, 6, 10, 7, 1, tzinfo=UTC),
        batch_size=10,
    )

    assert processed == 1
    assert (
        COMPLETE_INVALID_REMINDER_SQL,
        (reminder_id, datetime(2026, 6, 10, 7, 1, tzinfo=UTC)),
    ) in (
        connection.execute_calls
    )
