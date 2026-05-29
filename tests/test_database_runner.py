from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from agent_service.database import MigrationError, apply_migrations
from agent_service.database.runner import MIGRATION_LOCK_ID
from agent_service.database.schema import SqlMigration


@dataclass(slots=True)
class FakeTransaction:
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> "FakeTransaction":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self.exited = True


@dataclass(slots=True)
class FakeMigrationConnection:
    applied_rows: list[Mapping[str, object]] = field(default_factory=list)
    execute_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    fetch_calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    transactions: list[FakeTransaction] = field(default_factory=list)

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        return "OK"

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
        self.fetch_calls.append((query, args))
        return list(self.applied_rows)

    def transaction(self) -> FakeTransaction:
        transaction = FakeTransaction()
        self.transactions.append(transaction)
        return transaction

    async def close(self) -> None:
        return None


def migration(
    *,
    version: str = "0001",
    name: str = "0001_users.sql",
    sql: str = "CREATE TABLE users (id uuid PRIMARY KEY)",
) -> SqlMigration:
    return SqlMigration(version=version, name=name, sql=sql)


async def test_apply_migrations_uses_advisory_lock_and_records_applied_migration() -> None:
    connection = FakeMigrationConnection()

    result = await apply_migrations(connection=connection, migrations=(migration(),))

    assert result.applied == ("0001_users.sql",)
    assert result.skipped == ()
    assert connection.transactions[0].entered
    assert connection.transactions[0].exited
    assert connection.execute_calls[0] == (
        "SELECT pg_advisory_xact_lock($1)",
        (MIGRATION_LOCK_ID,),
    )
    assert any("CREATE TABLE users" in call[0] for call in connection.execute_calls)
    assert connection.execute_calls[-1][1][0:2] == ("0001", "0001_users.sql")


async def test_apply_migrations_skips_already_applied_matching_migration() -> None:
    first = migration()
    connection = FakeMigrationConnection()
    first_result = await apply_migrations(connection=connection, migrations=(first,))
    applied_checksum = connection.execute_calls[-1][1][2]

    second_connection = FakeMigrationConnection(
        applied_rows=[
            {
                "version": "0001",
                "name": "0001_users.sql",
                "checksum": applied_checksum,
            }
        ]
    )

    second_result = await apply_migrations(connection=second_connection, migrations=(first,))

    assert first_result.applied == ("0001_users.sql",)
    assert second_result.applied == ()
    assert second_result.skipped == ("0001_users.sql",)
    assert all("CREATE TABLE users" not in call[0] for call in second_connection.execute_calls)


async def test_apply_migrations_rejects_checksum_drift_for_applied_version() -> None:
    connection = FakeMigrationConnection(
        applied_rows=[
            {
                "version": "0001",
                "name": "0001_users.sql",
                "checksum": "different",
            }
        ]
    )

    with pytest.raises(MigrationError):
        await apply_migrations(connection=connection, migrations=(migration(),))
