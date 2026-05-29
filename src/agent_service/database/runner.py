import asyncio
import hashlib
import os
from collections.abc import Awaitable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol, cast

import asyncpg

from agent_service.config import AppSettings
from agent_service.database.schema import SqlMigration, load_sql_migrations

MIGRATIONS_TABLE = "agent_service_schema_migrations"
MIGRATION_LOCK_ID = 909_202_605_290

CREATE_MIGRATIONS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
    version text PRIMARY KEY,
    name text NOT NULL,
    checksum text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""

SELECT_APPLIED_MIGRATIONS_SQL = f"""
SELECT version, name, checksum
FROM {MIGRATIONS_TABLE}
ORDER BY version
"""

INSERT_APPLIED_MIGRATION_SQL = f"""
INSERT INTO {MIGRATIONS_TABLE} (
    version,
    name,
    checksum
) VALUES ($1, $2, $3)
"""


class MigrationError(Exception):
    """Raised when database migrations cannot be applied safely."""


class MigrationConnection(Protocol):
    async def execute(self, query: str, *args: object) -> str: ...

    async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]: ...

    def transaction(self) -> AbstractAsyncContextManager[object]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MigrationRunResult:
    applied: tuple[str, ...]
    skipped: tuple[str, ...]


async def migrate_database(
    *,
    dsn: str,
    command_timeout_seconds: float = 30.0,
) -> MigrationRunResult:
    connection = await _connect(dsn=dsn, command_timeout_seconds=command_timeout_seconds)
    try:
        return await apply_migrations(
            connection=connection,
            migrations=load_sql_migrations(),
        )
    finally:
        await connection.close()


async def apply_migrations(
    *,
    connection: MigrationConnection,
    migrations: tuple[SqlMigration, ...],
) -> MigrationRunResult:
    async with connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock($1)", MIGRATION_LOCK_ID)
        await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
        applied_rows = await connection.fetch(SELECT_APPLIED_MIGRATIONS_SQL)
        applied_by_version = _applied_migrations_by_version(applied_rows)

        applied: list[str] = []
        skipped: list[str] = []
        for migration in migrations:
            checksum = _migration_checksum(migration)
            existing = applied_by_version.get(migration.version)
            if existing is not None:
                if existing != (migration.name, checksum):
                    raise MigrationError(
                        f"Migration {migration.version} was already applied with different content"
                    )
                skipped.append(migration.name)
                continue

            await connection.execute(migration.sql)
            await connection.execute(
                INSERT_APPLIED_MIGRATION_SQL,
                migration.version,
                migration.name,
                checksum,
            )
            applied.append(migration.name)

    return MigrationRunResult(
        applied=tuple(applied),
        skipped=tuple(skipped),
    )


async def _connect(
    *,
    dsn: str,
    command_timeout_seconds: float,
) -> MigrationConnection:
    connection = await cast(
        Awaitable[object],
        asyncpg.connect(
            dsn=dsn,
            command_timeout=command_timeout_seconds,
        ),
    )
    return cast(MigrationConnection, connection)


def _applied_migrations_by_version(
    rows: list[Mapping[str, object]],
) -> dict[str, tuple[str, str]]:
    return {
        str(row["version"]): (
            str(row["name"]),
            str(row["checksum"]),
        )
        for row in rows
    }


def _migration_checksum(migration: SqlMigration) -> str:
    return hashlib.sha256(migration.sql.encode("utf-8")).hexdigest()


def main() -> None:
    settings = AppSettings()
    dsn = settings.postgres_dsn or os.environ.get("AGENT_SERVICE_POSTGRES_DSN")
    if dsn is None:
        raise SystemExit("AGENT_SERVICE_POSTGRES_DSN is required")

    result = asyncio.run(
        migrate_database(
            dsn=dsn,
            command_timeout_seconds=settings.postgres_command_timeout_seconds,
        )
    )
    if result.applied:
        print(f"Applied migrations: {', '.join(result.applied)}")
    if result.skipped:
        print(f"Already applied migrations: {', '.join(result.skipped)}")
    if not result.applied and not result.skipped:
        print("No migrations found")
