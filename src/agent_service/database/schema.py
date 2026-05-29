from dataclasses import dataclass
from importlib import resources

MIGRATIONS_PACKAGE = "agent_service.database.migrations"


@dataclass(frozen=True, slots=True)
class SqlMigration:
    version: str
    name: str
    sql: str


def load_sql_migrations() -> tuple[SqlMigration, ...]:
    migration_files = sorted(
        (
            file
            for file in resources.files(MIGRATIONS_PACKAGE).iterdir()
            if file.is_file() and file.name.endswith(".sql")
        ),
        key=lambda file: file.name,
    )

    return tuple(
        SqlMigration(
            version=file.name.split("_", maxsplit=1)[0],
            name=file.name,
            sql=file.read_text(encoding="utf-8"),
        )
        for file in migration_files
    )
