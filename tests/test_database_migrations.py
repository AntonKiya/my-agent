from agent_service.database import load_sql_migrations


def test_users_migration_defines_identity_tables_and_constraints() -> None:
    migrations = load_sql_migrations()

    assert [migration.name for migration in migrations] == ["0001_users.sql"]

    sql = migrations[0].sql
    assert "CREATE TABLE IF NOT EXISTS users" in sql
    assert "CREATE TABLE IF NOT EXISTS channel_identities" in sql
    assert "CHECK (status IN ('active', 'blocked', 'pending'))" in sql
    assert "UNIQUE (channel, external_user_id)" in sql
    assert "REFERENCES users(id) ON DELETE RESTRICT" in sql
    assert "metadata jsonb NOT NULL DEFAULT '{}'::jsonb" in sql


def test_migrations_are_loaded_in_version_order() -> None:
    migrations = load_sql_migrations()

    assert tuple(migration.version for migration in migrations) == ("0001",)
