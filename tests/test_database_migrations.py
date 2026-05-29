from agent_service.database import load_sql_migrations


def test_users_migration_defines_identity_tables_and_constraints() -> None:
    migrations = load_sql_migrations()

    assert [migration.name for migration in migrations] == [
        "0001_users.sql",
        "0002_conversations.sql",
    ]

    sql = migrations[0].sql
    assert "CREATE TABLE IF NOT EXISTS users" in sql
    assert "CREATE TABLE IF NOT EXISTS channel_identities" in sql
    assert "CHECK (status IN ('active', 'blocked', 'pending'))" in sql
    assert "UNIQUE (channel, external_user_id)" in sql
    assert "REFERENCES users(id) ON DELETE RESTRICT" in sql
    assert "metadata jsonb NOT NULL DEFAULT '{}'::jsonb" in sql


def test_conversations_migration_defines_conversation_table_and_constraints() -> None:
    migrations = load_sql_migrations()

    sql = migrations[1].sql
    assert "CREATE TABLE IF NOT EXISTS conversations" in sql
    assert "user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT" in sql
    assert "conversation_key text NOT NULL" in sql
    assert "CHECK (type IN ('private', 'group', 'thread'))" in sql
    assert "CHECK (status IN ('active', 'archived'))" in sql
    assert "UNIQUE (conversation_key)" in sql
    assert "metadata jsonb NOT NULL DEFAULT '{}'::jsonb" in sql


def test_migrations_are_loaded_in_version_order() -> None:
    migrations = load_sql_migrations()

    assert tuple(migration.version for migration in migrations) == ("0001", "0002")
