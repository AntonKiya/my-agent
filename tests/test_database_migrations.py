from agent_service.database import load_sql_migrations


def test_users_migration_defines_identity_tables_and_constraints() -> None:
    migrations = load_sql_migrations()

    assert [migration.name for migration in migrations] == [
        "0001_users.sql",
        "0002_conversations.sql",
        "0003_conversation_messages.sql",
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

    assert tuple(migration.version for migration in migrations) == ("0001", "0002", "0003")


def test_conversation_messages_migration_defines_history_table_and_indexes() -> None:
    migrations = load_sql_migrations()

    sql = migrations[2].sql
    assert "ADD COLUMN IF NOT EXISTS message_sequence bigint NOT NULL DEFAULT 0" in sql
    assert "CREATE TABLE IF NOT EXISTS conversation_messages" in sql
    assert "conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT" in sql
    assert "user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT" in sql
    assert "sequence bigint NOT NULL CHECK (sequence > 0)" in sql
    assert (
        "role text NOT NULL CHECK (role IN ('user', 'assistant', 'tool_call', 'tool_result'))"
        in sql
    )
    assert "attachments jsonb NOT NULL DEFAULT '[]'::jsonb" in sql
    assert "UNIQUE (conversation_id, sequence)" in sql
    assert "conversation_messages_conversation_sequence_idx" in sql
    assert "conversation_messages_tool_call_id_idx" in sql
    assert "token_count integer CHECK (token_count IS NULL OR token_count >= 0)" in sql
