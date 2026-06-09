from agent_service.database import load_sql_migrations


def test_users_migration_defines_identity_tables_and_constraints() -> None:
    migrations = load_sql_migrations()

    assert [migration.name for migration in migrations] == [
        "0001_users.sql",
        "0002_conversations.sql",
        "0003_conversation_messages.sql",
        "0004_conversation_summaries.sql",
        "0005_inbound_idempotency.sql",
        "0006_drop_conversation_messages_token_count.sql",
        "0007_media_assets.sql",
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

    assert tuple(migration.version for migration in migrations) == (
        "0001",
        "0002",
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
    )


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
    assert "token_count integer CHECK (token_count IS NULL OR token_count >= 0)" in sql
    assert "UNIQUE (conversation_id, sequence)" in sql
    assert "conversation_messages_conversation_sequence_idx" in sql
    assert "conversation_messages_tool_call_id_idx" in sql


def test_conversation_summaries_migration_defines_compaction_state_table() -> None:
    migrations = load_sql_migrations()

    sql = migrations[3].sql
    assert "conversations_id_user_id_unique" in sql
    assert "CREATE TABLE IF NOT EXISTS conversation_summaries" in sql
    assert "conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT" in sql
    assert "user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT" in sql
    assert "from_sequence bigint NOT NULL CHECK (from_sequence > 0)" in sql
    assert "to_sequence bigint NOT NULL CHECK (to_sequence >= from_sequence)" in sql
    assert "compacted_message_ids jsonb NOT NULL DEFAULT '[]'::jsonb" in sql
    assert (
        "last_compacted_message_id uuid REFERENCES conversation_messages(id) ON DELETE RESTRICT"
        in sql
    )
    assert (
        "status text NOT NULL CHECK (status IN ('completed', 'failed_retryable', 'dead_letter'))"
        in sql
    )
    assert "FOREIGN KEY (conversation_id, user_id)" in sql
    assert "REFERENCES conversations(id, user_id)" in sql
    assert "conversation_summaries_completed_to_sequence_unique" in sql
    assert "WHERE status = 'completed'" in sql
    assert "conversation_summaries_latest_completed_idx" in sql


def test_inbound_idempotency_migration_defines_processing_gate() -> None:
    migrations = load_sql_migrations()

    sql = migrations[4].sql
    assert "CREATE TABLE IF NOT EXISTS inbound_event_processing" in sql
    assert "event_id uuid PRIMARY KEY" in sql
    assert "idempotency_key text NOT NULL" in sql
    assert "external_update_id text" in sql
    assert "user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT" in sql
    assert "UNIQUE (idempotency_key)" in sql
    assert "inbound_event_processing_channel_update_unique" in sql
    assert "WHERE external_update_id IS NOT NULL" in sql
    assert (
        "status IN (\n            'queued',\n            'processing',\n            'completed',"
        in sql
    )


def test_drop_conversation_message_token_count_migration_removes_unused_column() -> None:
    migrations = load_sql_migrations()

    sql = migrations[5].sql
    assert "ALTER TABLE conversation_messages" in sql
    assert "DROP COLUMN IF EXISTS token_count" in sql


def test_media_assets_migration_defines_general_media_index() -> None:
    migrations = load_sql_migrations()

    sql = migrations[6].sql
    assert "CREATE TABLE IF NOT EXISTS media_assets" in sql
    assert "media_id text PRIMARY KEY" in sql
    assert "user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT" in sql
    assert "conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT" in sql
    assert "media_type text NOT NULL CHECK" in sql
    assert "'image', 'audio', 'document', 'video', 'other'" in sql
    assert "storage_key text NOT NULL" in sql
    assert "media_assets_conversation_user_fk" in sql
    assert "media_assets_user_conversation_created_at_idx" in sql
