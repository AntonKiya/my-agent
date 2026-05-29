CREATE TABLE IF NOT EXISTS conversations (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    channel text NOT NULL,
    conversation_key text NOT NULL,
    external_chat_id text NOT NULL,
    type text NOT NULL CHECK (type IN ('private', 'group', 'thread')),
    thread_id text,
    status text NOT NULL CHECK (status IN ('active', 'archived')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CONSTRAINT conversations_conversation_key_unique
        UNIQUE (conversation_key)
);

CREATE INDEX IF NOT EXISTS conversations_user_id_idx
    ON conversations (user_id);

CREATE INDEX IF NOT EXISTS conversations_channel_external_chat_id_idx
    ON conversations (channel, external_chat_id);
