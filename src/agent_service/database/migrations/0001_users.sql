CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('active', 'blocked', 'pending')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_identities (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    channel text NOT NULL,
    external_user_id text NOT NULL,
    external_chat_id text,
    username text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    CONSTRAINT channel_identities_channel_external_user_id_unique
        UNIQUE (channel, external_user_id)
);

CREATE INDEX IF NOT EXISTS channel_identities_user_id_idx
    ON channel_identities (user_id);

CREATE INDEX IF NOT EXISTS channel_identities_last_seen_at_idx
    ON channel_identities (last_seen_at);
