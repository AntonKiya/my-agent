CREATE TABLE IF NOT EXISTS media_assets (
    media_id text PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    media_type text NOT NULL CHECK (
        media_type IN ('image', 'audio', 'document', 'video', 'other')
    ),
    storage_key text NOT NULL,
    content_type text,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    source_channel text NOT NULL,
    source_attachment_id text,
    source_inbound_event_id uuid,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    CONSTRAINT media_assets_conversation_user_fk
        FOREIGN KEY (conversation_id, user_id)
        REFERENCES conversations(id, user_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS media_assets_user_conversation_created_at_idx
    ON media_assets (user_id, conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS media_assets_user_media_type_created_at_idx
    ON media_assets (user_id, media_type, created_at DESC);

CREATE INDEX IF NOT EXISTS media_assets_source_inbound_event_id_idx
    ON media_assets (source_inbound_event_id)
    WHERE source_inbound_event_id IS NOT NULL;
