CREATE TABLE IF NOT EXISTS feedback (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    source_channel text NOT NULL,
    source_external_user_id text NOT NULL,
    source_external_chat_id text NOT NULL,
    source_thread_id text,
    source_inbound_event_id uuid REFERENCES inbound_event_processing(event_id) ON DELETE RESTRICT,
    request_inbound_event_id uuid REFERENCES inbound_event_processing(event_id) ON DELETE SET NULL,
    text text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    CONSTRAINT feedback_source_inbound_event_id_unique
        UNIQUE (source_inbound_event_id),
    CONSTRAINT feedback_conversation_user_fk
        FOREIGN KEY (conversation_id, user_id)
        REFERENCES conversations(id, user_id) ON DELETE RESTRICT,
    CONSTRAINT feedback_text_not_blank
        CHECK (length(btrim(text)) > 0)
);

CREATE INDEX IF NOT EXISTS feedback_user_created_at_idx
    ON feedback (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS feedback_conversation_created_at_idx
    ON feedback (conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS feedback_source_channel_created_at_idx
    ON feedback (source_channel, created_at DESC);
