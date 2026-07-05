CREATE TABLE IF NOT EXISTS tool_result_references (
    selection_id text PRIMARY KEY,
    provider text NOT NULL,
    source_tool_name text NOT NULL,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    item_kind text NOT NULL,
    item_index integer NOT NULL CHECK (item_index >= 0),
    label text,
    display_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
    ref_payload jsonb NOT NULL,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    CONSTRAINT tool_result_references_display_snapshot_check
        CHECK (jsonb_typeof(display_snapshot) = 'object'),
    CONSTRAINT tool_result_references_ref_payload_check
        CHECK (jsonb_typeof(ref_payload) = 'object'),
    CONSTRAINT tool_result_references_conversation_user_fk
        FOREIGN KEY (conversation_id, user_id)
        REFERENCES conversations(id, user_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS tool_result_references_conversation_created_at_idx
    ON tool_result_references (conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS tool_result_references_user_created_at_idx
    ON tool_result_references (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS tool_result_references_expires_at_idx
    ON tool_result_references (expires_at);

CREATE INDEX IF NOT EXISTS tool_result_references_provider_tool_idx
    ON tool_result_references (provider, source_tool_name);
