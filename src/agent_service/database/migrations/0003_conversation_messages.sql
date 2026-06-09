ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS message_sequence bigint NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS conversation_messages (
    id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    sequence bigint NOT NULL CHECK (sequence > 0),
    role text NOT NULL CHECK (role IN ('user', 'assistant', 'tool_call', 'tool_result')),
    text text,
    attachments jsonb NOT NULL DEFAULT '[]'::jsonb,
    tool_name text,
    tool_call_id text,
    inbound_event_id uuid,
    outbound_event_id uuid,
    trace_id text,
    token_count integer CHECK (token_count IS NULL OR token_count >= 0),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    CONSTRAINT conversation_messages_conversation_sequence_unique
        UNIQUE (conversation_id, sequence),
    CONSTRAINT conversation_messages_content_check
        CHECK (
            text IS NOT NULL
            OR jsonb_array_length(attachments) > 0
            OR role IN ('tool_call', 'tool_result')
        )
);

CREATE INDEX IF NOT EXISTS conversation_messages_conversation_sequence_idx
    ON conversation_messages (conversation_id, sequence);

CREATE INDEX IF NOT EXISTS conversation_messages_user_created_at_idx
    ON conversation_messages (user_id, created_at);

CREATE INDEX IF NOT EXISTS conversation_messages_inbound_event_id_idx
    ON conversation_messages (inbound_event_id)
    WHERE inbound_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS conversation_messages_outbound_event_id_idx
    ON conversation_messages (outbound_event_id)
    WHERE outbound_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS conversation_messages_tool_call_id_idx
    ON conversation_messages (tool_call_id)
    WHERE tool_call_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS conversation_messages_trace_id_idx
    ON conversation_messages (trace_id)
    WHERE trace_id IS NOT NULL;
