DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'conversations_id_user_id_unique'
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_id_user_id_unique UNIQUE (id, user_id);
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    from_sequence bigint NOT NULL CHECK (from_sequence > 0),
    to_sequence bigint NOT NULL CHECK (to_sequence >= from_sequence),
    previous_summary text,
    summary text,
    compacted_message_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    last_compacted_message_id uuid REFERENCES conversation_messages(id) ON DELETE RESTRICT,
    input_token_count integer NOT NULL DEFAULT 0 CHECK (input_token_count >= 0),
    output_token_count integer NOT NULL DEFAULT 0 CHECK (output_token_count >= 0),
    model text,
    status text NOT NULL CHECK (status IN ('completed', 'failed_retryable', 'dead_letter')),
    trace_id text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    CONSTRAINT conversation_summaries_sequence_bounds_check
        CHECK (to_sequence >= from_sequence),
    CONSTRAINT conversation_summaries_completed_content_check
        CHECK (
            status <> 'completed'
            OR (summary IS NOT NULL AND last_compacted_message_id IS NOT NULL)
        ),
    CONSTRAINT conversation_summaries_compacted_message_ids_check
        CHECK (jsonb_typeof(compacted_message_ids) = 'array'),
    CONSTRAINT conversation_summaries_conversation_user_fk
        FOREIGN KEY (conversation_id, user_id)
        REFERENCES conversations(id, user_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS conversation_summaries_conversation_created_at_idx
    ON conversation_summaries (conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS conversation_summaries_latest_completed_idx
    ON conversation_summaries (conversation_id, to_sequence DESC)
    WHERE status = 'completed';

CREATE UNIQUE INDEX IF NOT EXISTS conversation_summaries_completed_to_sequence_unique
    ON conversation_summaries (conversation_id, to_sequence)
    WHERE status = 'completed';

CREATE INDEX IF NOT EXISTS conversation_summaries_user_created_at_idx
    ON conversation_summaries (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS conversation_summaries_trace_id_idx
    ON conversation_summaries (trace_id)
    WHERE trace_id IS NOT NULL;
