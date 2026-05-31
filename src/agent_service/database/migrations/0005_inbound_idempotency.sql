CREATE TABLE IF NOT EXISTS inbound_event_processing (
    event_id uuid PRIMARY KEY,
    channel text NOT NULL,
    idempotency_key text NOT NULL,
    external_update_id text,
    external_chat_id text NOT NULL,
    external_message_id text,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status text NOT NULL CHECK (
        status IN (
            'queued',
            'processing',
            'completed',
            'failed_retryable',
            'dead_letter',
            'fallback_sent'
        )
    ),
    trace_id text,
    failure_reason text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    first_received_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    processing_started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL,
    CONSTRAINT inbound_event_processing_idempotency_key_unique
        UNIQUE (idempotency_key)
);

CREATE UNIQUE INDEX IF NOT EXISTS inbound_event_processing_channel_update_unique
    ON inbound_event_processing (channel, external_update_id)
    WHERE external_update_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS inbound_event_processing_user_updated_at_idx
    ON inbound_event_processing (user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS inbound_event_processing_status_updated_at_idx
    ON inbound_event_processing (status, updated_at DESC);

CREATE INDEX IF NOT EXISTS inbound_event_processing_trace_id_idx
    ON inbound_event_processing (trace_id)
    WHERE trace_id IS NOT NULL;
