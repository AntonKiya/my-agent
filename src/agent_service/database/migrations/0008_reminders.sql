CREATE TABLE IF NOT EXISTS reminders (
    id uuid PRIMARY KEY,
    group_id uuid,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    channel text NOT NULL,
    external_chat_id text NOT NULL,
    thread_id text,
    source_conversation_id uuid REFERENCES conversations(id) ON DELETE RESTRICT,
    source_inbound_event_id uuid,
    status text NOT NULL CHECK (status IN ('active', 'paused', 'completed', 'deleted')),
    timezone text NOT NULL,
    message text NOT NULL,
    schedule_json jsonb NOT NULL,
    source_text text,
    assumptions_json jsonb,
    next_fire_at_utc timestamptz,
    last_fire_at_utc timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS reminders_status_next_fire_at_idx
ON reminders (status, next_fire_at_utc);

CREATE INDEX IF NOT EXISTS reminders_user_status_created_at_idx
ON reminders (user_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS reminders_source_inbound_event_idx
ON reminders (source_inbound_event_id)
WHERE source_inbound_event_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS reminder_events (
    id uuid PRIMARY KEY,
    reminder_id uuid NOT NULL REFERENCES reminders(id) ON DELETE RESTRICT,
    scheduled_for_utc timestamptz NOT NULL,
    status text NOT NULL CHECK (status IN ('queued', 'sent', 'failed', 'skipped', 'expired')),
    idempotency_key text NOT NULL,
    outbound_event_id uuid,
    created_at timestamptz NOT NULL,
    sent_at timestamptz,
    error text,
    UNIQUE (reminder_id, scheduled_for_utc),
    UNIQUE (idempotency_key)
);

CREATE INDEX IF NOT EXISTS reminder_events_reminder_created_at_idx
ON reminder_events (reminder_id, created_at DESC);

CREATE TABLE IF NOT EXISTS notification_outbox (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    source_conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    reminder_event_id uuid REFERENCES reminder_events(id) ON DELETE RESTRICT,
    channel text NOT NULL,
    external_chat_id text NOT NULL,
    thread_id text,
    payload_json jsonb NOT NULL,
    status text NOT NULL CHECK (status IN ('pending', 'sending', 'sent', 'failed', 'dead_letter')),
    retry_count integer NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    available_at timestamptz NOT NULL,
    locked_until timestamptz,
    locked_by text,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS notification_outbox_status_available_lock_idx
ON notification_outbox (status, available_at, locked_until);

CREATE INDEX IF NOT EXISTS notification_outbox_reminder_event_idx
ON notification_outbox (reminder_event_id)
WHERE reminder_event_id IS NOT NULL;
