CREATE TABLE IF NOT EXISTS usage_quota_policies (
    id uuid PRIMARY KEY,
    name text NOT NULL UNIQUE,
    metric text NOT NULL,
    period text NOT NULL,
    limit_count integer NOT NULL CHECK (limit_count > 0),
    enabled boolean NOT NULL DEFAULT true,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CONSTRAINT usage_quota_policies_metric_period_unique
        UNIQUE (metric, period)
);

CREATE TABLE IF NOT EXISTS user_quota_overrides (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    metric text NOT NULL,
    period text NOT NULL,
    limit_count integer NOT NULL CHECK (limit_count > 0),
    enabled boolean NOT NULL DEFAULT true,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CONSTRAINT user_quota_overrides_user_metric_period_unique
        UNIQUE (user_id, metric, period)
);

CREATE TABLE IF NOT EXISTS usage_quota_counters (
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    metric text NOT NULL,
    period text NOT NULL,
    period_start timestamptz NOT NULL,
    used_count integer NOT NULL CHECK (used_count >= 0),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (user_id, metric, period, period_start)
);

CREATE INDEX IF NOT EXISTS usage_quota_counters_period_metric_idx
    ON usage_quota_counters (period_start, metric);

INSERT INTO usage_quota_policies (
    id,
    name,
    metric,
    period,
    limit_count,
    enabled,
    metadata,
    created_at,
    updated_at
) VALUES (
    '6dfdb0bf-61bd-4d6f-80e1-6e8e2e96ee90',
    'free_beta_default_agent_turn_day',
    'agent_turn',
    'day',
    100,
    true,
    '{"scope":"free_beta_default"}'::jsonb,
    now(),
    now()
)
ON CONFLICT (metric, period) DO UPDATE
SET limit_count = EXCLUDED.limit_count,
    enabled = EXCLUDED.enabled,
    metadata = EXCLUDED.metadata,
    updated_at = now();
