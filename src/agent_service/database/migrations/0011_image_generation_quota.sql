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
    'ae1b2675-0bfd-4b7a-9dd9-bde8e3212acd',
    'free_beta_default_image_generation_day',
    'image_generation',
    'day',
    3,
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
