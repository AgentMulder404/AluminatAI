-- 038_data_retention.sql
-- Per-user data retention policies for gpu_metrics and related tables.
-- Minimum 7 days, default 90 days. Daily cron deletes expired rows.

BEGIN;

CREATE TABLE IF NOT EXISTS data_retention_policies (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    table_name      TEXT NOT NULL DEFAULT 'gpu_metrics'
                    CHECK (table_name IN ('gpu_metrics', 'waste_events', 'webhook_deliveries', 'budget_alerts')),
    retention_days  INT NOT NULL DEFAULT 90
                    CHECK (retention_days >= 7),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One policy per user per table
CREATE UNIQUE INDEX idx_retention_unique ON data_retention_policies(user_id, table_name);

COMMIT;
