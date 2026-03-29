-- 039_slack_and_exports.sql
-- Slack OAuth installations + S3/GCS export configurations.

BEGIN;

-- ── Slack installations ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS slack_installations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    team_id         UUID REFERENCES teams(id) ON DELETE CASCADE,
    slack_team_id   TEXT NOT NULL,
    slack_team_name TEXT NOT NULL DEFAULT '',
    bot_token       TEXT NOT NULL,          -- xoxb-...
    bot_user_id     TEXT NOT NULL DEFAULT '',
    channel_id      TEXT,                   -- default channel for daily summaries
    channel_name    TEXT,
    scope           TEXT NOT NULL DEFAULT '',
    installed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE UNIQUE INDEX idx_slack_install_unique
    ON slack_installations(user_id, slack_team_id);
CREATE INDEX idx_slack_install_active
    ON slack_installations(is_active) WHERE is_active = TRUE;

-- ── Export configurations ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS export_configs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider        TEXT NOT NULL CHECK (provider IN ('s3', 'gcs')),
    bucket          TEXT NOT NULL,
    prefix          TEXT NOT NULL DEFAULT 'aluminatai/',
    region          TEXT,                   -- e.g. us-east-1
    format          TEXT NOT NULL DEFAULT 'csv'
                    CHECK (format IN ('csv', 'jsonl')),
    schedule        TEXT NOT NULL DEFAULT 'weekly'
                    CHECK (schedule IN ('daily', 'weekly', 'monthly')),
    -- Credentials stored encrypted; only service-role can read
    access_key_id   TEXT,                   -- S3 only
    secret_key      TEXT,                   -- S3 only
    gcs_credentials JSONB,                  -- GCS service account JSON
    last_export_at  TIMESTAMPTZ,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_export_configs_user ON export_configs(user_id);
CREATE INDEX idx_export_configs_active
    ON export_configs(is_active) WHERE is_active = TRUE;

COMMIT;
