-- 037_webhooks.sql
-- User-configured event webhooks with HMAC-SHA256 signing and delivery tracking.

BEGIN;

CREATE TABLE IF NOT EXISTS webhooks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    team_id         UUID REFERENCES teams(id) ON DELETE CASCADE,
    url             TEXT NOT NULL,
    secret          TEXT NOT NULL,  -- HMAC-SHA256 signing secret
    description     TEXT DEFAULT '',
    event_types     TEXT[] NOT NULL DEFAULT '{}',
                    -- e.g. {'budget.warning', 'budget.exceeded', 'waste.detected', 'agent.offline'}
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_webhooks_user ON webhooks(user_id);
CREATE INDEX idx_webhooks_team ON webhooks(team_id) WHERE team_id IS NOT NULL;
CREATE INDEX idx_webhooks_active ON webhooks(is_active) WHERE is_active = TRUE;

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id      UUID NOT NULL REFERENCES webhooks(id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}',
    response_status INT,
    response_body   TEXT,
    duration_ms     INT,
    success         BOOLEAN NOT NULL DEFAULT FALSE,
    attempt         INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_webhook_deliveries_webhook ON webhook_deliveries(webhook_id);
CREATE INDEX idx_webhook_deliveries_created ON webhook_deliveries(created_at);

-- Auto-prune delivery logs older than 30 days (handled by data-retention cron,
-- but add a partial index to make the cleanup query fast)
CREATE INDEX idx_webhook_deliveries_old
    ON webhook_deliveries(created_at)
    WHERE created_at < now() - INTERVAL '30 days';

COMMIT;
