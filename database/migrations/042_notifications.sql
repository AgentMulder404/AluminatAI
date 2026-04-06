-- Migration 042: In-app notifications and notification preferences
BEGIN;

CREATE TABLE IF NOT EXISTS notifications (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    type        TEXT NOT NULL CHECK (type IN ('budget_alert', 'waste_detected', 'agent_offline', 'system', 'team')),
    title       TEXT NOT NULL,
    message     TEXT NOT NULL,
    read        BOOLEAN NOT NULL DEFAULT FALSE,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_notifications_user_unread ON notifications(user_id, read, created_at DESC)
    WHERE read = FALSE;
CREATE INDEX idx_notifications_user_recent ON notifications(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS notification_preferences (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
    in_app      BOOLEAN NOT NULL DEFAULT TRUE,
    email       BOOLEAN NOT NULL DEFAULT TRUE,
    slack       BOOLEAN NOT NULL DEFAULT TRUE,
    pagerduty   BOOLEAN NOT NULL DEFAULT FALSE,
    opsgenie    BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;
