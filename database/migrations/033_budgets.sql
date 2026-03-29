-- Migration 033: Budget alerts with notification channels
--
-- Supports per-user budgets scoped to global, team, project, cluster, or GPU model.
-- Budget alerts are deduplicated per (budget, alert_type, period) to prevent spam.

CREATE TABLE IF NOT EXISTS budgets (
  id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID          NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name            TEXT          NOT NULL,
  scope_type      TEXT          NOT NULL CHECK (scope_type IN ('global', 'team', 'project', 'cluster', 'gpu_model')),
  scope_value     TEXT,           -- NULL for global; team_id, project_id, cluster_tag, or gpu_model
  period          TEXT          NOT NULL CHECK (period IN ('daily', 'weekly', 'monthly')),
  limit_usd       NUMERIC(12,2) NOT NULL CHECK (limit_usd > 0),
  warn_pct        INT           NOT NULL DEFAULT 80 CHECK (warn_pct BETWEEN 1 AND 100),
  notify_channels JSONB         NOT NULL DEFAULT '[]',
  is_active       BOOLEAN       NOT NULL DEFAULT TRUE,
  created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_budgets_user ON budgets (user_id) WHERE is_active = TRUE;

CREATE TABLE IF NOT EXISTS budget_alerts (
  id            UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  budget_id     UUID          NOT NULL REFERENCES budgets(id) ON DELETE CASCADE,
  user_id       UUID          NOT NULL,
  alert_type    TEXT          NOT NULL CHECK (alert_type IN ('warn', 'exceeded')),
  spend_usd     NUMERIC(12,2) NOT NULL,
  limit_usd     NUMERIC(12,2) NOT NULL,
  period_start  TIMESTAMPTZ   NOT NULL,
  notified_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  channels      JSONB         NOT NULL DEFAULT '[]'
);

-- Prevent duplicate alerts per budget per period
CREATE UNIQUE INDEX IF NOT EXISTS idx_budget_alerts_dedup
  ON budget_alerts (budget_id, alert_type, period_start);

-- No RLS — service role access only

-- Rollback:
-- DROP TABLE IF EXISTS budget_alerts;
-- DROP TABLE IF EXISTS budgets;
