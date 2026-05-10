-- Migration 046: Agent Health Monitoring
-- Extends agent_heartbeats with error/health telemetry, adds error log + health alerts tables.

-- ── Extend agent_heartbeats ────────────────────────────────────────────────────

ALTER TABLE agent_heartbeats
  ADD COLUMN IF NOT EXISTS error_count_total     INT         DEFAULT 0,
  ADD COLUMN IF NOT EXISTS error_count_last_hour INT         DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_error_message    TEXT,
  ADD COLUMN IF NOT EXISTS last_error_at         TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS os_info               TEXT,
  ADD COLUMN IF NOT EXISTS python_version        TEXT,
  ADD COLUMN IF NOT EXISTS gpu_backend           TEXT,
  ADD COLUMN IF NOT EXISTS agent_mode            TEXT        DEFAULT 'normal';

-- ── Agent error log ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_error_log (
  id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  machine_id     TEXT        NOT NULL,
  hostname       TEXT        NOT NULL DEFAULT '',
  error_type     TEXT        NOT NULL,  -- collection, upload, scheduler, power_control, attribution
  error_message  TEXT        NOT NULL,
  stack_trace    TEXT,
  gpu_index      INT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_error_log_user_created
  ON agent_error_log (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_error_log_machine_created
  ON agent_error_log (machine_id, created_at DESC);

-- ── Agent health alerts ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_health_alerts (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  machine_id  TEXT        NOT NULL,
  hostname    TEXT        NOT NULL DEFAULT '',
  alert_type  TEXT        NOT NULL,  -- agent_offline, version_mismatch, config_drift, high_error_rate
  severity    TEXT        NOT NULL DEFAULT 'warning',  -- warning, critical
  message     TEXT        NOT NULL,
  resolved    BOOLEAN     NOT NULL DEFAULT FALSE,
  resolved_at TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_health_alerts_user_resolved
  ON agent_health_alerts (user_id, resolved, created_at DESC);

-- ── Rollback ───────────────────────────────────────────────────────────────────
-- ALTER TABLE agent_heartbeats
--   DROP COLUMN IF EXISTS error_count_total,
--   DROP COLUMN IF EXISTS error_count_last_hour,
--   DROP COLUMN IF EXISTS last_error_message,
--   DROP COLUMN IF EXISTS last_error_at,
--   DROP COLUMN IF EXISTS os_info,
--   DROP COLUMN IF EXISTS python_version,
--   DROP COLUMN IF EXISTS gpu_backend,
--   DROP COLUMN IF EXISTS agent_mode;
-- DROP TABLE IF EXISTS agent_health_alerts;
-- DROP TABLE IF EXISTS agent_error_log;
