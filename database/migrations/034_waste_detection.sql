-- Migration 034: Waste detection events and scheduling recommendations
--
-- waste_events: populated by the waste-detect cron (every 6h)
-- scheduling_recommendations: populated alongside waste events with actionable suggestions

CREATE TABLE IF NOT EXISTS waste_events (
  id                    UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id               UUID          NOT NULL,
  gpu_uuid              TEXT          NOT NULL,
  gpu_name              TEXT,
  job_id                TEXT,
  team_id               TEXT,
  cluster_tag           TEXT,
  waste_type            TEXT          NOT NULL CHECK (waste_type IN (
    'idle_gpu', 'low_utilization', 'oversized_gpu', 'long_idle_between_jobs'
  )),
  avg_utilization_pct   NUMERIC(5,2),
  duration_hours        NUMERIC(8,2)  NOT NULL CHECK (duration_hours > 0),
  estimated_waste_usd   NUMERIC(10,2) NOT NULL CHECK (estimated_waste_usd >= 0),
  detected_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  dismissed             BOOLEAN       NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_waste_events_user
  ON waste_events (user_id, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_waste_events_undismissed
  ON waste_events (user_id) WHERE dismissed = FALSE;

CREATE TABLE IF NOT EXISTS scheduling_recommendations (
  id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                 UUID          NOT NULL,
  recommendation_type     TEXT          NOT NULL CHECK (recommendation_type IN (
    'time_shift', 'gpu_downsize', 'spot_pricing', 'consolidation'
  )),
  title                   TEXT          NOT NULL,
  description             TEXT          NOT NULL,
  estimated_savings_usd   NUMERIC(10,2),
  estimated_savings_pct   NUMERIC(5,2),
  context                 JSONB         NOT NULL DEFAULT '{}',
  created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  dismissed               BOOLEAN       NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_recommendations_user
  ON scheduling_recommendations (user_id, created_at DESC);

-- No RLS — service role access only

-- Rollback:
-- DROP TABLE IF EXISTS scheduling_recommendations;
-- DROP TABLE IF EXISTS waste_events;
