-- 051_experience_log.sql
-- Self-learning agent: experience log for (context, action, outcome) tuples.
-- Used to build the training corpus for the contextual bandit (Phase 2).

CREATE TABLE IF NOT EXISTS experience_log (
  id                UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           UUID          NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  machine_id        TEXT          NOT NULL,
  gpu_index         INT           NOT NULL,
  gpu_name          TEXT          NOT NULL,
  gpu_arch          TEXT          NOT NULL DEFAULT '',

  -- Context snapshot
  workload_class    TEXT          NOT NULL DEFAULT 'unknown',
  utilization_gpu   NUMERIC(5,2)  NOT NULL DEFAULT 0,
  utilization_mem   NUMERIC(5,2)  NOT NULL DEFAULT 0,
  memory_pressure   NUMERIC(5,4)  NOT NULL DEFAULT 0,
  power_draw_w      NUMERIC(8,2)  NOT NULL DEFAULT 0,
  power_limit_w     NUMERIC(8,2)  NOT NULL DEFAULT 0,
  temperature_c     NUMERIC(5,1)  NOT NULL DEFAULT 0,

  -- Action taken
  action_type       TEXT          NOT NULL,
  action_source     TEXT          NOT NULL,
  recommended_value NUMERIC(10,2) NOT NULL DEFAULT 0,
  current_value     NUMERIC(10,2) NOT NULL DEFAULT 0,
  estimated_savings NUMERIC(5,2)  NOT NULL DEFAULT 0,

  -- Outcome (NULL until observed after outcome window)
  energy_before_j   NUMERIC(12,4),
  energy_after_j    NUMERIC(12,4),
  throughput_before  NUMERIC(8,4),
  throughput_after   NUMERIC(8,4),
  actual_savings_pct NUMERIC(5,2),
  rec_status         TEXT,
  observation_s      NUMERIC(8,1),

  -- Reward (computed from outcome, used for bandit training)
  reward            NUMERIC(6,4),

  recorded_at       TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- Indexes for bandit training queries (filter by GPU arch + time, workload + action)
CREATE INDEX idx_exp_log_user_gpu ON experience_log(user_id, gpu_arch, recorded_at DESC);
CREATE INDEX idx_exp_log_workload ON experience_log(user_id, workload_class, action_type);

-- ── RLS ───────────────────────────────────────────────────────────────────────

ALTER TABLE experience_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own experience logs"
  ON experience_log FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Service role can insert experience logs"
  ON experience_log FOR INSERT
  WITH CHECK (true);

-- ── Rollback ───────────────────────────────────────────────────────────────────
-- DROP POLICY IF EXISTS "Users can view own experience logs" ON experience_log;
-- DROP POLICY IF EXISTS "Service role can insert experience logs" ON experience_log;
-- DROP INDEX IF EXISTS idx_exp_log_user_gpu;
-- DROP INDEX IF EXISTS idx_exp_log_workload;
-- DROP TABLE IF EXISTS experience_log;
