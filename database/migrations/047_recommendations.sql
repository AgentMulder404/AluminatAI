-- Migration 047: Optimization Recommendations + Command Queue (Advisor Tier)

-- ── Optimization recommendations ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS optimization_recommendations (
  id                    UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id               UUID          NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  machine_id            TEXT          NOT NULL,
  hostname              TEXT          NOT NULL DEFAULT '',
  gpu_index             INT,
  gpu_name              TEXT,

  source                TEXT          NOT NULL,  -- auto_tuner, workload_analyzer, carbon_scheduler, swarm_policy
  category              TEXT          NOT NULL,  -- power_cap, precision, utilization, idle, carbon_schedule, gpu_match, thermal, memory
  priority              TEXT          NOT NULL DEFAULT 'P2',  -- P1, P2, P3

  title                 TEXT          NOT NULL,
  description           TEXT          NOT NULL DEFAULT '',
  action                TEXT          NOT NULL DEFAULT '',
  detail                TEXT          NOT NULL DEFAULT '',

  estimated_savings_pct NUMERIC(5,2)  DEFAULT 0,
  estimated_savings_usd NUMERIC(10,2) DEFAULT 0,
  effort_score          INT           DEFAULT 3,  -- 1-5 (1=easy, 5=hard)

  status                TEXT          NOT NULL DEFAULT 'pending',  -- pending, approved, applied, rejected, expired, rolled_back
  approved_by           UUID          REFERENCES auth.users(id),
  approved_at           TIMESTAMPTZ,
  applied_at            TIMESTAMPTZ,
  rolled_back_at        TIMESTAMPTZ,

  action_payload        JSONB         NOT NULL DEFAULT '{}',  -- machine-readable params for agent execution
  actual_savings_pct    NUMERIC(5,2),
  actual_savings_usd    NUMERIC(10,2),

  expires_at            TIMESTAMPTZ,
  created_at            TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recommendations_user_status
  ON optimization_recommendations (user_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_recommendations_machine_status
  ON optimization_recommendations (machine_id, status);

-- ── Recommendation audit trail ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS recommendation_actions (
  id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  recommendation_id UUID        NOT NULL REFERENCES optimization_recommendations(id) ON DELETE CASCADE,
  user_id           UUID        REFERENCES auth.users(id),
  action            TEXT        NOT NULL,  -- created, approved, applied, rejected, rolled_back, feedback_recorded
  metadata          JSONB       NOT NULL DEFAULT '{}',
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Agent command queue ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_commands (
  id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  machine_id          TEXT        NOT NULL,
  recommendation_id   UUID        REFERENCES optimization_recommendations(id) ON DELETE SET NULL,
  command_type        TEXT        NOT NULL,  -- apply_power_cap, rollback_power_cap
  params              JSONB       NOT NULL DEFAULT '{}',
  status              TEXT        NOT NULL DEFAULT 'pending',  -- pending, dispatched, applied, failed
  result              JSONB       DEFAULT '{}',
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  dispatched_at       TIMESTAMPTZ,
  completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_commands_machine_status
  ON agent_commands (machine_id, status, created_at ASC);

-- ── RLS ───────────────────────────────────────────────────────────────────────

ALTER TABLE optimization_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendation_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_commands ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own recommendations"
  ON optimization_recommendations FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can update own recommendations"
  ON optimization_recommendations FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Service role can insert recommendations"
  ON optimization_recommendations FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Users can view own recommendation actions"
  ON recommendation_actions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Service role can insert recommendation actions"
  ON recommendation_actions FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Users can view own commands"
  ON agent_commands FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage commands"
  ON agent_commands FOR ALL
  WITH CHECK (true);

-- ── Rollback ───────────────────────────────────────────────────────────────────
-- DROP POLICY IF EXISTS "Users can view own recommendations" ON optimization_recommendations;
-- DROP POLICY IF EXISTS "Users can update own recommendations" ON optimization_recommendations;
-- DROP POLICY IF EXISTS "Service role can insert recommendations" ON optimization_recommendations;
-- DROP POLICY IF EXISTS "Users can view own recommendation actions" ON recommendation_actions;
-- DROP POLICY IF EXISTS "Service role can insert recommendation actions" ON recommendation_actions;
-- DROP POLICY IF EXISTS "Users can view own commands" ON agent_commands;
-- DROP POLICY IF EXISTS "Service role can manage commands" ON agent_commands;
-- ALTER TABLE optimization_recommendations DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE recommendation_actions DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE agent_commands DISABLE ROW LEVEL SECURITY;
-- DROP TABLE IF EXISTS agent_commands;
-- DROP TABLE IF EXISTS recommendation_actions;
-- DROP TABLE IF EXISTS optimization_recommendations;
