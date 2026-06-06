-- 052_bandit_models.sql
-- Fleet-wide bandit model sharing.
-- Agents in the same cluster can share trained contextual bandit models.

CREATE TABLE IF NOT EXISTS bandit_models (
  id                UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           UUID          NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  gpu_class         TEXT          NOT NULL,
  model_version     INT           NOT NULL DEFAULT 1,
  corpus_size       INT           NOT NULL DEFAULT 0,
  estimated_reward  NUMERIC(6,4)  NOT NULL DEFAULT 0,
  model_blob        BYTEA,
  cluster_tag       TEXT          NOT NULL DEFAULT '',
  created_at        TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_bandit_models_user_gpu ON bandit_models(user_id, gpu_class, model_version DESC);
CREATE INDEX idx_bandit_models_cluster ON bandit_models(user_id, cluster_tag, gpu_class);

-- ── RLS ───────────────────────────────────────────────────────────────────────

ALTER TABLE bandit_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own bandit models"
  ON bandit_models FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage bandit models"
  ON bandit_models FOR ALL
  WITH CHECK (true);

-- ── Rollback ───────────────────────────────────────────────────────────────────
-- DROP POLICY IF EXISTS "Users can view own bandit models" ON bandit_models;
-- DROP POLICY IF EXISTS "Service role can manage bandit models" ON bandit_models;
-- DROP INDEX IF EXISTS idx_bandit_models_user_gpu;
-- DROP INDEX IF EXISTS idx_bandit_models_cluster;
-- DROP TABLE IF EXISTS bandit_models;
