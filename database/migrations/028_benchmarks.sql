-- Migration 028: Energy benchmarking — opt-in anonymous submissions + leaderboard
--
-- Privacy: user_id is never stored. user_hash = HMAC-SHA256(user_id, BENCHMARK_SALT)
-- computed server-side; individual rows are never exposed via API.

-- Opt-in flag on users table (created in aluminatai-landing migrations)
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS benchmark_opt_in BOOLEAN NOT NULL DEFAULT FALSE;

-- Anonymous benchmark submissions: one row per (user_hash, gpu_arch, model_tag)
CREATE TABLE IF NOT EXISTS benchmark_submissions (
  id                    UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_hash             TEXT          NOT NULL,
  gpu_arch              TEXT          NOT NULL,          -- e.g. "A100-SXM4-80GB"
  model_tag             TEXT          NOT NULL,          -- e.g. "llama-3-8b"
  precision_tag         TEXT          NOT NULL DEFAULT 'unknown',
  batch_size_hint       INT,
  avg_power_w           NUMERIC(8,2)  NOT NULL,
  energy_j_per_gpu_hour NUMERIC(14,4) NOT NULL,
  duration_seconds      INT           NOT NULL,
  gpu_count             INT           NOT NULL DEFAULT 1,
  opt_in_at             TIMESTAMPTZ   NOT NULL,
  created_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- One submission per (user, gpu, model) — upsert deduplication key
CREATE UNIQUE INDEX IF NOT EXISTS idx_benchmark_submissions_dedup
  ON benchmark_submissions (user_hash, gpu_arch, model_tag);

-- Leaderboard queries
CREATE INDEX IF NOT EXISTS idx_benchmark_submissions_leaderboard
  ON benchmark_submissions (gpu_arch, model_tag);

-- No RLS — service role only; individual rows never exposed via API

-- Leaderboard aggregation: only expose groups with ≥10 submissions for k-anonymity
CREATE OR REPLACE FUNCTION benchmark_leaderboard_stats()
RETURNS TABLE (
  gpu_arch              TEXT,
  model_tag             TEXT,
  sample_count          BIGINT,
  p25_j_per_gpu_hour    NUMERIC,
  p50_j_per_gpu_hour    NUMERIC,
  p75_j_per_gpu_hour    NUMERIC,
  best_j_per_gpu_hour   NUMERIC,
  best_precision_tag    TEXT
) LANGUAGE SQL STABLE AS $$
  SELECT
    gpu_arch,
    model_tag,
    COUNT(*),
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour),
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour),
    MIN(energy_j_per_gpu_hour),
    MODE() WITHIN GROUP (ORDER BY precision_tag)
  FROM benchmark_submissions
  GROUP BY gpu_arch, model_tag
  HAVING COUNT(*) >= 10
  ORDER BY gpu_arch, PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour);
$$;

-- Rollback:
-- DROP FUNCTION IF EXISTS benchmark_leaderboard_stats();
-- DROP TABLE IF EXISTS benchmark_submissions;
-- ALTER TABLE users DROP COLUMN IF EXISTS benchmark_opt_in;
