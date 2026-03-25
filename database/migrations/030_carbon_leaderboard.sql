-- Migration 030: Carbon efficiency leaderboard
--
-- One row per (user_hash, gpu_arch, model_tag, grid_zone).
-- Same privacy model as benchmark_submissions: user ID is never stored, only an HMAC hash.

CREATE TABLE IF NOT EXISTS carbon_submissions (
  id                   UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_hash            TEXT          NOT NULL,
  gpu_arch             TEXT          NOT NULL,
  model_tag            TEXT          NOT NULL,
  grid_zone            TEXT          NOT NULL,
  co2e_g_per_gpu_hour  NUMERIC(14,4) NOT NULL,
  kwh_per_gpu_hour     NUMERIC(10,6) NOT NULL,
  carbon_g_per_kwh     NUMERIC(8,2)  NOT NULL,
  duration_seconds     INT           NOT NULL,
  opt_in_at            TIMESTAMPTZ   NOT NULL,
  created_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_carbon_submissions_dedup
  ON carbon_submissions (user_hash, gpu_arch, model_tag, grid_zone);

CREATE INDEX IF NOT EXISTS idx_carbon_submissions_leaderboard
  ON carbon_submissions (gpu_arch, model_tag, grid_zone);

-- k-anonymity threshold: 5 submissions per group
CREATE OR REPLACE FUNCTION carbon_leaderboard_stats()
RETURNS TABLE (
  gpu_arch                TEXT,
  model_tag               TEXT,
  grid_zone               TEXT,
  sample_count            BIGINT,
  p50_co2e_g_per_gpu_hour NUMERIC,
  best_co2e_g_per_gpu_hour NUMERIC,
  avg_carbon_g_per_kwh    NUMERIC
) LANGUAGE SQL STABLE AS $$
  SELECT
    gpu_arch,
    model_tag,
    grid_zone,
    COUNT(*),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY co2e_g_per_gpu_hour),
    MIN(co2e_g_per_gpu_hour),
    AVG(carbon_g_per_kwh)
  FROM carbon_submissions
  GROUP BY gpu_arch, model_tag, grid_zone
  HAVING COUNT(*) >= 5
  ORDER BY gpu_arch, model_tag,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY co2e_g_per_gpu_hour);
$$;

-- Rollback:
-- DROP FUNCTION IF EXISTS carbon_leaderboard_stats();
-- DROP INDEX IF EXISTS idx_carbon_submissions_leaderboard;
-- DROP INDEX IF EXISTS idx_carbon_submissions_dedup;
-- DROP TABLE IF EXISTS carbon_submissions;
