-- Migration 031: Green AI Index — extend benchmark_submissions with throughput fields
-- and replace leaderboard stats function to include kWh/1M tokens.

-- Extend benchmark_submissions with throughput + framework fields
ALTER TABLE benchmark_submissions
  ADD COLUMN IF NOT EXISTS tokens_per_second  NUMERIC(12,2),   -- from --throughput CLI flag
  ADD COLUMN IF NOT EXISTS framework_tag      TEXT NOT NULL DEFAULT 'unknown',
  ADD COLUMN IF NOT EXISTS kwh_per_1m_tokens  NUMERIC(12,6);   -- pre-computed at insert

-- Index for framework-level filtering
CREATE INDEX IF NOT EXISTS idx_benchmark_submissions_framework
  ON benchmark_submissions (gpu_arch, model_tag, framework_tag);

-- Replace leaderboard stats function to add throughput fields.
-- Backward-compatible: new columns return NULL for existing rows without throughput data.
CREATE OR REPLACE FUNCTION benchmark_leaderboard_stats()
RETURNS TABLE (
  gpu_arch               TEXT,
  model_tag              TEXT,
  sample_count           BIGINT,
  p25_j_per_gpu_hour     NUMERIC,
  p50_j_per_gpu_hour     NUMERIC,
  p75_j_per_gpu_hour     NUMERIC,
  best_j_per_gpu_hour    NUMERIC,
  best_precision_tag     TEXT,
  best_kwh_per_1m_tokens NUMERIC,      -- NULL if no throughput submissions
  p50_kwh_per_1m_tokens  NUMERIC,      -- NULL if no throughput submissions
  top_framework_tag      TEXT          -- most common non-'unknown' framework
) LANGUAGE SQL STABLE AS $$
  SELECT
    gpu_arch,
    model_tag,
    COUNT(*),
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour),
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour),
    MIN(energy_j_per_gpu_hour),
    MODE()  WITHIN GROUP (ORDER BY precision_tag),
    MIN(kwh_per_1m_tokens),
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY kwh_per_1m_tokens)
      FILTER (WHERE kwh_per_1m_tokens IS NOT NULL),
    MODE()  WITHIN GROUP (ORDER BY framework_tag)
      FILTER (WHERE framework_tag <> 'unknown')
  FROM benchmark_submissions
  GROUP BY gpu_arch, model_tag
  HAVING COUNT(*) >= 10
  ORDER BY gpu_arch,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY energy_j_per_gpu_hour);
$$;

-- Rollback:
-- DROP INDEX IF EXISTS idx_benchmark_submissions_framework;
-- ALTER TABLE benchmark_submissions
--   DROP COLUMN IF EXISTS tokens_per_second,
--   DROP COLUMN IF EXISTS framework_tag,
--   DROP COLUMN IF EXISTS kwh_per_1m_tokens;
-- (Restore original benchmark_leaderboard_stats() from migration 028 rollback)
