-- Migration 032: User-configurable electricity rates + cloud GPU pricing
--
-- Replaces the hardcoded $0.12/kWh in today-cost and jobs API routes.
-- Users can set their own electricity rate and optionally track cloud GPU costs.

CREATE TABLE IF NOT EXISTS cost_rates (
  id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID          NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  rate_type       TEXT          NOT NULL CHECK (rate_type IN ('electricity', 'cloud_gpu')),
  rate_usd        NUMERIC(10,6) NOT NULL CHECK (rate_usd >= 0),
  gpu_model       TEXT,           -- NULL for electricity rates; e.g. 'A100-SXM4-80GB' for cloud GPU
  provider        TEXT,           -- 'aws', 'gcp', 'azure', 'custom', NULL for electricity
  label           TEXT,           -- user-facing label, e.g. 'US-West Datacenter'
  is_default      BOOLEAN       NOT NULL DEFAULT FALSE,
  effective_from  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- Only one default rate per user per type
CREATE UNIQUE INDEX IF NOT EXISTS idx_cost_rates_user_default
  ON cost_rates (user_id, rate_type) WHERE is_default = TRUE;

CREATE INDEX IF NOT EXISTS idx_cost_rates_user
  ON cost_rates (user_id, rate_type);

-- No RLS — service role access only (same pattern as benchmark_submissions)

-- Rollback:
-- DROP TABLE IF EXISTS cost_rates;
