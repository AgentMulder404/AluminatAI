-- Migration 035: Reference cloud GPU pricing for savings calculator
--
-- System-wide table of cloud GPU on-demand and spot rates.
-- Used by the recommendations API to suggest GPU swaps and spot pricing.

CREATE TABLE IF NOT EXISTS gpu_reference_pricing (
  id                          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
  gpu_model                   TEXT          NOT NULL,
  provider                    TEXT          NOT NULL,
  instance_type               TEXT,
  rate_usd_per_gpu_hour       NUMERIC(10,4) NOT NULL CHECK (rate_usd_per_gpu_hour >= 0),
  spot_rate_usd_per_gpu_hour  NUMERIC(10,4),
  tdp_watts                   INT,
  memory_gb                   INT,
  updated_at                  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_ref_pricing_dedup
  ON gpu_reference_pricing (gpu_model, provider, COALESCE(instance_type, ''));

-- Seed reference pricing (current as of 2026-03)
INSERT INTO gpu_reference_pricing (gpu_model, provider, instance_type, rate_usd_per_gpu_hour, spot_rate_usd_per_gpu_hour, tdp_watts, memory_gb) VALUES
  ('A100-SXM4-80GB', 'aws',       'p4d.24xlarge',   3.40, 1.50, 400, 80),
  ('A100-SXM4-80GB', 'gcp',       'a2-highgpu-1g',  2.21, 0.88, 400, 80),
  ('A100-SXM4-80GB', 'azure',     'NC96ads_A100_v4', 2.48, 1.10, 400, 80),
  ('A100-SXM4-80GB', 'lambda',    'gpu_1x_a100_sxm4', 1.29, NULL, 400, 80),
  ('A100-SXM4-80GB', 'coreweave', 'a100_80gb',       2.06, NULL, 400, 80),
  ('H100-SXM5-80GB', 'aws',       'p5.48xlarge',     8.22, 3.50, 700, 80),
  ('H100-SXM5-80GB', 'gcp',       'a3-highgpu-1g',   3.22, 1.29, 700, 80),
  ('H100-SXM5-80GB', 'azure',     'ND96isr_H100_v5', 3.67, 1.60, 700, 80),
  ('H100-SXM5-80GB', 'lambda',    'gpu_1x_h100_sxm5', 2.49, NULL, 700, 80),
  ('H100-SXM5-80GB', 'coreweave', 'h100_80gb',       2.49, NULL, 700, 80),
  ('RTX-4090',       'lambda',    'gpu_1x_rtx4090',   0.69, NULL, 450, 24),
  ('RTX-4090',       'runpod',    '1x_rtx4090',       0.59, 0.39, 450, 24),
  ('A10G',           'aws',       'g5.xlarge',        1.01, 0.40, 150, 24),
  ('A10G',           'gcp',       'g2-standard-4',    0.84, 0.34, 150, 24),
  ('L4',             'gcp',       'g2-standard-4',    0.65, 0.26, 72,  24),
  ('L40S',           'aws',       'g6e.xlarge',       1.24, 0.55, 350, 48),
  ('T4',             'aws',       'g4dn.xlarge',      0.53, 0.16, 70,  16),
  ('T4',             'gcp',       'n1-standard-4',    0.35, 0.14, 70,  16),
  ('V100-SXM2-16GB', 'aws',       'p3.2xlarge',       3.06, 0.92, 300, 16),
  ('V100-SXM2-16GB', 'gcp',       'n1-standard-8',    2.48, 0.74, 300, 16)
ON CONFLICT (gpu_model, provider, COALESCE(instance_type, '')) DO UPDATE
  SET rate_usd_per_gpu_hour      = EXCLUDED.rate_usd_per_gpu_hour,
      spot_rate_usd_per_gpu_hour = EXCLUDED.spot_rate_usd_per_gpu_hour,
      tdp_watts                  = EXCLUDED.tdp_watts,
      memory_gb                  = EXCLUDED.memory_gb,
      updated_at                 = NOW();

-- Rollback:
-- DROP TABLE IF EXISTS gpu_reference_pricing;
