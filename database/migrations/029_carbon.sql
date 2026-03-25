-- Migration 029: Carbon intensity cache + stamp columns on gpu_metrics
--
-- Creates:
--   carbon_intensities  — hourly cache of gCO₂eq/kWh per grid zone (from Electricity Maps)
--   gpu_metrics.grid_zone         — zone code stamped at ingest time
--   gpu_metrics.carbon_g_per_kwh  — intensity locked at moment of consumption

CREATE TABLE IF NOT EXISTS carbon_intensities (
  zone              TEXT         NOT NULL,
  carbon_g_per_kwh  NUMERIC(8,2) NOT NULL,
  emission_factor   TEXT         NOT NULL DEFAULT 'lifecycle',
  is_estimated      BOOLEAN      NOT NULL DEFAULT TRUE,
  source            TEXT         NOT NULL DEFAULT 'electricity_maps',
  fetched_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  PRIMARY KEY (zone, fetched_at)
);

CREATE INDEX IF NOT EXISTS idx_carbon_intensities_zone_latest
  ON carbon_intensities (zone, fetched_at DESC);

-- Stamp columns on gpu_metrics — populated at ingest time
ALTER TABLE gpu_metrics
  ADD COLUMN IF NOT EXISTS grid_zone        TEXT,
  ADD COLUMN IF NOT EXISTS carbon_g_per_kwh NUMERIC(8,2);

CREATE INDEX IF NOT EXISTS idx_gpu_metrics_zone
  ON gpu_metrics (user_id, grid_zone, time DESC)
  WHERE grid_zone IS NOT NULL;

-- Rollback:
-- DROP INDEX IF EXISTS idx_gpu_metrics_zone;
-- ALTER TABLE gpu_metrics DROP COLUMN IF EXISTS grid_zone, DROP COLUMN IF EXISTS carbon_g_per_kwh;
-- DROP INDEX IF EXISTS idx_carbon_intensities_zone_latest;
-- DROP TABLE IF EXISTS carbon_intensities;
