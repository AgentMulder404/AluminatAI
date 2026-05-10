-- Migration 048: Swarm production scaling — leader election + fleet state view

-- ── Leader lease ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS swarm_leader_leases (
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  cluster_tag   TEXT        NOT NULL DEFAULT '',
  machine_id    TEXT        NOT NULL,
  lease_token   UUID        NOT NULL DEFAULT gen_random_uuid(),
  acquired_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at    TIMESTAMPTZ NOT NULL,
  CONSTRAINT swarm_leader_leases_pkey PRIMARY KEY (user_id, cluster_tag)
);

ALTER TABLE swarm_leader_leases ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role manages leader leases"
  ON swarm_leader_leases FOR ALL
  WITH CHECK (true);

-- ── Fleet state view (pre-aggregated latest metrics per GPU) ──────────────────

CREATE OR REPLACE FUNCTION get_fleet_gpu_latest(p_user_id UUID)
RETURNS TABLE (
  machine_id          TEXT,
  gpu_index           INT,
  gpu_name            TEXT,
  power_draw_w        DOUBLE PRECISION,
  power_limit_w       DOUBLE PRECISION,
  utilization_gpu_pct DOUBLE PRECISION,
  temperature_c       DOUBLE PRECISION,
  memory_used_mb      DOUBLE PRECISION,
  memory_total_mb     DOUBLE PRECISION,
  energy_j_last_hour  DOUBLE PRECISION,
  model_tag           TEXT,
  job_id              TEXT,
  grid_zone           TEXT,
  carbon_g_per_kwh    DOUBLE PRECISION,
  sample_time         TIMESTAMPTZ
) LANGUAGE sql STABLE AS $$
  WITH latest AS (
    SELECT DISTINCT ON (m.machine_id, m.gpu_index)
      m.machine_id,
      m.gpu_index,
      m.gpu_name,
      m.power_draw_w,
      m.power_limit_w,
      m.utilization_gpu_pct,
      m.temperature_c,
      m.memory_used_mb,
      m.memory_total_mb,
      m.model_tag,
      m.job_id,
      m.grid_zone,
      m.carbon_g_per_kwh,
      m.time AS sample_time
    FROM gpu_metrics m
    WHERE m.user_id = p_user_id
      AND m.time > now() - INTERVAL '10 minutes'
    ORDER BY m.machine_id, m.gpu_index, m.time DESC
  ),
  energy AS (
    SELECT
      m.machine_id,
      m.gpu_index,
      COALESCE(SUM(m.energy_delta_j), 0) AS energy_j_last_hour
    FROM gpu_metrics m
    WHERE m.user_id = p_user_id
      AND m.time > now() - INTERVAL '1 hour'
    GROUP BY m.machine_id, m.gpu_index
  )
  SELECT
    l.machine_id,
    l.gpu_index,
    l.gpu_name,
    l.power_draw_w,
    l.power_limit_w,
    l.utilization_gpu_pct,
    l.temperature_c,
    l.memory_used_mb,
    l.memory_total_mb,
    COALESCE(e.energy_j_last_hour, 0) AS energy_j_last_hour,
    l.model_tag,
    l.job_id,
    l.grid_zone,
    l.carbon_g_per_kwh,
    l.sample_time
  FROM latest l
  LEFT JOIN energy e USING (machine_id, gpu_index);
$$;

-- ── Recommendation dedup index ────────────────────────────────────────────────

CREATE UNIQUE INDEX IF NOT EXISTS idx_recommendations_dedup
  ON optimization_recommendations (user_id, machine_id, category, gpu_index)
  WHERE status = 'pending' AND created_at > now() - INTERVAL '1 hour';

-- ── Rollback ──────────────────────────────────────────────────────────────────
-- DROP INDEX IF EXISTS idx_recommendations_dedup;
-- DROP FUNCTION IF EXISTS get_fleet_gpu_latest;
-- DROP TABLE IF EXISTS swarm_leader_leases;
