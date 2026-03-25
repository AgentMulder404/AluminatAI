-- Migration 027: Cluster summary view (last 24 hours)
-- Used by GET /api/dashboard/clusters to power the ClusterFilter UI.

CREATE OR REPLACE VIEW cluster_summary AS
SELECT
  user_id,
  COALESCE(NULLIF(cluster_tag, ''), 'default')   AS cluster_tag,
  COUNT(DISTINCT machine_id)                       AS machine_count,
  COUNT(DISTINCT gpu_uuid)                         AS gpu_count,
  SUM(COALESCE(energy_delta_j, 0)) / 3600000.0    AS total_kwh,
  AVG(power_draw_w)                                AS avg_power_w,
  AVG(utilization_gpu_pct)                         AS avg_utilization_pct,
  MAX(time)                                        AS last_seen
FROM gpu_metrics
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY user_id, COALESCE(NULLIF(cluster_tag, ''), 'default');

-- Rollback:
-- DROP VIEW IF EXISTS cluster_summary;
