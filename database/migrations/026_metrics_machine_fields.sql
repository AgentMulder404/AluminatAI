-- Migration 026: Machine identity fields on gpu_metrics
-- Enables per-machine and per-cluster filtering on all dashboard queries.

ALTER TABLE gpu_metrics
  ADD COLUMN IF NOT EXISTS machine_id  TEXT,
  ADD COLUMN IF NOT EXISTS cluster_tag TEXT;

-- Cluster-scoped queries (dashboard filter, cost rollup)
CREATE INDEX IF NOT EXISTS idx_gpu_metrics_cluster
  ON gpu_metrics (user_id, cluster_tag, time DESC)
  WHERE cluster_tag IS NOT NULL AND cluster_tag != '';

-- Machine-scoped queries (per-host breakdown)
CREATE INDEX IF NOT EXISTS idx_gpu_metrics_machine
  ON gpu_metrics (user_id, machine_id, time DESC)
  WHERE machine_id IS NOT NULL;

-- Rollback:
-- DROP INDEX IF EXISTS idx_gpu_metrics_cluster;
-- DROP INDEX IF EXISTS idx_gpu_metrics_machine;
-- ALTER TABLE gpu_metrics
--   DROP COLUMN IF EXISTS machine_id,
--   DROP COLUMN IF EXISTS cluster_tag;
