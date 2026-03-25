-- Migration 025: Machine identity fields on agent_heartbeats
-- Adds stable machine UUID, cluster tag, location hint, and GPU name list.
-- Backwards-compatible: old agents that don't send these fields leave them NULL.

ALTER TABLE agent_heartbeats
  ADD COLUMN IF NOT EXISTS machine_id     TEXT,
  ADD COLUMN IF NOT EXISTS cluster_tag    TEXT DEFAULT '',
  ADD COLUMN IF NOT EXISTS location_hint  TEXT DEFAULT '',
  ADD COLUMN IF NOT EXISTS gpu_names      TEXT[];

-- Fast lookup by cluster for the /api/agent/agents endpoint
CREATE INDEX IF NOT EXISTS idx_heartbeats_cluster
  ON agent_heartbeats (user_id, cluster_tag)
  WHERE cluster_tag IS NOT NULL AND cluster_tag != '';

-- Rollback:
-- DROP INDEX IF EXISTS idx_heartbeats_cluster;
-- ALTER TABLE agent_heartbeats
--   DROP COLUMN IF EXISTS machine_id,
--   DROP COLUMN IF EXISTS cluster_tag,
--   DROP COLUMN IF EXISTS location_hint,
--   DROP COLUMN IF EXISTS gpu_names;
