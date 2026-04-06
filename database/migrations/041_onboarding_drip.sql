-- Migration 041: Onboarding email drip tracking
-- Tracks how many onboarding emails a user has received (0-4 = 5 emails total)

BEGIN;

ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_drip_sent INT NOT NULL DEFAULT 0;

-- Partial index: only index users who haven't completed the drip sequence
CREATE INDEX IF NOT EXISTS idx_users_drip_pending
  ON users(onboarding_drip_sent, created_at)
  WHERE onboarding_drip_sent < 5;

COMMIT;
