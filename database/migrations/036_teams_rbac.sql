-- 036_teams_rbac.sql
-- Multi-tenant teams with role-based access control.
-- Roles: owner (full control), admin (manage members + budgets), viewer (read-only), billing (cost data only).

BEGIN;

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    slug        TEXT NOT NULL UNIQUE,
    owner_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    plan        TEXT NOT NULL DEFAULT 'free'
                CHECK (plan IN ('free', 'pro', 'enterprise')),
    max_members INT NOT NULL DEFAULT 5,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_teams_owner ON teams(owner_id);
CREATE INDEX idx_teams_slug ON teams(slug);

-- Team members (join table with roles)
CREATE TABLE IF NOT EXISTS team_members (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id     UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role        TEXT NOT NULL DEFAULT 'viewer'
                CHECK (role IN ('owner', 'admin', 'viewer', 'billing')),
    invited_by  UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    invited_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    accepted_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One membership per user per team
CREATE UNIQUE INDEX idx_team_members_unique ON team_members(team_id, user_id);
CREATE INDEX idx_team_members_user ON team_members(user_id);
CREATE INDEX idx_team_members_team ON team_members(team_id);

-- Add team_id to gpu_metrics for team-scoped queries
ALTER TABLE gpu_metrics ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES teams(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_gpu_metrics_team ON gpu_metrics(team_id) WHERE team_id IS NOT NULL;

-- Add team_id to budgets for team-scoped budgets
ALTER TABLE budgets ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES teams(id) ON DELETE CASCADE;

-- Add team_id to waste_events
ALTER TABLE waste_events ADD COLUMN IF NOT EXISTS team_id UUID REFERENCES teams(id) ON DELETE CASCADE;

-- Trigger to auto-add owner as team_member on team creation
CREATE OR REPLACE FUNCTION fn_team_auto_add_owner()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO team_members (team_id, user_id, role, accepted_at)
    VALUES (NEW.id, NEW.owner_id, 'owner', now())
    ON CONFLICT (team_id, user_id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_team_auto_add_owner ON teams;
CREATE TRIGGER trg_team_auto_add_owner
    AFTER INSERT ON teams
    FOR EACH ROW
    EXECUTE FUNCTION fn_team_auto_add_owner();

COMMIT;
