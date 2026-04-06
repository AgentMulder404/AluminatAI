-- Migration 043: Organizations / workspace support
-- Adds org layer above teams for billing and access control.

BEGIN;

CREATE TABLE IF NOT EXISTS organizations (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                        TEXT NOT NULL,
    slug                        TEXT NOT NULL UNIQUE,
    owner_id                    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    stripe_customer_id          TEXT,
    plan                        TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise')),
    plan_period_end             TIMESTAMPTZ,
    plan_cancel_at_period_end   BOOLEAN DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_orgs_slug ON organizations(slug);
CREATE INDEX idx_orgs_owner ON organizations(owner_id);

-- Link teams to orgs
ALTER TABLE teams ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id);

-- Org membership (distinct from team membership)
CREATE TABLE IF NOT EXISTS org_members (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role        TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member')),
    joined_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_org_members_unique ON org_members(org_id, user_id);
CREATE INDEX idx_org_members_user ON org_members(user_id);

COMMIT;
