-- 040_subscriptions.sql
-- Stripe billing integration: customer + subscription columns on users,
-- subscription history table for audit trail.

BEGIN;

-- Add Stripe columns to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'free'
    CHECK (plan IN ('free', 'pro', 'enterprise'));
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_period_end TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_cancel_at_period_end BOOLEAN DEFAULT FALSE;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_stripe_customer
    ON users(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_plan ON users(plan);

-- Subscription event history (populated by Stripe webhooks)
CREATE TABLE IF NOT EXISTS subscription_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    stripe_event_id TEXT NOT NULL UNIQUE,  -- idempotency key
    event_type      TEXT NOT NULL,         -- e.g. 'checkout.session.completed', 'invoice.paid'
    plan            TEXT,
    amount_usd      NUMERIC(10,2),
    currency        TEXT DEFAULT 'usd',
    period_start    TIMESTAMPTZ,
    period_end      TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_sub_events_user ON subscription_events(user_id);
CREATE INDEX idx_sub_events_type ON subscription_events(event_type);

COMMIT;
