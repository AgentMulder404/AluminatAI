BEGIN;

CREATE TABLE IF NOT EXISTS prospects (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name    TEXT NOT NULL,
    company_url     TEXT,
    linkedin_url    TEXT,
    domain          TEXT,
    industry        TEXT,
    company_size    TEXT,
    employee_count  INTEGER,
    location        TEXT,
    description     TEXT DEFAULT '',
    contact_name    TEXT,
    contact_email   TEXT,
    contact_title   TEXT,
    contact_phone   TEXT,
    contact_linkedin TEXT,
    email_verified  BOOLEAN DEFAULT FALSE,
    email_status    TEXT,
    source          TEXT NOT NULL DEFAULT 'apify',
    source_query    TEXT,
    category        TEXT,
    status          TEXT NOT NULL DEFAULT 'new'
                    CHECK (status IN ('new','contacted','replied','demo_booked','customer','not_interested')),
    notes           TEXT DEFAULT '',
    discovered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      UUID REFERENCES auth.users(id) ON DELETE SET NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_prospects_domain_email
    ON prospects(domain, contact_email) WHERE domain IS NOT NULL AND contact_email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prospects_status ON prospects(status);
CREATE INDEX IF NOT EXISTS idx_prospects_category ON prospects(category);
CREATE INDEX IF NOT EXISTS idx_prospects_discovered ON prospects(discovered_at DESC);

COMMIT;
