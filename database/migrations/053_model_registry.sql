-- Model Intelligence Pipeline: registry for auto-discovered AI model profiles.
-- Stores model metadata, roofline-derived profiles, and precomputed GPU rankings.

CREATE TABLE IF NOT EXISTS model_registry (
    id                UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id          TEXT          NOT NULL UNIQUE,
    tag               TEXT          NOT NULL UNIQUE,
    family            TEXT          NOT NULL,
    source            TEXT          NOT NULL DEFAULT 'huggingface',

    -- ModelProfile fields (mirrors gpu_specs.ModelProfile)
    math_intensity    NUMERIC(8,2)  NOT NULL,
    precision         TEXT          NOT NULL DEFAULT 'bf16',
    is_memory_bound   BOOLEAN       NOT NULL DEFAULT false,
    typical_util_min  INT           NOT NULL DEFAULT 50,
    typical_util_max  INT           NOT NULL DEFAULT 80,

    -- HuggingFace metadata
    parameter_count   BIGINT,
    architecture      TEXT,
    library           TEXT,
    license           TEXT,
    downloads_30d     BIGINT        DEFAULT 0,
    trending_score    NUMERIC(8,4)  DEFAULT 0,

    -- Pipeline metadata
    profiled_at       TIMESTAMPTZ,
    estimated_at      TIMESTAMPTZ,
    hf_metadata       JSONB         NOT NULL DEFAULT '{}',
    gpu_rankings      JSONB         NOT NULL DEFAULT '[]',

    status            TEXT          NOT NULL DEFAULT 'detected'
                      CHECK (status IN ('detected', 'profiled', 'estimated', 'active', 'archived')),
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_registry_tag
    ON model_registry (tag);

CREATE INDEX IF NOT EXISTS idx_model_registry_family
    ON model_registry (family);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON model_registry (status);

CREATE INDEX IF NOT EXISTS idx_model_registry_created
    ON model_registry (created_at DESC);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_model_registry_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_model_registry_updated
    BEFORE UPDATE ON model_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_model_registry_timestamp();

-- RLS: public reads, service-role writes
ALTER TABLE model_registry ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read model registry"
    ON model_registry FOR SELECT
    USING (true);

CREATE POLICY "Service role can manage model registry"
    ON model_registry FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
