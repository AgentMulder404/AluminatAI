-- Add quantization variant analysis to model registry.

ALTER TABLE model_registry
    ADD COLUMN IF NOT EXISTS quantization_variants JSONB NOT NULL DEFAULT '[]';

COMMENT ON COLUMN model_registry.quantization_variants IS
    'Precomputed quantization variant analysis: [{variant, precision, memory_reduction_pct, quality_impact, best_gpu}]';
