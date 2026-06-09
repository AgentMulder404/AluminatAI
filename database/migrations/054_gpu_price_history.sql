-- GPU Price History: tracks GPU cloud pricing over time for trend analysis.

CREATE TABLE IF NOT EXISTS gpu_price_history (
    id                UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    gpu_model         TEXT          NOT NULL,
    provider          TEXT          NOT NULL,
    instance_type     TEXT,
    on_demand_rate    NUMERIC(10,4) NOT NULL,
    spot_rate         NUMERIC(10,4),
    region            TEXT          DEFAULT 'us-east-1',
    snapshot_date     DATE          NOT NULL DEFAULT CURRENT_DATE,
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gpu_price_history_model_date
    ON gpu_price_history (gpu_model, snapshot_date DESC);

CREATE INDEX IF NOT EXISTS idx_gpu_price_history_date
    ON gpu_price_history (snapshot_date DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_price_history_dedup
    ON gpu_price_history (gpu_model, provider, COALESCE(instance_type, ''), snapshot_date);

-- Price-performance alerts for user notifications.
CREATE TABLE IF NOT EXISTS price_performance_alerts (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    gpu_model     TEXT        NOT NULL,
    provider      TEXT,
    alert_type    TEXT        NOT NULL
                  CHECK (alert_type IN ('new_best_value', 'price_drop', 'spot_opportunity')),
    message       TEXT        NOT NULL,
    model_family  TEXT,
    previous_rate NUMERIC(10,4),
    current_rate  NUMERIC(10,4) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_price_alerts_created
    ON price_performance_alerts (created_at DESC);
