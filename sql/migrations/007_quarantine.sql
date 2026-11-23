-- Source rows that failed a hard data contract (blueprint 5.4), kept with the reason. A row that
-- fails several rules gets one row per reason.
CREATE TABLE quarantine (
    run_id         bigint NOT NULL REFERENCES pipeline_runs (run_id),
    -- SHA-256 of the source record as received.
    row_hash       text NOT NULL,
    source         text NOT NULL,
    record         jsonb NOT NULL,
    -- e.g. 'non_positive_price', 'min_above_modal', 'conflicting_duplicate' (codes defined in M04).
    reason_code    text NOT NULL,
    detail         text,
    quarantined_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, row_hash, reason_code)
);
