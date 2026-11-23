-- One row per VERSION of an observation (blueprint 5.2). Observation key =
-- (source, market, commodity, variety, grade, arrival_date). A new row is added only when a value
-- differs from the latest stored version for that key, so 3000 -> 3200 -> 3000 gives three rows.
-- Only rows that passed validation land here; failures go to quarantine.
-- Prices are assumed to be Rs per quintal (the API states no unit; see docs/M01).
CREATE TABLE price_vintages (
    vintage_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source             text NOT NULL,
    market             text NOT NULL,
    commodity          text NOT NULL,
    variety            text NOT NULL,
    grade              text NOT NULL,
    arrival_date       date NOT NULL,
    market_as_received text NOT NULL,
    district           text NOT NULL,
    min_price          numeric(12, 2) NOT NULL,
    max_price          numeric(12, 2) NOT NULL,
    modal_price        numeric(12, 2) NOT NULL,
    -- Hash of the parsed values; compared with the latest vintage to detect a change.
    row_hash           text NOT NULL,
    -- When OUR pipeline first saw this version (the source has no revision timestamp).
    first_seen_at      timestamptz NOT NULL,
    run_id             bigint NOT NULL REFERENCES pipeline_runs (run_id),
    -- Backstop for the M04 price contracts.
    CHECK (min_price > 0 AND min_price <= modal_price AND modal_price <= max_price),
    -- A retried run can't add the same observation twice.
    UNIQUE (source, market, commodity, variety, grade, arrival_date, run_id)
);

-- As-of path: "latest version of each key with first_seen_at <= cutoff".
CREATE INDEX price_vintages_as_of_idx
    ON price_vintages (source, market, commodity, variety, grade, arrival_date, first_seen_at DESC);

CREATE TRIGGER price_vintages_append_only
    BEFORE UPDATE OR DELETE ON price_vintages
    FOR EACH ROW EXECUTE FUNCTION reject_update_or_delete();
