-- Every API response as received, one row per response (blueprint 5.2). Append-only: lets us
-- re-parse history after a parsing bug without downloading it again.

-- Shared by all append-only tables: any UPDATE or DELETE raises an error.
-- (TRUNCATE is not blocked; a deliberate full rebuild stays possible.)
CREATE FUNCTION reject_update_or_delete() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only: % is not allowed', TG_TABLE_NAME, TG_OP;
END;
$$;

CREATE TABLE raw_snapshots (
    run_id         bigint NOT NULL REFERENCES pipeline_runs (run_id),
    -- SHA-256 of the response body; the same response stored twice in one run is one row.
    payload_hash   text NOT NULL,
    -- data.gov.in resource id, e.g. 35985678-0d79-46b4-9ed6-6f13308a1d24 (variety-wise prices).
    source         text NOT NULL,
    -- Query filters and paging used, WITHOUT the API key.
    request_params jsonb NOT NULL,
    -- Full JSON response. jsonb keeps every value and its type (prices as "1800" vs 1800) but not
    -- key order or whitespace.
    payload        jsonb NOT NULL,
    -- When our pipeline received it (for the M01 files: their saved fetch time).
    ingested_at    timestamptz NOT NULL,
    PRIMARY KEY (run_id, payload_hash)
);

CREATE TRIGGER raw_snapshots_append_only
    BEFORE UPDATE OR DELETE ON raw_snapshots
    FOR EACH ROW EXECUTE FUNCTION reject_update_or_delete();
