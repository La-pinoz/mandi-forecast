-- One row per job run (backfill, daily ingest, recheck, ...). Every other table points back here
-- through run_id, so any stored row can be traced to the run that produced it.
CREATE TABLE pipeline_runs (
    run_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_type         text NOT NULL,
    -- Information cutoff for jobs that use one (forecasting); NULL for pure ingest runs.
    cutoff_at        timestamptz,
    status           text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at       timestamptz NOT NULL DEFAULT now(),
    finished_at      timestamptz,
    rows_in          integer CHECK (rows_in >= 0),
    rows_out         integer CHECK (rows_out >= 0),
    rows_quarantined integer CHECK (rows_quarantined >= 0),
    error_message    text
);
