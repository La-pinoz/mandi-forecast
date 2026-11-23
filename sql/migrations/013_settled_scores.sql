-- Settled scores (blueprint 6.1), added in M09.
--
-- A forecast gets ONE settled score, once its target date is older than the settling window (default
-- 7 days, the source's edit window; the daily recheck looks at each arrival date for the last time on
-- target date + 8). The settled score uses the actual's vintage in prices_current at that moment.
-- `settling_days` records the window used, so a later change of the window doesn't change what old
-- settled scores mean. It is NULL for provisional scores.
ALTER TABLE forecast_scores DROP CONSTRAINT forecast_scores_status_check;
ALTER TABLE forecast_scores ADD CONSTRAINT forecast_scores_status_check
    CHECK (status IN ('provisional', 'settled'));
ALTER TABLE forecast_scores ADD COLUMN settling_days integer CHECK (settling_days > 0);
ALTER TABLE forecast_scores ADD CONSTRAINT forecast_scores_window_only_when_settled
    CHECK ((status = 'settled') = (settling_days IS NOT NULL));

CREATE UNIQUE INDEX forecast_scores_one_settled ON forecast_scores (forecast_id) WHERE status = 'settled';

-- A settled actual that the source revised afterwards. The settled score is kept as it is; the revision
-- is recorded here as an event (one row per settled score and revised vintage). Append-only.
CREATE TABLE settled_score_revisions (
    revision_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    score_id           bigint NOT NULL REFERENCES forecast_scores (score_id),
    -- The vintage now in prices_current, which differs from the settled score's actual_vintage_id.
    revised_vintage_id bigint NOT NULL REFERENCES price_vintages (vintage_id),
    revised_price      numeric(12, 2) NOT NULL,
    -- point_forecast - revised_price: what the error would be against the revised actual.
    revised_error      numeric(12, 2) NOT NULL,
    detected_at        timestamptz NOT NULL DEFAULT now(),
    run_id             bigint NOT NULL REFERENCES pipeline_runs (run_id),
    UNIQUE (score_id, revised_vintage_id)
);

CREATE TRIGGER settled_score_revisions_append_only
    BEFORE UPDATE OR DELETE ON settled_score_revisions
    FOR EACH ROW EXECUTE FUNCTION reject_update_or_delete();
