-- Scores of live forecasts against reported actuals (blueprint 6.1). Append-only.
--
-- A provisional score is added as soon as the target date has a reported price. Each score stores the actual
-- vintage it used. If that price is later revised (a new vintage), the next scoring run adds another
-- provisional score for the new vintage; the earlier score stays. The current provisional score of a
-- forecast is the one whose actual vintage is in prices_current. Settled scores are added in M09.
CREATE TABLE forecast_scores (
    score_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    forecast_id       bigint NOT NULL REFERENCES forecasts (forecast_id),
    status            text NOT NULL CHECK (status IN ('provisional')),
    actual_vintage_id bigint NOT NULL REFERENCES price_vintages (vintage_id),
    -- The actual's modal price (Rs per quintal), copied from that vintage.
    actual_price      numeric(12, 2) NOT NULL,
    -- point_forecast - actual_price: positive means the forecast was too high.
    error             numeric(12, 2) NOT NULL,
    scored_at         timestamptz NOT NULL DEFAULT now(),
    score_run_id      bigint NOT NULL REFERENCES pipeline_runs (run_id),
    UNIQUE (forecast_id, status, actual_vintage_id)
);

CREATE TRIGGER forecast_scores_append_only
    BEFORE UPDATE OR DELETE ON forecast_scores
    FOR EACH ROW EXECUTE FUNCTION reject_update_or_delete();
