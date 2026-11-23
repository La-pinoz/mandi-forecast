-- Live forecasts (blueprint 6, 7, 10), one row per series, origin day, horizon and model. Never updated:
-- live scoring is only honest if a forecast can't be edited after its actual is known.
--
-- The origin is the IST calendar date of the run's information cutoff (input_as_of_at); the target is
-- origin_date + horizon_days CALENDAR days. A baseline that can't be computed (e.g. seasonal naive when the
-- day 7 days before the target is unreported) is still stored, with point_forecast NULL and a reason, so its
-- availability rate can be reported.
CREATE TABLE forecasts (
    forecast_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    series_id           integer NOT NULL REFERENCES series (series_id),
    origin_date         date NOT NULL,
    horizon_days        integer NOT NULL CHECK (horizon_days BETWEEN 1 AND 7),
    target_date         date NOT NULL,
    -- e.g. 'naive.v1', 'seasonal_naive.v1'.
    model_version       text NOT NULL,
    -- Modal price forecast, Rs per quintal. NULL when the model is unavailable for this target.
    point_forecast      numeric(12, 2),
    unavailable_reason  text,
    -- Information cutoff: the inputs are the price vintages with first_seen_at <= this moment.
    input_as_of_at      timestamptz NOT NULL,
    forecast_created_at timestamptz NOT NULL DEFAULT now(),
    forecast_run_id     bigint NOT NULL REFERENCES pipeline_runs (run_id),
    CHECK (target_date = origin_date + horizon_days),
    CHECK ((point_forecast IS NULL) = (unavailable_reason IS NOT NULL)),
    -- One forecast per origin day: a re-run on the same day can't add a second, later-informed forecast.
    UNIQUE (series_id, origin_date, horizon_days, model_version)
);

CREATE INDEX forecasts_target_idx ON forecasts (series_id, target_date);

CREATE TRIGGER forecasts_append_only
    BEFORE UPDATE OR DELETE ON forecasts
    FOR EACH ROW EXECUTE FUNCTION reject_update_or_delete();
