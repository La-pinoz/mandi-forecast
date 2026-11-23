-- Trained LightGBM models and which one is the champion (blueprint 10, 13). One row per model version.
--
-- The model itself is stored here, not in a file: the daily GitHub Actions job can always reach the
-- database, and the M13 weekly retrain can add a version without committing to the repository.
-- `models` maps each horizon ("1".."7") to that horizon's LightGBM model in LightGBM's own text format
-- (Booster.model_to_string), which also holds the categories of the categorical features.
--
-- Not append-only: M13 sets retired_at on the old champion when it promotes a new one. The model,
-- its parameters and its features are never changed after insert.
CREATE TABLE model_registry (
    model_version       text PRIMARY KEY,
    -- LightGBM parameters, number of boosting rounds and layout ('per_horizon' or 'pooled').
    params              jsonb NOT NULL,
    -- Feature names in the order the model expects them.
    features            jsonb NOT NULL,
    -- Rolling-origin backtest metrics of this model's method (M08), with their sample counts.
    backtest_metrics    jsonb NOT NULL,
    models              jsonb NOT NULL,
    -- The model was trained on prices as known at this moment (as-of query on price_vintages).
    training_data_as_of timestamptz NOT NULL,
    training_rows       integer NOT NULL CHECK (training_rows > 0),
    trained_at          timestamptz NOT NULL DEFAULT now(),
    promoted_at         timestamptz,
    retired_at          timestamptz,
    run_id              bigint NOT NULL REFERENCES pipeline_runs (run_id),
    CHECK (retired_at IS NULL OR promoted_at IS NOT NULL)
);

-- At most one champion: a promoted model that is not retired. The daily run forecasts with it.
CREATE UNIQUE INDEX model_registry_one_champion
    ON model_registry ((true))
    WHERE promoted_at IS NOT NULL AND retired_at IS NULL;
