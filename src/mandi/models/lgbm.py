"""M08: global LightGBM point forecast (P50), one model across all series (blueprint 7.3-7.4).

The model predicts `target_log_ratio` = log(price(target) / p_last) from the features in
`mandi.features.build`; `predict_prices` turns that back into a price, p_last * exp(prediction).

Two layouts, compared in the backtest (M08 decision, see docs/M08-lightgbm.md):
    pooled       one model for all horizons, with `horizon_days` as a feature
    per horizon  seven models, one trained on each horizon's rows, without `horizon_days`
Both are returned as {horizon: model}; for pooled, every horizon maps to the same model. So prediction is
always "use models[h] for the rows of horizon h".

The parameters are fixed in advance (owner decision, 2026-11-23): no search. The L1 objective fits the
median, which is the P50 forecast and the loss that MAE rewards.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from mandi.features.build import CATEGORICAL_FEATURES, FEATURES
from mandi.models.baselines import HORIZONS_DAYS

MODEL_VERSION = "lgbm.v1"
PARAMS = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "seed": 42,
    "deterministic": True,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 300
FEATURES_PER_HORIZON = [feature for feature in FEATURES if feature != "horizon_days"]


def fit_one(rows: pd.DataFrame, features: list[str]) -> lgb.Booster:
    """Train one LightGBM on `rows` (which must all have a reported target) with the given features."""
    if rows["target_log_ratio"].isna().any():
        raise ValueError("training rows must all have a reported target")
    dataset = lgb.Dataset(
        rows[features], label=rows["target_log_ratio"], categorical_feature=CATEGORICAL_FEATURES
    )
    return lgb.train(PARAMS, dataset, num_boost_round=NUM_BOOST_ROUND)


def train(training_rows: pd.DataFrame, per_horizon: bool) -> dict[int, lgb.Booster]:
    """Train the pooled model or the seven per-horizon models. Returns {horizon: model}."""
    if not per_horizon:
        pooled = fit_one(training_rows, FEATURES)
        return {horizon: pooled for horizon in HORIZONS_DAYS}

    models = {}
    for horizon in HORIZONS_DAYS:
        horizon_rows = training_rows[training_rows["horizon_days"] == horizon]
        models[horizon] = fit_one(horizon_rows, FEATURES_PER_HORIZON)
    return models


def predict_prices(models: dict[int, lgb.Booster], rows: pd.DataFrame) -> pd.Series:
    """Price forecasts for `rows` (same index): last_price * exp(predicted log ratio)."""
    forecasts = pd.Series(np.nan, index=rows.index)
    for horizon, model in models.items():
        selected = rows["horizon_days"] == horizon
        if not selected.any():
            continue
        horizon_rows = rows[selected]
        log_ratios = model.predict(horizon_rows[model.feature_name()])
        forecasts[selected] = horizon_rows["last_price"].to_numpy() * np.exp(log_ratios)
    return forecasts


def feature_importance(models: dict[int, lgb.Booster]) -> pd.Series:
    """Each feature's share (%) of the total split gain, summed over the seven horizons' models.

    For the pooled layout the same model is counted seven times, which leaves the shares unchanged.
    """
    total = pd.Series(dtype=float)
    for model in models.values():
        gains = pd.Series(model.feature_importance(importance_type="gain"), index=model.feature_name())
        total = total.add(gains, fill_value=0.0)
    return (100 * total / total.sum()).sort_values(ascending=False)
