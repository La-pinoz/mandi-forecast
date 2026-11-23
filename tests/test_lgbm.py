"""LightGBM training and prediction on a small synthetic feature table (no database)."""

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from mandi.features.build import build_feature_table
from mandi.models.lgbm import feature_importance, predict_prices, train

SERIES_INFO = pd.DataFrame(
    {"series_id": [1, 2], "market": ["Barasat", "Asansol"], "crop": ["Tomato", "Onion"]}
)


def growing_table(daily_growth: float) -> pd.DataFrame:
    """Two complete series whose price grows by exactly `daily_growth` (log) every day, for 120 days."""
    prices = {}
    spreads = {}
    for series_id, start in [(1, 2000.0), (2, 1500.0)]:
        prices[series_id] = {}
        spreads[series_id] = {}
        for offset in range(120):
            day = date(2024, 1, 1) + timedelta(days=offset)
            prices[series_id][day] = start * math.exp(daily_growth * offset)
            spreads[series_id][day] = 0.1
    origins = [date(2024, 1, 2) + timedelta(days=offset) for offset in range(110)]
    table = build_feature_table(prices, spreads, SERIES_INFO, origins)
    return table[table["actual"].notna()]


def test_predictions_are_last_price_times_exp_of_the_learned_log_ratio():
    # Origin D's last price is D - 1, so horizon h's target is h + 1 days of growth ahead.
    table = growing_table(daily_growth=0.01)
    models = train(table, per_horizon=True)
    forecasts = predict_prices(models, table)

    expected = table["last_price"] * np.exp(0.01 * (table["horizon_days"] + 1))
    assert forecasts.to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-6)


def test_pooled_layout_uses_one_model_with_the_horizon_as_a_feature():
    models = train(growing_table(0.01), per_horizon=False)
    assert sorted(models) == [1, 2, 3, 4, 5, 6, 7]
    assert len({id(model) for model in models.values()}) == 1
    assert "horizon_days" in models[1].feature_name()


def test_per_horizon_layout_has_seven_models_without_the_horizon_feature():
    table = growing_table(0.01)
    models = train(table, per_horizon=True)
    assert len({id(model) for model in models.values()}) == 7
    assert "horizon_days" not in models[3].feature_name()


def test_training_refuses_rows_without_a_reported_target():
    table = growing_table(0.01).copy()
    table.loc[table.index[0], "target_log_ratio"] = math.nan
    with pytest.raises(ValueError, match="reported target"):
        train(table, per_horizon=False)


def test_feature_importance_shares_add_up_to_100():
    shares = feature_importance(train(growing_table(0.01), per_horizon=False))
    assert shares.sum() == pytest.approx(100)
