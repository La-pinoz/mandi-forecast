"""Backtest folds, information rule, volatility labels and score tables (pure, no database)."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from mandi.evaluate.backtest import (
    availability_table,
    backtest_baselines,
    common_sample,
    known_before,
    make_folds,
    score_table,
    scored_rows,
    volatility_labels,
)
from mandi.models.baselines import NAIVE_MODEL, SEASONAL_NAIVE_MODEL

SENTINEL_DATE = date(2024, 1, 10)
SENTINEL_PRICE = 5000.0


def daily_prices() -> dict[date, float]:
    """Every day of 2023 to 31/01/2024 reported. Before the sentinel date the price alternates 1000/1100 by
    day, so every 7-day pair differs by exactly 100. From the sentinel date on, the price is 5000."""
    prices = {}
    day = date(2023, 1, 1)
    while day <= date(2024, 1, 31):
        if day >= SENTINEL_DATE:
            prices[day] = SENTINEL_PRICE
        elif day.toordinal() % 2 == 0:
            prices[day] = 1000.0
        else:
            prices[day] = 1100.0
        day += timedelta(days=1)
    return prices


def test_folds_are_weekly_and_every_origin_appears_once_with_all_targets_in_the_data():
    folds = make_folds(date(2024, 1, 1), last_data_date=date(2024, 1, 31))

    assert [fold.train_cutoff for fold in folds] == [date(2024, 1, d) for d in (1, 8, 15, 22)]
    all_origins = [origin for fold in folds for origin in fold.origin_dates]
    expected = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(24)]  # 1st to 24th
    assert all_origins == expected
    assert all_origins[-1] + timedelta(days=7) == date(2024, 1, 31)  # last origin's h=7 target is in the data


def test_no_folds_training_data_is_later_than_its_origin():
    prices = daily_prices()
    for fold in make_folds(date(2024, 1, 1), last_data_date=date(2024, 1, 31)):
        training = known_before(prices, fold.train_cutoff)
        assert max(training) < min(fold.origin_dates)
        for origin in fold.origin_dates:
            assert max(known_before(prices, origin)) < origin


def test_forecasts_see_only_prices_before_their_origin():
    prices = daily_prices()
    results = backtest_baselines({1: prices}, make_folds(date(2024, 1, 1), date(2024, 1, 31)))
    available = results[results["point_forecast"].notna()]

    # Up to origin 10/01 (which sees up to 09/01), the 5000 sentinel must never appear in a forecast...
    early = available[available["origin_date"] <= SENTINEL_DATE]
    assert len(early) > 0
    assert (early["point_forecast"] != SENTINEL_PRICE).all()
    # ...and from origin 11/01 naive uses it, so the check above is not vacuous.
    naive_on_11th = available[
        (available["origin_date"] == date(2024, 1, 11)) & (available["model_version"] == NAIVE_MODEL)
    ]
    assert (naive_on_11th["point_forecast"] == SENTINEL_PRICE).all()


def test_mase_scale_comes_from_the_folds_training_period_only():
    prices = daily_prices()
    results = backtest_baselines({1: prices}, make_folds(date(2024, 1, 1), date(2024, 1, 31)))
    scale_by_cutoff = results.groupby("train_cutoff")["mase_scale"].first()

    # Cutoffs 01/01 and 08/01: training ends before the sentinel, every pair differs by 100.
    assert scale_by_cutoff[date(2024, 1, 1)] == 100.0
    assert scale_by_cutoff[date(2024, 1, 8)] == 100.0
    # Cutoff 15/01 includes the jump to 5000, so its scale is larger.
    assert scale_by_cutoff[date(2024, 1, 15)] > 100.0


def test_both_models_get_the_same_origins_and_targets():
    folds = make_folds(date(2024, 1, 1), date(2024, 1, 31))
    results = backtest_baselines({1: daily_prices(), 2: daily_prices()}, folds)

    assert len(results) == 2 * 24 * 7 * 2  # series x origins x horizons x models
    keys = ["series_id", "origin_date", "horizon_days", "target_date"]
    naive_keys = results[results["model_version"] == NAIVE_MODEL][keys].reset_index(drop=True)
    seasonal_keys = results[results["model_version"] == SEASONAL_NAIVE_MODEL][keys].reset_index(drop=True)
    assert naive_keys.equals(seasonal_keys)


def test_seasonal_naive_at_horizon_7_is_unavailable_as_in_the_live_run():
    # Its lag day is the origin itself, which a forecast at origin D can't see (same as the 07:17 live run).
    results = backtest_baselines({1: daily_prices()}, make_folds(date(2024, 1, 1), date(2024, 1, 31)))
    seasonal = results[results["model_version"] == SEASONAL_NAIVE_MODEL]

    assert seasonal[seasonal["horizon_days"] == 7]["point_forecast"].isna().all()
    assert seasonal[seasonal["horizon_days"] < 7]["point_forecast"].notna().all()


def week_of_changes() -> dict[date, float]:
    """Days 1-7 at 100; days 8-14 at 100 * e^(k/10) for k = 1 ... 7, so day 7 + k has a 7-day log change of
    k/10."""
    prices = {}
    for day_number in range(1, 8):
        prices[date(2024, 1, day_number)] = 100.0
    for k in range(1, 8):
        prices[date(2024, 1, 7 + k)] = 100.0 * math.exp(k / 10)
    return prices


def test_volatile_is_the_top_decile_of_7_day_log_changes():
    threshold, labels = volatility_labels(week_of_changes())

    # Changes 0.1 ... 0.7; the 90th percentile (linear interpolation) is 0.6 + 0.4 * 0.1 = 0.64.
    assert threshold == pytest.approx(0.64)
    assert labels[date(2024, 1, 14)] == "volatile"  # change 0.7
    for day_number in range(8, 14):
        assert labels[date(2024, 1, day_number)] == "normal"
    # Days 1-7 have no reported day 7 days earlier: no label.
    assert date(2024, 1, 7) not in labels


def test_a_day_whose_week_before_is_unreported_gets_no_label():
    prices = week_of_changes()
    del prices[date(2024, 1, 3)]
    _, labels = volatility_labels(prices)
    assert date(2024, 1, 10) not in labels
    assert date(2024, 1, 11) in labels


def results_table() -> pd.DataFrame:
    """Two keys x two models. Key A: both scored. Key B: seasonal naive unavailable, naive scored.
    Key C: target unreported for both."""
    rows = [
        # key, model, forecast, actual, scale
        ("A", NAIVE_MODEL, 110.0, 100.0, 10.0),
        ("A", SEASONAL_NAIVE_MODEL, 130.0, 100.0, 10.0),
        ("B", NAIVE_MODEL, 200.0, 150.0, 10.0),
        ("B", SEASONAL_NAIVE_MODEL, float("nan"), 150.0, 10.0),
        ("C", NAIVE_MODEL, 120.0, float("nan"), 10.0),
        ("C", SEASONAL_NAIVE_MODEL, 120.0, float("nan"), 10.0),
    ]
    origins = {"A": date(2024, 1, 1), "B": date(2024, 1, 2), "C": date(2024, 1, 3)}
    records = []
    for key, model, forecast, actual, scale in rows:
        records.append(
            {
                "series_id": 1,
                "origin_date": origins[key],
                "horizon_days": 1,
                "model_version": model,
                "point_forecast": forecast,
                "actual": actual,
                "mase_scale": scale,
            }
        )
    return pd.DataFrame(records)


def test_only_forecasts_with_a_value_and_a_reported_target_are_scored():
    table = score_table(scored_rows(results_table()), []).set_index("model_version")

    # Naive: keys A (error 10) and B (error 50) -> n 2, MAE 30, MASE (1 + 5) / 2 = 3.
    assert table.loc[NAIVE_MODEL, "n"] == 2
    assert table.loc[NAIVE_MODEL, "MAE"] == 30.0
    assert table.loc[NAIVE_MODEL, "MASE"] == 3.0
    # Seasonal naive: only key A (error 30).
    assert table.loc[SEASONAL_NAIVE_MODEL, "n"] == 1
    assert table.loc[SEASONAL_NAIVE_MODEL, "MAE"] == 30.0


def test_common_sample_keeps_only_keys_every_model_scored():
    table = score_table(common_sample(results_table()), []).set_index("model_version")

    assert table.loc[NAIVE_MODEL, "n"] == 1  # key A only
    assert table.loc[NAIVE_MODEL, "MAE"] == 10.0
    assert table.loc[SEASONAL_NAIVE_MODEL, "MAE"] == 30.0


def test_availability_counts_unavailable_and_unreported_separately():
    table = availability_table(results_table(), []).set_index("model_version")

    assert table.loc[NAIVE_MODEL].to_dict() == {
        "forecasts": 3,
        "available": 3,
        "available %": 100.0,
        "target unreported": 1,
        "scored": 2,
    }
    assert table.loc[SEASONAL_NAIVE_MODEL, "available"] == 2
    assert table.loc[SEASONAL_NAIVE_MODEL, "available %"] == pytest.approx(66.7)
    assert table.loc[SEASONAL_NAIVE_MODEL, "scored"] == 1
