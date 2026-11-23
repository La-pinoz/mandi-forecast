"""LightGBM features (pure, no database): the leakage rule and missing values."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from mandi.features.build import FEATURES, build_feature_table, origin_features

SERIES_INFO = pd.DataFrame({"series_id": [1], "market": ["Barasat"], "crop": ["Tomato"]})


def history() -> tuple[dict[date, float], dict[date, float]]:
    """60 days from 01/01/2024 with a different price every day and a few unreported days."""
    prices = {}
    spreads = {}
    for offset in range(60):
        day = date(2024, 1, 1) + timedelta(days=offset)
        if offset % 9 == 4:  # unreported: the 5th, 14th, 23rd, ...
            continue
        prices[day] = 1000.0 + 13 * offset + (offset % 5) * 40
        spreads[day] = 0.1 + offset / 1000
    return prices, spreads


def before(values: dict[date, float], day: date) -> dict[date, float]:
    return {key: value for key, value in values.items() if key < day}


def test_feature_row_for_origin_t_never_uses_data_on_or_after_t_whatever_the_horizon():
    """Milestone leakage test: deleting every price from day T on must not change origin T's features."""
    prices, spreads = history()
    origin = date(2024, 2, 11)
    assert origin in prices  # day T itself is reported, so there is something on T that could leak
    full = build_feature_table({1: prices}, {1: spreads}, SERIES_INFO, [origin])
    truncated = build_feature_table(
        {1: before(prices, origin)}, {1: before(spreads, origin)}, SERIES_INFO, [origin]
    )

    assert list(full["horizon_days"]) == [1, 2, 3, 4, 5, 6, 7]
    pd.testing.assert_frame_equal(full[FEATURES], truncated[FEATURES])
    # The labels do differ (they are the future), so the comparison above is not vacuous.
    assert full["actual"].notna().any()
    assert truncated["actual"].isna().all()


def test_a_shock_on_the_origin_day_does_not_reach_the_features():
    prices, spreads = history()
    origin = date(2024, 2, 10)
    calm = origin_features(prices, spreads, origin)
    prices[origin] = 99_999.0  # reported on the origin day itself: not known at 07:17 IST
    spreads[origin] = 5.0
    assert origin_features(prices, spreads, origin) == pytest.approx(calm, nan_ok=True)


def test_missing_lags_stay_nan_and_are_not_filled_from_neighbours():
    prices, spreads = history()
    origin = date(2024, 2, 10)
    del prices[origin - timedelta(days=7)]
    del spreads[origin - timedelta(days=7)]

    features = origin_features(prices, spreads, origin)

    assert math.isnan(features["lag_7"])
    last_price = prices[origin - timedelta(days=1)]
    assert features["lag_14"] == pytest.approx(math.log(prices[origin - timedelta(days=14)] / last_price))


def test_relative_lags_staleness_and_windows_by_hand():
    origin = date(2024, 3, 10)
    prices = {
        date(2024, 3, 2): 800.0,  # 8 days before: outside the 7-day window
        date(2024, 3, 3): 1000.0,  # 7 days before
        date(2024, 3, 7): 2000.0,  # 3 days before: the last price (8th and 9th unreported)
    }
    spreads = {day: 0.2 for day in prices}

    features = origin_features(prices, spreads, origin)

    assert features["last_price"] == 2000.0
    assert features["days_since_last"] == 3
    assert math.isnan(features["lag_1"]) and math.isnan(features["lag_2"])
    assert features["lag_3"] == 0.0
    assert features["lag_7"] == pytest.approx(math.log(0.5))
    assert features["count_7"] == 2
    assert features["mean_7"] == pytest.approx(math.log(0.5) / 2)
    assert features["min_7"] == pytest.approx(math.log(0.5))
    assert features["max_7"] == 0.0
    assert features["count_14"] == 3
    assert features["spread_last"] == 0.2


def test_std_needs_two_observations_and_an_empty_window_is_nan():
    origin = date(2024, 3, 10)
    prices = {date(2024, 2, 20): 1000.0}  # 19 days before: only in the 28-day window
    spreads = {date(2024, 2, 20): 0.1}

    features = origin_features(prices, spreads, origin)

    assert features["count_7"] == 0
    assert math.isnan(features["mean_7"]) and math.isnan(features["spread_mean_7"])
    assert features["count_28"] == 1
    assert math.isnan(features["std_28"])
    assert features["days_since_last"] == 19


def test_no_row_for_an_origin_without_any_earlier_price():
    prices, spreads = history()
    first_day = min(prices)
    table = build_feature_table({1: prices}, {1: spreads}, SERIES_INFO, [first_day, first_day + timedelta(1)])
    assert set(table["origin_date"]) == {first_day + timedelta(1)}


def test_target_is_the_log_ratio_to_the_last_price_and_nan_when_unreported():
    prices, spreads = history()
    origin = date(2024, 1, 12)  # 14/01 (horizon 2) is unreported
    table = build_feature_table({1: prices}, {1: spreads}, SERIES_INFO, [origin]).set_index("horizon_days")

    assert math.isnan(table.loc[2, "actual"]) and math.isnan(table.loc[2, "target_log_ratio"])
    expected = math.log(prices[date(2024, 1, 13)] / prices[date(2024, 1, 11)])
    assert table.loc[1, "target_log_ratio"] == pytest.approx(expected)
    assert table.loc[1, "target_day_of_week"] == date(2024, 1, 13).weekday()


def test_categories_are_every_series_value_even_if_one_series_has_no_rows():
    prices, spreads = history()
    info = pd.DataFrame({"series_id": [1, 2], "market": ["Barasat", "Habra"], "crop": ["Tomato", "Tomato"]})
    table = build_feature_table({1: prices, 2: {}}, {1: spreads, 2: {}}, info, [date(2024, 2, 1)])
    assert list(table["market"].cat.categories) == ["Barasat", "Habra"]
    assert list(table["series_id"].cat.categories) == [1, 2]
