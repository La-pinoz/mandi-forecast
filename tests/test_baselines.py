"""Naive and seasonal-naive baselines (pure functions, no database)."""

from datetime import date, timedelta

import pytest

from mandi.models.baselines import (
    NAIVE_MODEL,
    SEASONAL_NAIVE_MODEL,
    baseline_forecasts,
    naive_forecast,
    seasonal_naive_forecast,
)

ORIGIN = date(2026, 11, 21)

# 16/11 and 20/11 are unreported (no entry). 22/11 is after the origin: it must never be used.
PRICES = {
    date(2026, 11, 14): 3000.0,
    date(2026, 11, 15): 3100.0,
    date(2026, 11, 17): 3300.0,
    date(2026, 11, 18): 3400.0,
    date(2026, 11, 19): 3500.0,
    date(2026, 11, 22): 9999.0,
}


def test_naive_is_the_latest_price_on_or_before_the_origin():
    # 20/11 is unreported, so the latest known price is 19/11; 22/11 is after the origin and ignored.
    assert naive_forecast(PRICES, ORIGIN) == 3500.0


def test_naive_uses_a_price_on_the_origin_date_itself():
    prices = {date(2026, 11, 19): 3500.0, ORIGIN: 3650.0}
    assert naive_forecast(prices, ORIGIN) == 3650.0


def test_naive_is_unavailable_without_history():
    assert naive_forecast({}, ORIGIN) is None
    assert naive_forecast({date(2026, 11, 22): 3000.0}, ORIGIN) is None


def test_seasonal_naive_uses_exactly_seven_calendar_days_before_the_target():
    assert seasonal_naive_forecast(PRICES, ORIGIN, target_date=date(2026, 11, 22)) == 3100.0  # 15/11
    assert seasonal_naive_forecast(PRICES, ORIGIN, target_date=date(2026, 11, 26)) == 3500.0  # 19/11


def test_seasonal_naive_is_unavailable_when_target_minus_7_is_unreported():
    # Target 23/11 -> 16/11, unreported. 15/11 and 17/11 are reported but must not be used instead.
    assert seasonal_naive_forecast(PRICES, ORIGIN, target_date=date(2026, 11, 23)) is None


def test_seasonal_naive_refuses_a_horizon_beyond_seven_days():
    with pytest.raises(ValueError, match="more than 7 days"):
        seasonal_naive_forecast(PRICES, ORIGIN, target_date=ORIGIN + timedelta(days=8))


def test_baseline_forecasts_cover_horizons_1_to_7_in_calendar_days():
    forecasts = baseline_forecasts(PRICES, ORIGIN)

    assert len(forecasts) == 14
    for forecast in forecasts:
        assert forecast.target_date == ORIGIN + timedelta(days=forecast.horizon_days)

    naive = {f.horizon_days: f.point_forecast for f in forecasts if f.model_version == NAIVE_MODEL}
    assert naive == {1: 3500.0, 2: 3500.0, 3: 3500.0, 4: 3500.0, 5: 3500.0, 6: 3500.0, 7: 3500.0}

    seasonal = {f.horizon_days: f for f in forecasts if f.model_version == SEASONAL_NAIVE_MODEL}
    values = {horizon: forecast.point_forecast for horizon, forecast in seasonal.items()}
    # h=2 -> 16/11 unreported; h=6 -> 20/11 unreported; h=7 -> 21/11 (the origin) unreported.
    assert values == {1: 3100.0, 2: None, 3: 3300.0, 4: 3400.0, 5: 3500.0, 6: None, 7: None}
    assert seasonal[2].unavailable_reason == "target_minus_7_days_unreported"
    assert seasonal[1].unavailable_reason is None


def test_baseline_forecasts_without_history_are_all_unavailable():
    forecasts = baseline_forecasts({}, ORIGIN)

    assert len(forecasts) == 14
    for forecast in forecasts:
        assert forecast.point_forecast is None
        assert forecast.unavailable_reason is not None
