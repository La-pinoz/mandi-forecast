"""Point-forecast metrics (blueprint 8.2): MAE, sMAPE and MASE.

Every function takes forecasts and actuals of the same rows (already filtered to forecasts that have a value
and targets that were reported). Prices are positive (M04 contracts), so no denominator can be zero.
"""

from datetime import date, timedelta

import pandas as pd

from mandi.models.baselines import SEASON_DAYS


def mean_absolute_error(forecasts: pd.Series, actuals: pd.Series) -> float:
    """Mean of |forecast - actual|, in rupees per quintal."""
    return float((forecasts - actuals).abs().mean())


def smape(forecasts: pd.Series, actuals: pd.Series) -> float:
    """Symmetric MAPE in percent: mean of 200 * |F - A| / (|F| + |A|). Ranges from 0 to 200."""
    ratio = 200 * (forecasts - actuals).abs() / (forecasts.abs() + actuals.abs())
    return float(ratio.mean())


def seasonal_naive_scale(prices: dict[date, float], cutoff_date: date) -> float | None:
    """The MASE scale for one series: the seasonal-naive MAE in the training period.

    Mean of |price(t) - price(t - 7 days)| over days t before `cutoff_date` where **both** t and t - 7
    calendar days are reported. A pair with an unreported day is skipped; no neighbouring day is used in its
    place. Returns None if there is no such pair.
    """
    differences = []
    for day, price in prices.items():
        if day >= cutoff_date:
            continue
        week_before = day - timedelta(days=SEASON_DAYS)
        if week_before in prices:
            differences.append(abs(price - prices[week_before]))
    if not differences:
        return None
    return sum(differences) / len(differences)


def mase(forecasts: pd.Series, actuals: pd.Series, scales: pd.Series) -> float:
    """Mean absolute scaled error: mean of |F - A| / scale, each row divided by its own series' scale.

    Dividing row by row (not one MAE by one scale) lets rows from different series, with different price
    levels, be averaged together. Raises ValueError if a scale is missing or zero, instead of silently
    averaging over fewer rows.
    """
    if scales.isna().any() or (scales == 0).any():
        raise ValueError("every row needs a positive MASE scale")
    return float(((forecasts - actuals).abs() / scales).mean())
