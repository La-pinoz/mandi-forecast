"""The two baseline forecasts (blueprint 7.3). Pure functions: no database, so M07's backtest can reuse them.

Inputs are one series' modal prices as a dict {arrival_date: price}, holding only what was known at the
information cutoff. A date missing from the dict is unreported. Nothing is filled in.

    naive            the latest reported price on or before the origin date, for every horizon
    seasonal naive   the price reported exactly 7 calendar days before the target date; if that day is
                     unreported the forecast is unavailable. No other day is used in its place.
"""

from dataclasses import dataclass
from datetime import date, timedelta

NAIVE_MODEL = "naive.v1"
SEASONAL_NAIVE_MODEL = "seasonal_naive.v1"
HORIZONS_DAYS = range(1, 8)  # 1-7 calendar days ahead
SEASON_DAYS = 7


@dataclass
class BaselineForecast:
    """One forecast for one target date. `point_forecast` is None when the model is unavailable."""

    model_version: str
    horizon_days: int
    target_date: date
    point_forecast: float | None
    unavailable_reason: str | None


def naive_forecast(prices: dict[date, float], origin_date: date) -> float | None:
    """The latest price on or before `origin_date`, or None if there is none.

    Dates after the origin are ignored, so a price that should not be known yet can never be used.
    """
    known_dates = [day for day in prices if day <= origin_date]
    if not known_dates:
        return None
    return prices[max(known_dates)]


def seasonal_naive_forecast(prices: dict[date, float], origin_date: date, target_date: date) -> float | None:
    """The price exactly 7 calendar days before `target_date`, or None if that day is unreported.

    Raises ValueError if that day is after the origin (a horizon over 7 days): it would use the future.
    """
    season_date = target_date - timedelta(days=SEASON_DAYS)
    if season_date > origin_date:
        raise ValueError(f"target {target_date} is more than {SEASON_DAYS} days after origin {origin_date}")
    return prices.get(season_date)


def baseline_forecasts(prices: dict[date, float], origin_date: date) -> list[BaselineForecast]:
    """Naive and seasonal-naive forecasts for horizons 1-7 from `origin_date` (14 forecasts)."""
    forecasts = []
    naive_value = naive_forecast(prices, origin_date)
    for horizon in HORIZONS_DAYS:
        target_date = origin_date + timedelta(days=horizon)

        forecasts.append(
            BaselineForecast(
                model_version=NAIVE_MODEL,
                horizon_days=horizon,
                target_date=target_date,
                point_forecast=naive_value,
                unavailable_reason=None if naive_value is not None else "no_price_on_or_before_origin",
            )
        )

        seasonal_value = seasonal_naive_forecast(prices, origin_date, target_date)
        forecasts.append(
            BaselineForecast(
                model_version=SEASONAL_NAIVE_MODEL,
                horizon_days=horizon,
                target_date=target_date,
                point_forecast=seasonal_value,
                unavailable_reason=None if seasonal_value is not None else "target_minus_7_days_unreported",
            )
        )
    return forecasts
