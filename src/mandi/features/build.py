"""M08: the LightGBM training/forecast table: one row per (series, origin, horizon) (blueprint 7.4).

The information rule is the same as the M07 backtest and the live run at 07:17 IST: a forecast at origin D
sees only prices with arrival date **before** D. Every feature of a row is computed from those prices only,
whatever the horizon. The target date is used for two things only: its calendar (day of week, month, known
in advance) and the label (`actual`, `target_log_ratio`), which is what the model learns, never an input.

Prices are expressed relative to the last known price p_last (the price naive would forecast):
    lag_k            log(price(D - k) / p_last), NaN if day D - k is unreported (nothing is filled in)
    mean/std/min/max_w   of log(price / p_last) over the reported days in [D - w, D - 1]
    target_log_ratio log(price(target) / p_last), what the model predicts; forecast = p_last * exp(prediction)
So a prediction of 0 is exactly the naive forecast, and tomato and onion, at different price levels, share
one scale. `log_last_price` keeps the level itself as a feature.
"""

import math
from datetime import date, timedelta

import pandas as pd

from mandi.models.baselines import HORIZONS_DAYS

LAG_DAYS = (1, 2, 3, 7, 14, 28)
WINDOW_DAYS = (7, 14, 28)

ORIGIN_FEATURES = ["log_last_price", "days_since_last"]
ORIGIN_FEATURES += [f"lag_{lag}" for lag in LAG_DAYS]
for _window in WINDOW_DAYS:
    ORIGIN_FEATURES += [
        f"count_{_window}",
        f"mean_{_window}",
        f"std_{_window}",
        f"min_{_window}",
        f"max_{_window}",
    ]
ORIGIN_FEATURES += ["spread_last", "spread_mean_7"]

CATEGORICAL_FEATURES = ["series_id", "crop", "market"]
FEATURES = [*ORIGIN_FEATURES, "horizon_days", "target_day_of_week", "target_month", *CATEGORICAL_FEATURES]


def origin_features(
    prices: dict[date, float], spreads: dict[date, float], origin: date
) -> dict[str, float] | None:
    """Features of one series at `origin`, from arrival dates strictly before it. None if there are none.

    `prices` is {arrival_date: modal_price}; `spreads` is {arrival_date: (max - min) / modal} for the same
    days. The result also holds `last_price` (not a model feature: it turns predictions back into prices).
    """
    known_dates = [day for day in prices if day < origin]
    if not known_dates:
        return None
    last_date = max(known_dates)
    last_price = prices[last_date]

    features = {
        "last_price": last_price,
        "log_last_price": math.log(last_price),
        "days_since_last": (origin - last_date).days,
        "spread_last": spreads[last_date],
    }

    for lag in LAG_DAYS:
        day = origin - timedelta(days=lag)
        if day in prices:
            features[f"lag_{lag}"] = math.log(prices[day] / last_price)
        else:
            features[f"lag_{lag}"] = math.nan

    # Fixed calendar windows: a window counts the reported days in it; unreported days are simply absent.
    for window in WINDOW_DAYS:
        log_ratios = []
        for offset in range(1, window + 1):
            day = origin - timedelta(days=offset)
            if day in prices:
                log_ratios.append(math.log(prices[day] / last_price))
        values = pd.Series(log_ratios, dtype=float)  # mean/min/max of no values are NaN; std needs 2 values
        features[f"count_{window}"] = len(log_ratios)
        features[f"mean_{window}"] = values.mean()
        features[f"std_{window}"] = values.std()
        features[f"min_{window}"] = values.min()
        features[f"max_{window}"] = values.max()

    week_spreads = []
    for offset in range(1, 8):
        day = origin - timedelta(days=offset)
        if day in spreads:
            week_spreads.append(spreads[day])
    features["spread_mean_7"] = pd.Series(week_spreads, dtype=float).mean()
    return features


def build_feature_table(
    prices_by_series: dict[int, dict[date, float]],
    spreads_by_series: dict[int, dict[date, float]],
    series_info: pd.DataFrame,
    origins: list[date],
) -> pd.DataFrame:
    """One row per (series, origin, horizon 1-7) for every origin with at least one earlier price.

    Columns: the FEATURES, `last_price`, `target_date`, `origin_date`, `actual` (NaN if the target date is
    unreported) and `target_log_ratio` (NaN with it). `series_info` has series_id, market and crop. The
    categorical columns get every series' value as a category, so all folds use the same categories.
    """
    info = series_info.set_index("series_id")
    rows = []
    for series_id, prices in prices_by_series.items():
        for origin in origins:
            features = origin_features(prices, spreads_by_series[series_id], origin)
            if features is None:
                continue
            for horizon in HORIZONS_DAYS:
                target_date = origin + timedelta(days=horizon)
                actual = prices.get(target_date, math.nan)
                rows.append(
                    {
                        "series_id": series_id,
                        "crop": info.loc[series_id, "crop"],
                        "market": info.loc[series_id, "market"],
                        "origin_date": origin,
                        "horizon_days": horizon,
                        "target_date": target_date,
                        "target_day_of_week": target_date.weekday(),
                        "target_month": target_date.month,
                        **features,
                        "actual": actual,
                        "target_log_ratio": math.log(actual / features["last_price"]),
                    }
                )

    table = pd.DataFrame(rows)
    for column in CATEGORICAL_FEATURES:
        categories = sorted(series_info[column].unique())
        table[column] = pd.Categorical(table[column], categories=categories)
    return table
