"""M07: rolling-origin, expanding-window backtest of the baselines (blueprint 8.1-8.3).

Run from the project root (reads DATABASE_URL from the environment or .env; writes nothing to the database):
    uv run python -m mandi.evaluate.backtest

Design (owner decisions, 2026-11-23; see docs/M07-backtest-harness.md):
- Folds start every 7 days from FIRST_FOLD_CUTOFF. A fold's training data is every price with an arrival
  date **before** its cutoff (expanding window: each fold has all the earlier data). Its forecast origins are
  the 7 days starting at the cutoff, so every day is an origin, as in the live daily run, and a model trained
  per fold (M08) is retrained weekly, as planned for live (M13).
- A forecast at origin D sees only prices with arrival date **before** D. The live run at 07:17 IST never has
  day D's prices, so the backtest gets the same information. Targets are D + 1..7 calendar days.
- Inputs are the latest history (`prices_as_of` now = what `prices_current` holds), not the values as first
  published. This is the vintage-leakage limitation (blueprint 8.1): documented, not engineered around.

Flow of `main()`:
1. Load each active series' prices as known now, and its market and crop.
2. `make_folds`: weekly folds from FIRST_FOLD_CUTOFF to the last origin whose 7 targets are all in the data.
3. `backtest_baselines`: naive and seasonal-naive forecasts for every series, origin and horizon, each with
   its actual (missing if the target is unreported), its volatile/normal label and its fold's MASE scale.
4. Write one row per forecast to data/backtest/baseline_forecasts.csv and print the report tables (markdown).
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from mandi.evaluate.metrics import mase, mean_absolute_error, seasonal_naive_scale, smape
from mandi.ingest.snapshot import IST
from mandi.models.baselines import HORIZONS_DAYS, SEASON_DAYS, baseline_forecasts
from mandi.models.daily_forecast import prices_by_series
from mandi.store.db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_CSV = PROJECT_ROOT / "data" / "backtest" / "baseline_forecasts.csv"

FIRST_FOLD_CUTOFF = date(2024, 1, 1)  # every fold trains on at least one full year (2023)
FOLD_DAYS = 7
MAX_HORIZON_DAYS = max(HORIZONS_DAYS)
# Volatile (fixed before any results were seen): a target date whose absolute 7-day log price change is in the
# top decile of that series' 7-day changes over its full history.
VOLATILE_QUANTILE = 0.9

SERIES_INFO = """
SELECT series_id, market, commodity
FROM series
WHERE is_active
ORDER BY series_id
"""


@dataclass
class Fold:
    """One training cutoff and the origins forecast with it.

    Training data: arrival dates strictly before `train_cutoff`. The first origin is `train_cutoff` itself, so
    no training day is on or after any origin of the fold.
    """

    train_cutoff: date
    origin_dates: list[date]


def make_folds(first_cutoff: date, last_data_date: date) -> list[Fold]:
    """Weekly folds from `first_cutoff`, each with up to 7 consecutive daily origins.

    The last origin is `last_data_date` - 7 days, so all 7 targets of every origin are inside the data.
    """
    last_origin = last_data_date - timedelta(days=MAX_HORIZON_DAYS)
    folds = []
    cutoff = first_cutoff
    while cutoff <= last_origin:
        origins = []
        for offset in range(FOLD_DAYS):
            origin = cutoff + timedelta(days=offset)
            if origin <= last_origin:
                origins.append(origin)
        folds.append(Fold(train_cutoff=cutoff, origin_dates=origins))
        cutoff += timedelta(days=FOLD_DAYS)
    return folds


def known_before(prices: dict[date, float], day: date) -> dict[date, float]:
    """The prices with arrival date strictly before `day`: all a forecast or a fold may see."""
    return {arrival_date: price for arrival_date, price in prices.items() if arrival_date < day}


def volatility_labels(prices: dict[date, float]) -> tuple[float | None, dict[date, str]]:
    """Label each day of one series 'volatile' or 'normal'. Returns (threshold, labels).

    The 7-day change of day t is |log(price(t) / price(t - 7 days))|, defined only when both days are
    reported. The threshold is the VOLATILE_QUANTILE quantile of all changes in the series' full history; a
    change at or above it is volatile. A day without a change (it or t - 7 unreported) gets no label; callers
    call it 'unlabelled'. The labels describe the target dates, not model inputs, so using the full history is
    not leakage.
    """
    changes = {}
    for day, price in prices.items():
        week_before = day - timedelta(days=SEASON_DAYS)
        if week_before in prices:
            changes[day] = abs(math.log(price / prices[week_before]))
    if not changes:
        return None, {}

    threshold = float(pd.Series(list(changes.values())).quantile(VOLATILE_QUANTILE))
    labels = {}
    for day, change in changes.items():
        labels[day] = "volatile" if change >= threshold else "normal"
    return threshold, labels


def backtest_baselines(prices_by_series: dict[int, dict[date, float]], folds: list[Fold]) -> pd.DataFrame:
    """One row per (series, origin, horizon, model): the forecast and what it is scored against.

    Columns: series_id, train_cutoff, origin_date, horizon_days, target_date, model_version, point_forecast
    (NaN when unavailable), unavailable_reason, actual (NaN when the target is unreported), regime
    ('volatile' / 'normal' / 'unlabelled') and mase_scale (the series' scale for this fold, NaN if none).
    """
    rows = []
    for series_id, prices in prices_by_series.items():
        _, labels = volatility_labels(prices)
        for fold in folds:
            scale = seasonal_naive_scale(prices, fold.train_cutoff)
            for origin in fold.origin_dates:
                for forecast in baseline_forecasts(known_before(prices, origin), origin):
                    rows.append(
                        {
                            "series_id": series_id,
                            "train_cutoff": fold.train_cutoff,
                            "origin_date": origin,
                            "horizon_days": forecast.horizon_days,
                            "target_date": forecast.target_date,
                            "model_version": forecast.model_version,
                            "point_forecast": forecast.point_forecast,
                            "unavailable_reason": forecast.unavailable_reason,
                            "actual": prices.get(forecast.target_date),
                            "regime": labels.get(forecast.target_date, "unlabelled"),
                            "mase_scale": scale,
                        }
                    )
    # Explicit float columns, so None becomes NaN even if a column happens to hold no value at all.
    results = pd.DataFrame(rows)
    for column in ["point_forecast", "actual", "mase_scale"]:
        results[column] = results[column].astype(float)
    return results


def scored_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Forecasts that have a value and whose target date was reported: the only rows a metric can use."""
    return results[results["point_forecast"].notna() & results["actual"].notna()]


def common_sample(results: pd.DataFrame) -> pd.DataFrame:
    """Scored rows of the (series, origin, horizon) keys that every model scored.

    Seasonal naive is unavailable more often than naive, so each model's own scored rows are different
    samples. Comparing the models on these shared rows is like-for-like.
    """
    scored = scored_rows(results)
    models_per_key = scored.groupby(["series_id", "origin_date", "horizon_days"])["model_version"].transform(
        "nunique"
    )
    return scored[models_per_key == results["model_version"].nunique()]


def score_table(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """n, MAE, sMAPE and MASE of already-scored rows, per group of `by` and per model."""
    group_columns = [*by, "model_version"]
    rows = []
    for keys, group in scored.groupby(group_columns):
        row = dict(zip(group_columns, keys, strict=True))
        row["n"] = len(group)
        row["MAE"] = round(mean_absolute_error(group["point_forecast"], group["actual"]), 1)
        row["sMAPE %"] = round(smape(group["point_forecast"], group["actual"]), 2)
        row["MASE"] = round(mase(group["point_forecast"], group["actual"], group["mase_scale"]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def availability_table(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group and model: forecasts asked for, how many had a value, and how many could be scored."""
    group_columns = [*by, "model_version"]
    rows = []
    for keys, group in results.groupby(group_columns):
        has_value = group["point_forecast"].notna()
        has_actual = group["actual"].notna()
        row = dict(zip(group_columns, keys, strict=True))
        row["forecasts"] = len(group)
        row["available"] = int(has_value.sum())
        row["available %"] = round(100 * has_value.mean(), 1)
        row["target unreported"] = int((has_value & ~has_actual).sum())
        row["scored"] = int((has_value & has_actual).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    """A pandas table as a GitHub markdown table (pandas' own to_markdown needs an extra package)."""
    lines = ["| " + " | ".join(str(column) for column in table.columns) + " |"]
    lines.append("|" + "---|" * len(table.columns))
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_inputs(
    conn: psycopg.Connection, as_of: datetime
) -> tuple[dict[int, dict[date, float]], pd.DataFrame]:
    """Prices of every active series as known at `as_of` (the latest history), and each series' market and
    crop."""
    prices = prices_by_series(conn, as_of)
    series_info = pd.DataFrame(conn.execute(SERIES_INFO).fetchall(), columns=["series_id", "market", "crop"])
    return prices, series_info


def print_report(
    results: pd.DataFrame, series_info: pd.DataFrame, thresholds: dict[int, float | None]
) -> None:
    """Print the baseline section's tables as markdown."""
    results = results.merge(series_info, on="series_id", how="left")
    scored = scored_rows(results)

    print("### Volatile thresholds (90th percentile of |7-day log change|, full history)\n")
    threshold_table = series_info.copy()
    threshold_column = []
    for series_id in series_info["series_id"]:
        threshold = thresholds[series_id]
        threshold_column.append("none (no 7-day pair)" if threshold is None else round(threshold, 4))
    threshold_table["threshold"] = threshold_column
    print(markdown_table(threshold_table))

    print("\n### Availability, overall\n")
    print(markdown_table(availability_table(results, [])))
    print("\n### Availability by horizon\n")
    print(markdown_table(availability_table(results, ["horizon_days"])))

    print("\n### Each model on its own scored rows\n")
    print(markdown_table(score_table(scored, [])))
    print("\n### Like-for-like: rows both models scored\n")
    common = common_sample(results)
    print(markdown_table(score_table(common, [])))
    print("\n### Like-for-like by horizon\n")
    print(markdown_table(score_table(common, ["horizon_days"])))
    print("\n### By horizon (each model on its own rows)\n")
    print(markdown_table(score_table(scored, ["horizon_days"])))
    print("\n### By crop\n")
    print(markdown_table(score_table(scored, ["crop"])))
    print("\n### By market\n")
    print(markdown_table(score_table(scored, ["market"])))
    print("\n### By series\n")
    print(markdown_table(score_table(scored, ["market", "crop"])))
    print("\n### Normal vs. volatile target dates\n")
    print(markdown_table(score_table(scored, ["regime"])))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    as_of = datetime.now(IST)
    with get_connection() as conn:
        prices, series_info = load_inputs(conn, as_of)

    last_data_date = max(max(series_prices) for series_prices in prices.values() if series_prices)
    folds = make_folds(FIRST_FOLD_CUTOFF, last_data_date)
    results = backtest_baselines(prices, folds)

    thresholds = {}
    for series_id, series_prices in prices.items():
        thresholds[series_id], _ = volatility_labels(series_prices)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)

    print(f"Inputs: prices as known at {as_of:%Y-%m-%d %H:%M %Z}; latest arrival date {last_data_date}.")
    print(
        f"Series: {len(prices)}. Folds: {len(folds)}, "
        f"cutoffs {folds[0].train_cutoff} to {folds[-1].train_cutoff}."
    )
    print(
        f"Origins: {folds[0].origin_dates[0]} to {folds[-1].origin_dates[-1]}. Forecast rows: {len(results)}."
    )
    print(f"Per-forecast rows written to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}.\n")
    print_report(results, series_info, thresholds)


if __name__ == "__main__":
    main()
