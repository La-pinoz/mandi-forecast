"""M10: live scorecard, rolling performance and latest forecasts, read from the database (blueprint 14).

Used by the Streamlit dashboard (`dashboard/app.py`); M12 serves the same tables through the API.

Which score of a forecast counts (provisional and settled never overlap):
- **settled**: the forecast's one settled score (taken once the target is older than the settling window).
- **provisional**: for a forecast that is not settled yet, its latest provisional score, i.e. the score
  against the newest vintage of the actual. Older provisional scores of revised actuals stay in
  `forecast_scores` but are not counted twice.

Models are compared like-for-like: only the (series, origin, horizon) rows that every model scored are used
(`common_sample` from M07), so a model that started later or was unavailable more often is not compared on
different days. Every table carries its n; a group with fewer than `MIN_SCORED_FORECASTS` rows is kept and
marked "insufficient data", never hidden.

The MASE scale of a live score is the series' seasonal-naive MAE on all prices with an arrival date before
the forecast's origin: the same rule as a backtest fold, where the scale comes from the training period.
"""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import psycopg

from mandi.evaluate.backtest import common_sample, score_table
from mandi.evaluate.metrics import mase, mean_absolute_error, seasonal_naive_scale
from mandi.models.daily_forecast import prices_by_series

# Below this many scored forecasts a metric is shown as "insufficient data". A display rule, not a test:
# 30 rows of one crop and horizon are roughly one month of one market, far too few to rank models on.
MIN_SCORED_FORECASTS = 30
ROLLING_WINDOW_DAYS = 28

# The newest score of each forecast and status (score_id grows with every insert).
LATEST_SCORES = """
SELECT DISTINCT ON (sc.forecast_id, sc.status)
       sc.forecast_id, sc.status, f.series_id, s.commodity, s.market, f.origin_date, f.horizon_days,
       f.target_date, f.model_version, f.point_forecast, sc.actual_price
FROM forecast_scores sc
JOIN forecasts f ON f.forecast_id = sc.forecast_id
JOIN series s ON s.series_id = f.series_id
ORDER BY sc.forecast_id, sc.status, sc.score_id DESC
"""

SCORE_COLUMNS = [
    "forecast_id",
    "status",
    "series_id",
    "crop",
    "market",
    "origin_date",
    "horizon_days",
    "target_date",
    "model_version",
    "point_forecast",
    "actual",
]

# Every model's forecasts from each series' most recent origin date.
LATEST_FORECASTS = """
SELECT f.series_id, f.origin_date, f.horizon_days, f.target_date, f.model_version, f.point_forecast,
       f.unavailable_reason, f.input_as_of_at
FROM forecasts f
JOIN (SELECT series_id, max(origin_date) AS origin_date FROM forecasts GROUP BY series_id) latest
  ON latest.series_id = f.series_id AND latest.origin_date = f.origin_date
ORDER BY f.series_id, f.model_version, f.horizon_days
"""

FORECAST_COLUMNS = [
    "series_id",
    "origin_date",
    "horizon_days",
    "target_date",
    "model_version",
    "point_forecast",
    "unavailable_reason",
    "input_as_of_at",
]


def load_live_scores(conn: psycopg.Connection, now: datetime) -> pd.DataFrame:
    """One row per scored forecast: its settled score if it has one, otherwise its latest provisional score.

    Columns: SCORE_COLUMNS plus `mase_scale`. Prices for the MASE scale are those known at `now`
    (timezone-aware). A series with no seasonal pair before the origin gets a NaN scale, and `mase` then
    raises instead of quietly averaging over fewer rows.
    """
    rows = conn.execute(LATEST_SCORES).fetchall()
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    settled_ids = set(scores.loc[scores["status"] == "settled", "forecast_id"])
    replaced_by_settled = (scores["status"] == "provisional") & scores["forecast_id"].isin(settled_ids)
    scores = scores[~replaced_by_settled].reset_index(drop=True)
    scores["point_forecast"] = scores["point_forecast"].astype(float)
    scores["actual"] = scores["actual"].astype(float)

    prices = prices_by_series(conn, now)
    scales = {}
    for series_id, origin_date in (
        scores[["series_id", "origin_date"]].drop_duplicates().itertuples(index=False)
    ):
        scales[(series_id, origin_date)] = seasonal_naive_scale(prices.get(series_id, {}), origin_date)
    scale_column = []
    for series_id, origin_date in scores[["series_id", "origin_date"]].itertuples(index=False):
        scale_column.append(scales[(series_id, origin_date)])
    scores["mase_scale"] = pd.Series(scale_column, index=scores.index, dtype=float)
    return scores


def scorecard(scores: pd.DataFrame, by: list[str], min_n: int = MIN_SCORED_FORECASTS) -> pd.DataFrame:
    """One row per group of `by`: n, then MASE and MAE of every model on the same rows, the model with the
    lowest MASE, and a note when n < `min_n`. Losing rows are kept like any other.

    `scores` should hold one status only (all settled or all provisional). Returns an empty table if no row
    was scored by every model.
    """
    shared = common_sample(scores)
    if shared.empty:
        return pd.DataFrame()
    long_table = score_table(shared, by)
    wide = long_table.pivot(index=by, columns="model_version", values=["MASE", "MAE"])
    wide.columns = [f"{metric} {model}" for metric, model in wide.columns]
    mase_columns = [column for column in wide.columns if column.startswith("MASE ")]
    # On a tie idxmin names the first model in column order (alphabetical); exact ties are not expected.
    wide["lowest MASE"] = wide[mase_columns].idxmin(axis=1).str.removeprefix("MASE ")
    # Every model has the same n in a group, because the rows are shared.
    wide.insert(0, "n", long_table.groupby(by)["n"].first())
    wide["note"] = np.where(wide["n"] < min_n, f"insufficient data (n < {min_n})", "")
    return wide.reset_index()


def rolling_scores(
    scores: pd.DataFrame, window_days: int = ROLLING_WINDOW_DAYS, min_n: int = MIN_SCORED_FORECASTS
) -> pd.DataFrame:
    """MAE and MASE per model over a moving window of target dates, on the rows every model scored.

    One row per model and window end date (each date from the first to the last target date): n is the
    number of scores with a target date in the `window_days` days ending that date. MAE and MASE are NaN
    when n < `min_n`, so a thin window is never drawn as if it were solid. All crops, series and horizons
    are pooled. Columns: end_date, model_version, n, MAE, MASE.
    """
    columns = ["end_date", "model_version", "n", "MAE", "MASE"]
    shared = common_sample(scores)
    if shared.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    day = shared["target_date"].min()
    last_day = shared["target_date"].max()
    while day <= last_day:
        window_start = day - timedelta(days=window_days - 1)
        in_window = shared[(shared["target_date"] >= window_start) & (shared["target_date"] <= day)]
        for model_version, group in in_window.groupby("model_version"):
            row = {
                "end_date": day,
                "model_version": model_version,
                "n": len(group),
                "MAE": np.nan,
                "MASE": np.nan,
            }
            if len(group) >= min_n:
                row["MAE"] = mean_absolute_error(group["point_forecast"], group["actual"])
                row["MASE"] = mase(group["point_forecast"], group["actual"], group["mase_scale"])
            rows.append(row)
        day = day + timedelta(days=1)
    return pd.DataFrame(rows, columns=columns)


def load_latest_forecasts(conn: psycopg.Connection) -> pd.DataFrame:
    """All models' forecasts (with values or unavailable reasons) from each series' latest origin date."""
    rows = conn.execute(LATEST_FORECASTS).fetchall()
    forecasts = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    forecasts["point_forecast"] = forecasts["point_forecast"].astype(float)
    return forecasts


def price_history(prices: dict[date, float], first_day: date) -> pd.DataFrame:
    """Reported prices from `first_day` on, sorted by date. Unreported days are absent, not filled."""
    rows = []
    for day in sorted(prices):
        if day >= first_day:
            rows.append({"date": day, "price": prices[day]})
    return pd.DataFrame(rows, columns=["date", "price"])
