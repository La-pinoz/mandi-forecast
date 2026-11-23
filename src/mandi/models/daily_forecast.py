"""M06 + M09: write the day's forecasts for every active series to `forecasts`.

Run from the project root (uses DATABASE_URL from the environment or .env):
    uv run python -m mandi.models.daily_forecast

Two jobs with the same information cutoff, each its own pipeline_runs row (job type 'forecast'):
- `forecast_baselines` (M06): naive and seasonal naive, 14 rows per series.
- `forecast_lightgbm` (M09): the champion LightGBM from `model_registry`, 7 rows per series.
The baselines run first and are committed on their own, so a LightGBM failure (e.g. no champion registered)
can't take the baselines down with it; the command still fails, so the workflow ends red.

Flow of each job:
1. cutoff = now (IST). The origin date is the IST date of the cutoff; targets are origin + 1..7 days.
2. Read prices as of the cutoff (`prices_as_of`: the latest vintage of each observation with
   first_seen_at <= cutoff), so a forecast can only use what the pipeline had seen by then.
3. Build the forecasts. LightGBM uses the M08 features (features/build.py), which only look at arrival
   dates before the origin, exactly as in the backtest.
4. Insert them in one transaction. A forecast that already exists for the same series, origin day, horizon
   and model is left as it is, so re-running on the same day adds nothing and changes nothing.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from mandi.features.build import build_feature_table
from mandi.ingest.snapshot import IST
from mandi.models.baselines import HORIZONS_DAYS, baseline_forecasts
from mandi.models.lgbm import predict_prices
from mandi.models.registry import Champion, load_champion
from mandi.store.asof import prices_as_of
from mandi.store.db import get_connection
from mandi.store.runs import finish_run, start_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ACTIVE_SERIES = """
SELECT series_id, source, market, commodity, variety, grade
FROM series
WHERE is_active
ORDER BY series_id
"""

SERIES_INFO = """
SELECT series_id, market, commodity
FROM series
WHERE is_active
ORDER BY series_id
"""

INSERT_FORECAST = """
INSERT INTO forecasts (
    series_id, origin_date, horizon_days, target_date, model_version, point_forecast, unavailable_reason,
    input_as_of_at, forecast_run_id
)
VALUES (
    %(series_id)s, %(origin_date)s, %(horizon_days)s, %(target_date)s, %(model_version)s,
    %(point_forecast)s, %(unavailable_reason)s, %(input_as_of_at)s, %(forecast_run_id)s
)
ON CONFLICT (series_id, origin_date, horizon_days, model_version) DO NOTHING
"""


@dataclass
class ForecastSummary:
    origin_date: date
    series: int
    forecasts_built: int
    forecasts_unavailable: int  # of those built, how many have no value
    forecasts_inserted: int  # 0 when this origin day was already forecast


def prices_by_series(conn: psycopg.Connection, cutoff: datetime) -> dict[int, dict[date, float]]:
    """{series_id: {arrival_date: modal_price}} for every active series, as known at `cutoff`.

    A series with no price yet maps to an empty dict.
    """
    prices, _ = prices_and_spreads_by_series(conn, cutoff)
    return prices


def prices_and_spreads_by_series(
    conn: psycopg.Connection, cutoff: datetime
) -> tuple[dict[int, dict[date, float]], dict[int, dict[date, float]]]:
    """Modal prices as in `prices_by_series`, and each day's spread (max - min) / modal for the same days.

    The spread is a LightGBM feature (M08); prices are positive (M04 contracts), so modal is never zero.
    """
    series_ids = {}
    for series_id, source, market, commodity, variety, grade in conn.execute(ACTIVE_SERIES).fetchall():
        series_ids[(source, market, commodity, variety, grade)] = series_id

    prices = {series_id: {} for series_id in series_ids.values()}
    spreads = {series_id: {} for series_id in series_ids.values()}
    for row in prices_as_of(conn, cutoff):
        series_key = (row["source"], row["market"], row["commodity"], row["variety"], row["grade"])
        if series_key in series_ids:
            series_id = series_ids[series_key]
            modal = float(row["modal_price"])
            prices[series_id][row["arrival_date"]] = modal
            spreads[series_id][row["arrival_date"]] = (
                float(row["max_price"]) - float(row["min_price"])
            ) / modal
    return prices, spreads


def series_info_table(conn: psycopg.Connection) -> pd.DataFrame:
    """Each active series' id, market and crop: the categorical LightGBM features (M08)."""
    rows = conn.execute(SERIES_INFO).fetchall()
    return pd.DataFrame(rows, columns=["series_id", "market", "crop"])


def lightgbm_forecasts(
    champion: Champion,
    prices: dict[int, dict[date, float]],
    spreads: dict[int, dict[date, float]],
    series_info: pd.DataFrame,
    origin_date: date,
) -> list[dict]:
    """The champion's forecasts for horizons 1-7 from `origin_date`, 7 per series.

    A series with no price before the origin has no features; its 7 forecasts are stored as unavailable.
    """
    table = build_feature_table(prices, spreads, series_info, [origin_date])
    table["point_forecast"] = predict_prices(champion.models, table)

    forecasts = []
    for row in table.itertuples(index=False):
        forecasts.append(
            {
                "series_id": int(row.series_id),
                "horizon_days": int(row.horizon_days),
                "target_date": row.target_date,
                "model_version": champion.model_version,
                "point_forecast": float(row.point_forecast),
                "unavailable_reason": None,
            }
        )

    forecast_series = set(table["series_id"].astype(int))
    for series_id in prices:
        if series_id in forecast_series:
            continue
        for horizon in HORIZONS_DAYS:
            forecasts.append(
                {
                    "series_id": series_id,
                    "horizon_days": horizon,
                    "target_date": origin_date + timedelta(days=horizon),
                    "model_version": champion.model_version,
                    "point_forecast": None,
                    "unavailable_reason": "no_price_before_origin",
                }
            )
    return forecasts


def insert_forecasts(conn: psycopg.Connection, forecasts: list[dict], cutoff: datetime, run_id: int) -> int:
    """Insert `forecasts` with their origin date, cutoff and run id, and commit. Returns rows inserted."""
    origin_date = cutoff.astimezone(IST).date()
    params = []
    for forecast in forecasts:
        params.append(
            {
                **forecast,
                "origin_date": origin_date,
                "input_as_of_at": cutoff,
                "forecast_run_id": run_id,
            }
        )
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_FORECAST, params)
        inserted = cursor.rowcount
    conn.commit()
    return inserted


def finish_forecast_run(
    conn: psycopg.Connection, run_id: int, cutoff: datetime, series: int, forecasts: list[dict], inserted: int
) -> ForecastSummary:
    """Mark the run succeeded and summarise it."""
    unavailable = 0
    for forecast in forecasts:
        if forecast["point_forecast"] is None:
            unavailable += 1
    finish_run(conn, run_id, "succeeded", rows_in=len(forecasts), rows_out=inserted)
    return ForecastSummary(
        origin_date=cutoff.astimezone(IST).date(),
        series=series,
        forecasts_built=len(forecasts),
        forecasts_unavailable=unavailable,
        forecasts_inserted=inserted,
    )


def forecast_baselines(conn: psycopg.Connection, cutoff: datetime) -> tuple[int, ForecastSummary]:
    """Build and store baseline forecasts with information cutoff `cutoff`. Returns (run_id, summary).

    Any error rolls back the forecasts, marks the run 'failed' and is raised again.
    """
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    origin_date = cutoff.astimezone(IST).date()
    run_id = start_run(conn, "forecast", cutoff_at=cutoff)
    try:
        prices = prices_by_series(conn, cutoff)
        if not prices:
            raise RuntimeError("No active series in `series`; load them first (M03).")

        forecasts = []
        for series_id, series_prices in prices.items():
            for forecast in baseline_forecasts(series_prices, origin_date):
                forecasts.append(
                    {
                        "series_id": series_id,
                        "horizon_days": forecast.horizon_days,
                        "target_date": forecast.target_date,
                        "model_version": forecast.model_version,
                        "point_forecast": forecast.point_forecast,
                        "unavailable_reason": forecast.unavailable_reason,
                    }
                )
        inserted = insert_forecasts(conn, forecasts, cutoff, run_id)
    except Exception as error:
        conn.rollback()
        finish_run(conn, run_id, "failed", error_message=f"{type(error).__name__}: {error}")
        raise
    return run_id, finish_forecast_run(conn, run_id, cutoff, len(prices), forecasts, inserted)


def forecast_lightgbm(conn: psycopg.Connection, cutoff: datetime) -> tuple[int, ForecastSummary]:
    """Build and store the champion LightGBM's forecasts with information cutoff `cutoff`.

    Returns (run_id, summary). Any error (including no registered champion) rolls back the forecasts,
    marks the run 'failed' and is raised again.
    """
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    origin_date = cutoff.astimezone(IST).date()
    run_id = start_run(conn, "forecast", cutoff_at=cutoff)
    try:
        champion = load_champion(conn)
        prices, spreads = prices_and_spreads_by_series(conn, cutoff)
        if not prices:
            raise RuntimeError("No active series in `series`; load them first (M03).")
        series_info = series_info_table(conn)
        forecasts = lightgbm_forecasts(champion, prices, spreads, series_info, origin_date)
        inserted = insert_forecasts(conn, forecasts, cutoff, run_id)
    except Exception as error:
        conn.rollback()
        finish_run(conn, run_id, "failed", error_message=f"{type(error).__name__}: {error}")
        raise
    return run_id, finish_forecast_run(conn, run_id, cutoff, len(prices), forecasts, inserted)


def print_summary(job: str, run_id: int, summary: ForecastSummary) -> None:
    print(
        f"{job}: run {run_id} succeeded. Series: {summary.series}; forecasts built: {summary.forecasts_built}"
    )
    print(f"  Unavailable (no value stored, reason recorded): {summary.forecasts_unavailable}")
    print(f"  Inserted: {summary.forecasts_inserted}")
    if summary.forecasts_inserted == 0:
        print("  This origin day was already forecast; the existing forecasts are kept unchanged.")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    cutoff = datetime.now(IST)
    print(f"Cutoff {cutoff:%Y-%m-%d %H:%M:%S %Z}, origin date {cutoff.date()}.")
    with get_connection() as conn:
        run_id, summary = forecast_baselines(conn, cutoff)
        print_summary("Baselines", run_id, summary)
        run_id, summary = forecast_lightgbm(conn, cutoff)
        print_summary("LightGBM", run_id, summary)


if __name__ == "__main__":
    main()
