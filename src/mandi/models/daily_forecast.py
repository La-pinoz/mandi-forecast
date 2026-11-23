"""M06: write the day's baseline forecasts for every active series to `forecasts`.

Run from the project root (uses DATABASE_URL from the environment or .env):
    uv run python -m mandi.models.daily_forecast

Flow (one pipeline_runs row, job type 'forecast'):
1. cutoff = now (IST). The origin date is the IST date of the cutoff; targets are origin + 1..7 days.
2. Read prices as of the cutoff (`prices_as_of`: the latest vintage of each observation with
   first_seen_at <= cutoff), so a forecast can only use what the pipeline had seen by then.
3. For each active series: naive and seasonal-naive forecasts (models/baselines.py), 14 rows.
4. Insert them all in one transaction. A forecast that already exists for the same series, origin day,
   horizon and model is left as it is, so re-running on the same day adds nothing and changes nothing.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from mandi.ingest.snapshot import IST
from mandi.models.baselines import baseline_forecasts
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
    forecasts_unavailable: int  # of those built, how many have no value (mostly seasonal naive)
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

        params = []
        for series_id, series_prices in prices.items():
            for forecast in baseline_forecasts(series_prices, origin_date):
                params.append(
                    {
                        "series_id": series_id,
                        "origin_date": origin_date,
                        "horizon_days": forecast.horizon_days,
                        "target_date": forecast.target_date,
                        "model_version": forecast.model_version,
                        "point_forecast": forecast.point_forecast,
                        "unavailable_reason": forecast.unavailable_reason,
                        "input_as_of_at": cutoff,
                        "forecast_run_id": run_id,
                    }
                )
        with conn.cursor() as cursor:
            cursor.executemany(INSERT_FORECAST, params)
            inserted = cursor.rowcount
        conn.commit()
    except Exception as error:
        conn.rollback()
        finish_run(conn, run_id, "failed", error_message=f"{type(error).__name__}: {error}")
        raise

    unavailable = 0
    for row in params:
        if row["point_forecast"] is None:
            unavailable += 1
    summary = ForecastSummary(
        origin_date=origin_date,
        series=len(prices),
        forecasts_built=len(params),
        forecasts_unavailable=unavailable,
        forecasts_inserted=inserted,
    )
    finish_run(conn, run_id, "succeeded", rows_in=len(params), rows_out=inserted)
    return run_id, summary


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    cutoff = datetime.now(IST)
    with get_connection() as conn:
        run_id, summary = forecast_baselines(conn, cutoff)

    print(f"Run {run_id} succeeded. Cutoff {cutoff:%Y-%m-%d %H:%M:%S %Z}, origin date {summary.origin_date}.")
    print(f"Series: {summary.series}; forecasts built: {summary.forecasts_built}")
    print(f"Unavailable (no value stored, reason recorded): {summary.forecasts_unavailable}")
    print(f"Inserted: {summary.forecasts_inserted}")
    if summary.forecasts_inserted == 0:
        print("This origin day was already forecast; the existing forecasts are kept unchanged.")


if __name__ == "__main__":
    main()
