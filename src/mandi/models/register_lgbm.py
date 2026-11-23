"""M09: train lgbm.v1 on all prices known now and register it as the champion.

Run once from the project root, after the M08 backtest (its CSVs in data/backtest/ give the metrics):
    uv run python -m mandi.evaluate.backtest_lgbm      # if data/backtest/*.csv are missing
    uv run python -m mandi.models.register_lgbm

Flow (one pipeline_runs row, job type 'register_model'):
1. `held_out_metrics`: lgbm.v1 and naive on the M08 held-out origins (2025-01-01 onwards), like-for-like.
   These describe the method (layout, parameters, features), measured fold by fold in the backtest; the
   model registered here is the same method trained once on all the data.
2. `training_table`: the M08 feature rows for every origin, from prices as known now, keeping the rows whose
   target date was reported.
3. Train the per-horizon layout, M08's choice, with the fixed parameters in lgbm.py.
4. Insert it into model_registry with promoted_at = now, and commit.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from mandi.evaluate.backtest import OUTPUT_CSV as BASELINE_CSV
from mandi.evaluate.backtest import common_sample, score_table
from mandi.evaluate.backtest_lgbm import HELD_OUT_START, POOLED
from mandi.evaluate.backtest_lgbm import OUTPUT_CSV as LGBM_CSV
from mandi.features.build import build_feature_table
from mandi.ingest.snapshot import IST
from mandi.models.baselines import NAIVE_MODEL
from mandi.models.daily_forecast import prices_and_spreads_by_series, series_info_table
from mandi.models.lgbm import MODEL_VERSION, NUM_BOOST_ROUND, PARAMS, train
from mandi.models.registry import register_model
from mandi.store.db import get_connection
from mandi.store.runs import finish_run, start_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PER_HORIZON = True  # M08 chose the per-horizon layout on the 2024 development origins


def held_out_metrics(lgbm_csv: Path, baseline_csv: Path) -> dict:
    """n, MAE, sMAPE and MASE of lgbm.v1 and naive on the held-out rows both scored (M08 backtest).

    Raises ValueError if the CSV says lgbm.v1 is not the per-horizon layout.
    """
    lgbm_rows = pd.read_csv(lgbm_csv, parse_dates=["origin_date"])
    baseline_rows = pd.read_csv(baseline_csv, parse_dates=["origin_date"])
    # The CSV keeps the losing layout under its own name; if that is the pooled one, lgbm.v1 is per horizon.
    if POOLED not in set(lgbm_rows["model_version"]):
        raise ValueError(f"{lgbm_csv.name}: lgbm.v1 is not the per-horizon layout; re-check the M08 choice")

    rows = pd.concat(
        [
            lgbm_rows[lgbm_rows["model_version"] == MODEL_VERSION],
            baseline_rows[baseline_rows["model_version"] == NAIVE_MODEL],
        ],
        ignore_index=True,
    )
    held_out = rows[rows["origin_date"] >= pd.Timestamp(HELD_OUT_START)]
    scores = score_table(common_sample(held_out), [])

    metrics = {
        "source": "M08 rolling-origin backtest, weekly folds, latest-snapshot inputs",
        "sample": f"held-out origins from {HELD_OUT_START}, rows scored by both lgbm.v1 and naive",
    }
    for _, score in scores.iterrows():
        metrics[score["model_version"]] = {
            "n": int(score["n"]),
            "MAE": float(score["MAE"]),
            "sMAPE %": float(score["sMAPE %"]),
            "MASE": float(score["MASE"]),
        }
    return metrics


def training_table(conn: psycopg.Connection, as_of: datetime) -> pd.DataFrame:
    """Feature rows for every origin from the day after the first price, as known at `as_of`, keeping the
    rows with a reported target."""
    prices, spreads = prices_and_spreads_by_series(conn, as_of)
    first_data_date = min(min(series_prices) for series_prices in prices.values() if series_prices)
    last_data_date = max(max(series_prices) for series_prices in prices.values() if series_prices)

    origins = []
    origin = first_data_date + timedelta(days=1)
    while origin <= last_data_date:
        origins.append(origin)
        origin += timedelta(days=1)
    table = build_feature_table(prices, spreads, series_info_table(conn), origins)
    return table[table["actual"].notna()]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    for path in (LGBM_CSV, BASELINE_CSV):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the M07/M08 backtests first.")
    metrics = held_out_metrics(LGBM_CSV, BASELINE_CSV)

    as_of = datetime.now(IST)
    with get_connection() as conn:
        run_id = start_run(conn, "register_model", cutoff_at=as_of)
        try:
            training_rows = training_table(conn, as_of)
            models = train(training_rows, per_horizon=PER_HORIZON)
            params = {"lightgbm": PARAMS, "num_boost_round": NUM_BOOST_ROUND, "layout": "per_horizon"}
            register_model(
                conn,
                MODEL_VERSION,
                models,
                params,
                metrics,
                training_data_as_of=as_of,
                training_rows=len(training_rows),
                run_id=run_id,
                promote=True,
            )
            conn.commit()
        except Exception as error:
            conn.rollback()
            finish_run(conn, run_id, "failed", error_message=f"{type(error).__name__}: {error}")
            raise
        finish_run(conn, run_id, "succeeded", rows_in=len(training_rows), rows_out=1)

    last_target = training_rows["target_date"].max()
    print(f"Run {run_id}: registered {MODEL_VERSION} as champion.")
    print(f"Trained on {len(training_rows)} rows (prices as known at {as_of:%Y-%m-%d %H:%M %Z}).")
    print(f"Last target date in training: {last_target}.")
    print(f"Backtest metrics stored: {metrics}")


if __name__ == "__main__":
    main()
