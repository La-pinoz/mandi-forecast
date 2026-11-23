"""Dashboard loaders against the database (needs TEST_DATABASE_URL): scores, latest forecasts, data health.

Prices enter through the real path (`process_snapshot` + `refresh_prices_current`), forecasts through
`forecast_baselines` and scores through `score_forecasts`. One series, Barasat Tomato, price 3000 + 10 × day
for 1–21 November 2026, so every 7-day difference is 70.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import psycopg
import pytest

from mandi.evaluate.live_report import load_latest_forecasts, load_live_scores, scorecard
from mandi.evaluate.scoring import score_forecasts
from mandi.ingest.records import VARIETY_WISE_SOURCE
from mandi.ingest.snapshot import process_snapshot, tracked_series_keys
from mandi.models.baselines import NAIVE_MODEL, SEASONAL_NAIVE_MODEL
from mandi.models.daily_forecast import forecast_baselines
from mandi.monitor.health import load_data_health
from mandi.store.runs import finish_run, start_run
from mandi.store.vintages import refresh_prices_current

IST = timezone(timedelta(hours=5, minutes=30))


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=IST)


def receive(conn: psycopg.Connection, modal_by_day: dict[int, int], seen_at: datetime) -> None:
    """Store Barasat tomato records for the given November days, first seen at `seen_at`, as one run."""
    records = []
    for day, modal in modal_by_day.items():
        records.append(
            {
                "State": "West Bengal",
                "District": "North 24 Parganas",
                "Market": "Barasat APMC",
                "Commodity": "Tomato",
                "Variety": "Other",
                "Grade": "FAQ",
                "Arrival_Date": f"{day:02d}/11/2026",
                "Min_Price": modal - 200,
                "Max_Price": modal + 200,
                "Modal_Price": modal,
            }
        )
    run_id = start_run(conn, "test")
    process_snapshot(conn, run_id, VARIETY_WISE_SOURCE, records, seen_at, tracked_series_keys(conn))
    refresh_prices_current(conn)
    conn.commit()


@pytest.fixture
def conn(schema_conn: psycopg.Connection) -> psycopg.Connection:
    """Forecasts from origins 13/11 and 14/11; scored on 22/11 and, after a revision of 20/11, on 23/11."""
    schema_conn.execute(
        "INSERT INTO series (source, market, commodity, variety, grade, district) "
        "VALUES (%s, 'Barasat', 'Tomato', 'Other', 'FAQ', 'North 24 Parganas')",
        (VARIETY_WISE_SOURCE,),
    )
    schema_conn.commit()
    receive(schema_conn, {day: 3000 + 10 * day for day in range(1, 14)}, at(13, 8))
    forecast_baselines(schema_conn, at(13, 7, 17))  # nothing seen yet: all unavailable
    forecast_baselines(schema_conn, at(14, 7, 17))  # origin 14/11, sees 1..13/11
    receive(schema_conn, {day: 3000 + 10 * day for day in range(14, 22)}, at(22, 8))
    score_forecasts(schema_conn, date(2026, 11, 22))  # all provisional
    receive(schema_conn, {20: 3250}, at(22, 12))  # 20/11 revised from 3200
    score_forecasts(schema_conn, date(2026, 11, 23))  # target 15/11 settles; 20/11 rescored
    return schema_conn


def test_each_forecast_counts_once_settled_or_with_its_latest_provisional_score(conn):
    scores = load_live_scores(conn, at(23, 12))

    # Naive: 7 targets; seasonal naive: 6 (14/11 was not seen by the cutoff, so 21/11 is unavailable).
    assert scores["forecast_id"].is_unique
    assert len(scores) == 13
    settled = scores[scores["status"] == "settled"]
    assert settled["target_date"].tolist() == [date(2026, 11, 15)] * 2
    revised = scores[scores["target_date"] == date(2026, 11, 20)]
    assert revised["status"].tolist() == ["provisional"] * 2
    assert revised["actual"].tolist() == [3250.0, 3250.0]  # the newer vintage, not 3200
    # Seasonal-naive MAE before origin 14/11: every |price(t) - price(t - 7)| is 70.
    assert scores["mase_scale"].tolist() == [70.0] * 13


def test_settled_scorecard_from_the_database(conn):
    scores = load_live_scores(conn, at(23, 12))

    table = scorecard(scores[scores["status"] == "settled"], ["crop", "horizon_days"])

    # Target 15/11, actual 3150: naive 3130 (13/11), seasonal naive 3080 (8/11).
    row = table.iloc[0]
    assert (len(table), row["crop"], row["horizon_days"], row["n"]) == (1, "Tomato", 1, 1)
    assert (row[f"MAE {NAIVE_MODEL}"], row[f"MAE {SEASONAL_NAIVE_MODEL}"]) == (20.0, 70.0)
    assert (row[f"MASE {NAIVE_MODEL}"], row[f"MASE {SEASONAL_NAIVE_MODEL}"]) == (0.286, 1.0)
    assert row["lowest MASE"] == NAIVE_MODEL
    assert row["note"] == "insufficient data (n < 30)"


def test_latest_forecasts_come_from_the_most_recent_origin_only(conn):
    forecasts = load_latest_forecasts(conn)

    assert len(forecasts) == 14
    assert set(forecasts["origin_date"]) == {date(2026, 11, 14)}
    naive = forecasts[forecasts["model_version"] == NAIVE_MODEL]
    assert naive["point_forecast"].tolist() == [3130.0] * 7
    unavailable = forecasts[forecasts["point_forecast"].isna()]
    assert unavailable[["model_version", "target_date"]].values.tolist() == [
        [SEASONAL_NAIVE_MODEL, date(2026, 11, 21)]
    ]


def test_data_health_tables(schema_conn):
    for market in ["Barasat", "Habra"]:
        schema_conn.execute(
            "INSERT INTO series (source, market, commodity, variety, grade, district) "
            "VALUES ('test', %s, 'Tomato', 'Other', 'FAQ', 'North 24 Parganas')",
            (market,),
        )
    schema_conn.execute(
        """
        INSERT INTO series_day_status (series_id, arrival_date, status, is_anomaly, anomaly_reason) VALUES
            (1, '2026-11-01', 'reported', true, 'jump: test'),
            (1, '2026-11-02', 'unreported', false, NULL),
            (2, '2026-11-01', 'reported', false, NULL),
            (2, '2026-11-02', 'unreported', false, NULL)
        """
    )
    first_ingest = start_run(schema_conn, "daily_ingest")
    finish_run(schema_conn, first_ingest, "succeeded")
    second_ingest = start_run(schema_conn, "daily_ingest")
    finish_run(schema_conn, second_ingest, "failed", error_message="HTTPError: 503")
    start_run(schema_conn, "score")  # still running, never succeeded
    schema_conn.execute(
        "INSERT INTO quarantine (run_id, row_hash, source, record, reason_code) "
        "VALUES (%s, 'abc', 'test', '{}', 'conflicting_duplicate')",
        (first_ingest,),
    )

    health = load_data_health(schema_conn)

    runs = health.last_runs.set_index("job_type")
    first_finished = schema_conn.execute(
        "SELECT finished_at FROM pipeline_runs WHERE run_id = %s", (first_ingest,)
    ).fetchone()[0]
    assert runs.loc["daily_ingest", "latest_status"] == "failed"
    assert runs.loc["daily_ingest", "latest_error"] == "HTTPError: 503"
    assert runs.loc["daily_ingest", "last_succeeded_at"] == first_finished
    assert runs.loc["score", "latest_status"] == "running"
    assert pd.isna(runs.loc["score", "last_succeeded_at"])
    assert health.quarantine_by_reason.to_dict("records") == [
        {"reason_code": "conflicting_duplicate", "rows": 1}
    ]
    assert health.day_status.to_dict("records") == [
        {
            "crop": "Tomato",
            "market": "Barasat",
            "reported": 1,
            "no_transaction": 0,
            "unreported": 1,
            "anomaly_flags": 1,
        },
        {
            "crop": "Tomato",
            "market": "Habra",
            "reported": 1,
            "no_transaction": 0,
            "unreported": 1,
            "anomaly_flags": 0,
        },
    ]
    assert health.widespread_gap_dates.to_dict("records") == [
        {"arrival_date": date(2026, 11, 2), "series_unreported": 2, "series_tracked": 2}
    ]
