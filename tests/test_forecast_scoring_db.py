"""Live baseline forecasts and provisional scoring against the database (needs TEST_DATABASE_URL).

Prices enter the way the daily job stores them: raw API records -> `process_snapshot` (contracts,
quarantine, vintages) -> `refresh_prices_current`. One tracked series, Barasat Tomato.
"""

from datetime import date, datetime, timedelta, timezone

import psycopg
import pytest

from mandi.evaluate.scoring import ModelCounts, score_provisional, scoring_counts
from mandi.ingest.records import VARIETY_WISE_SOURCE
from mandi.ingest.snapshot import process_snapshot, tracked_series_keys
from mandi.models.baselines import NAIVE_MODEL, SEASONAL_NAIVE_MODEL
from mandi.models.daily_forecast import forecast_baselines
from mandi.store.runs import start_run
from mandi.store.vintages import refresh_prices_current

IST = timezone(timedelta(hours=5, minutes=30))
CUTOFF = datetime(2026, 11, 21, 7, 17, tzinfo=IST)  # origin date 21/11


def at(day: int, hour: int) -> datetime:
    return datetime(2026, 11, day, hour, 0, tzinfo=IST)


def receive(
    conn: psycopg.Connection, arrival_day: int, modal: int, seen_at: datetime, min_price: int | None = None
) -> None:
    """Store one Barasat tomato record for 2026-11-<arrival_day>, first seen at `seen_at`, as its own run."""
    record = {
        "State": "West Bengal",
        "District": "North 24 Parganas",
        "Market": "Barasat APMC",
        "Commodity": "Tomato",
        "Variety": "Other",
        "Grade": "FAQ",
        "Arrival_Date": f"{arrival_day:02d}/11/2026",
        "Min_Price": min_price if min_price is not None else modal - 200,
        "Max_Price": modal + 200,
        "Modal_Price": modal,
    }
    run_id = start_run(conn, "test")
    process_snapshot(conn, run_id, VARIETY_WISE_SOURCE, [record], seen_at, tracked_series_keys(conn))
    refresh_prices_current(conn)
    conn.commit()


def stored_forecasts(conn: psycopg.Connection, model_version: str) -> dict[int, float | None]:
    rows = conn.execute(
        "SELECT horizon_days, point_forecast FROM forecasts WHERE model_version = %s ORDER BY horizon_days",
        (model_version,),
    ).fetchall()
    result = {}
    for horizon, value in rows:
        result[horizon] = float(value) if value is not None else None
    return result


@pytest.fixture
def conn(schema_conn: psycopg.Connection) -> psycopg.Connection:
    """One series with history up to the cutoff, plus data the pipeline only saw AFTER the cutoff."""
    schema_conn.execute(
        "INSERT INTO series (source, market, commodity, variety, grade, district) "
        "VALUES (%s, 'Barasat', 'Tomato', 'Other', 'FAQ', 'North 24 Parganas')",
        (VARIETY_WISE_SOURCE,),
    )
    schema_conn.commit()
    receive(schema_conn, 14, 3000, at(15, 7))
    receive(schema_conn, 15, 3100, at(16, 7))
    # 16/11 never reported.
    receive(schema_conn, 17, 3300, at(18, 7))
    receive(schema_conn, 18, 3400, at(19, 7))
    receive(schema_conn, 19, 3500, at(20, 7))
    # Seen after the cutoff: a late row for 20/11 and a revision of 19/11. The forecast must not use either.
    receive(schema_conn, 20, 3600, at(21, 10))
    receive(schema_conn, 19, 3550, at(21, 12))
    return schema_conn


def test_forecasts_use_only_what_was_seen_by_the_cutoff(conn):
    run_id, summary = forecast_baselines(conn, CUTOFF)

    # Naive: 19/11 as first seen (3500), not the later revision 3550 nor the late 20/11 row 3600.
    assert stored_forecasts(conn, NAIVE_MODEL) == {h: 3500.0 for h in range(1, 8)}
    # Seasonal naive: target-7 = 15/11 .. 21/11. 16/11 is unreported; 20/11 wasn't seen yet; 21/11 neither.
    assert stored_forecasts(conn, SEASONAL_NAIVE_MODEL) == {
        1: 3100.0,
        2: None,
        3: 3300.0,
        4: 3400.0,
        5: 3500.0,
        6: None,
        7: None,
    }
    assert (summary.forecasts_built, summary.forecasts_unavailable, summary.forecasts_inserted) == (14, 3, 14)

    distinct_inputs = conn.execute(
        "SELECT origin_date, input_as_of_at, forecast_run_id FROM forecasts GROUP BY 1, 2, 3"
    ).fetchall()
    assert distinct_inputs == [(date(2026, 11, 21), CUTOFF, run_id)]
    run = conn.execute(
        "SELECT job_type, status, cutoff_at, rows_out FROM pipeline_runs WHERE run_id = %s", (run_id,)
    ).fetchone()
    assert run == ("forecast", "succeeded", CUTOFF, 14)


def test_unavailable_seasonal_naive_is_stored_with_a_reason(conn):
    forecast_baselines(conn, CUTOFF)

    row = conn.execute(
        "SELECT point_forecast, unavailable_reason FROM forecasts "
        "WHERE model_version = %s AND target_date = '2026-11-23'",
        (SEASONAL_NAIVE_MODEL,),
    ).fetchone()
    assert row == (None, "target_minus_7_days_unreported")


def test_a_second_run_on_the_same_day_keeps_the_first_forecasts(conn):
    forecast_baselines(conn, CUTOFF)
    # Later the same day the pipeline knows more (the 20/11 row and the 19/11 revision).
    _, summary = forecast_baselines(conn, at(21, 18))

    assert summary.forecasts_inserted == 0
    assert stored_forecasts(conn, NAIVE_MODEL) == {h: 3500.0 for h in range(1, 8)}
    assert conn.execute("SELECT count(*) FROM forecasts").fetchone()[0] == 14


@pytest.mark.parametrize(
    "statement",
    ["UPDATE forecasts SET point_forecast = 1", "DELETE FROM forecasts"],
    ids=["update", "delete"],
)
def test_forecasts_cannot_be_changed_or_deleted(conn, statement):
    forecast_baselines(conn, CUTOFF)

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        conn.execute(statement)


def test_scoring_uses_only_reported_actuals_and_records_the_vintage(conn):
    forecast_baselines(conn, CUTOFF)
    receive(conn, 22, 3700, at(23, 7))  # reported
    # 23/11: only a row that fails a contract (min above modal) -> quarantined, so still unreported.
    receive(conn, 23, 3800, at(24, 7), min_price=3900)
    # 24/11: nothing at all.
    assert conn.execute("SELECT count(*) FROM quarantine").fetchone()[0] == 1

    run_id, added = score_provisional(conn)

    # Target 22/11: naive (3500) and seasonal naive (15/11 = 3100) are both scored; nothing else.
    assert added == 2
    vintage_22 = conn.execute(
        "SELECT vintage_id FROM price_vintages WHERE arrival_date = '2026-11-22'"
    ).fetchone()[0]
    scores = conn.execute(
        "SELECT f.model_version, f.target_date, s.status, s.actual_vintage_id, s.actual_price, s.error, "
        "s.score_run_id FROM forecast_scores s JOIN forecasts f USING (forecast_id) ORDER BY 1"
    ).fetchall()
    assert scores == [
        (NAIVE_MODEL, date(2026, 11, 22), "provisional", vintage_22, 3700, -200, run_id),
        (SEASONAL_NAIVE_MODEL, date(2026, 11, 22), "provisional", vintage_22, 3700, -600, run_id),
    ]

    # Counted, not scored: on 25/11, targets 23/11 and 24/11 have passed without a reported price.
    # Naive targets 22-28/11: 22 scored, 23 and 24 unreported, 25-28 awaiting.
    # Seasonal naive has values only for 22, 24, 25, 26/11: 22 scored, 24 unreported, 25 and 26 awaiting.
    assert scoring_counts(conn, today=date(2026, 11, 25)) == [
        ModelCounts(NAIVE_MODEL, 7, unavailable=0, scored=1, awaiting_target_date=4, unreported_so_far=2),
        ModelCounts(
            SEASONAL_NAIVE_MODEL, 7, unavailable=3, scored=1, awaiting_target_date=2, unreported_so_far=1
        ),
    ]


def test_rescoring_adds_nothing_and_a_revised_actual_adds_a_second_score(conn):
    forecast_baselines(conn, CUTOFF)
    receive(conn, 22, 3700, at(23, 7))
    score_provisional(conn)

    _, added_again = score_provisional(conn)
    assert added_again == 0

    receive(conn, 22, 3650, at(24, 7))  # the source edits 22/11
    _, added_after_revision = score_provisional(conn)

    assert added_after_revision == 2
    naive_scores = conn.execute(
        "SELECT s.actual_price, s.error FROM forecast_scores s JOIN forecasts f USING (forecast_id) "
        "WHERE f.model_version = %s ORDER BY s.score_id",
        (NAIVE_MODEL,),
    ).fetchall()
    # The first score (against 3700) is kept; the second uses the revised actual.
    assert naive_scores == [(3700, -200), (3650, -150)]
