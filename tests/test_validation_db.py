"""Validation against the database: quarantine, replay, day status, gap view (needs TEST_DATABASE_URL)."""

from datetime import date, datetime, timedelta, timezone

import psycopg
import pytest

from mandi.ingest.records import VARIETY_WISE_SOURCE
from mandi.ingest.snapshot import process_snapshot, tracked_series_keys
from mandi.store.raw_snapshots import save_raw_snapshot
from mandi.store.runs import start_run
from mandi.store.vintages import refresh_prices_current
from mandi.validate.day_status import refresh_series_day_status
from mandi.validate.replay import replay_snapshots

IST = timezone(timedelta(hours=5, minutes=30))
RECEIVED_AT = datetime(2026, 11, 22, 21, 40, tzinfo=IST)
FIRST_DAY = date(2026, 10, 1)


def record(market: str, day: int, modal: int) -> dict:
    """A tomato record for `market` on FIRST_DAY + day; min and max are 200 either side of the modal."""
    arrival = FIRST_DAY + timedelta(days=day)
    return {
        "Arrival_Date": arrival.strftime("%d/%m/%Y"),
        "Commodity": "Tomato",
        "District": "North 24 Parganas",
        "Grade": "FAQ",
        "Market": market,
        "Max_Price": modal + 200,
        "Min_Price": modal - 200,
        "Modal_Price": modal,
        "State": "West Bengal",
        "Variety": "Other",
    }


def add_series(conn: psycopg.Connection, *markets: str) -> None:
    for market in markets:
        conn.execute(
            "INSERT INTO series (source, market, commodity, variety, grade, district) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (VARIETY_WISE_SOURCE, market, "Tomato", "Other", "FAQ", "North 24 Parganas"),
        )
    conn.commit()


def store_snapshot(conn: psycopg.Connection, run_id: int, records: list[dict]) -> None:
    """Store a raw snapshot without validating it (as M03 did), for the replay tests."""
    payload = {"total": len(records), "records": records}
    save_raw_snapshot(
        conn, run_id, VARIETY_WISE_SOURCE, {"filters[State]": "West Bengal"}, payload, RECEIVED_AT
    )
    conn.commit()


def count(conn: psycopg.Connection, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def day_status(conn: psycopg.Connection) -> dict[int, tuple[str, bool]]:
    rows = conn.execute(
        "SELECT arrival_date, status, is_anomaly FROM series_day_status ORDER BY 1"
    ).fetchall()
    return {
        (arrival_date - FIRST_DAY).days: (status, is_anomaly) for arrival_date, status, is_anomaly in rows
    }


def test_a_missing_day_is_unreported_and_never_quarantined(schema_conn):
    # The source's NIL representation (M01): a no-trade day simply has no row.
    add_series(schema_conn, "Barasat")
    run_id = start_run(schema_conn, "test")
    records = [record("Barasat", 0, 2000), record("Barasat", 2, 2000)]  # day 1 has no row

    process_snapshot(
        schema_conn, run_id, VARIETY_WISE_SOURCE, records, RECEIVED_AT, tracked_series_keys(schema_conn)
    )
    refresh_prices_current(schema_conn)
    refresh_series_day_status(schema_conn)

    assert day_status(schema_conn) == {
        0: ("reported", False),
        1: ("unreported", False),
        2: ("reported", False),
    }
    assert count(schema_conn, "quarantine") == 0


def test_a_spike_is_flagged_but_stays_in_prices_current(schema_conn):
    add_series(schema_conn, "Barasat")
    run_id = start_run(schema_conn, "test")
    records = [record("Barasat", day, 2000) for day in range(20)]
    records[15] = record("Barasat", 15, 4000)

    process_snapshot(
        schema_conn, run_id, VARIETY_WISE_SOURCE, records, RECEIVED_AT, tracked_series_keys(schema_conn)
    )
    refresh_prices_current(schema_conn)
    refresh_series_day_status(schema_conn)

    flagged = [day for day, (_status, is_anomaly) in day_status(schema_conn).items() if is_anomaly]
    assert flagged == [15]
    spike_price = schema_conn.execute(
        "SELECT modal_price FROM prices_current WHERE arrival_date = %s", (FIRST_DAY + timedelta(days=15),)
    ).fetchone()
    assert spike_price == (4000,)
    assert count(schema_conn, "prices_current") == 20
    assert count(schema_conn, "quarantine") == 0


def test_replay_quarantines_history_and_a_second_replay_adds_nothing(schema_conn):
    add_series(schema_conn, "Barasat", "Habra")
    load_run = start_run(schema_conn, "test_m03_style_load")
    store_snapshot(
        schema_conn,
        load_run,
        [
            record("Barasat", 0, 2000),
            record("Barasat APMC", 0, 2000),  # identical duplicate under the new name: one observation
            record("Habra", 0, 1800),
            record("Habra", 0, 1900),  # conflicting duplicate: both rows quarantined
            {**record("Barasat", 1, 2000), "Modal_Price": "NR"},  # bad_type
            {**record("Kalipur", 1, 2000), "Min_Price": 0},  # untracked market: not checked
        ],
    )

    run_id, summary = replay_snapshots(schema_conn)

    assert (summary.snapshots, summary.tracked_records) == (1, 5)
    assert (summary.rows_failed, summary.rows_quarantined, summary.vintages_added) == (3, 3, 1)
    quarantined = schema_conn.execute(
        "SELECT record ->> 'Market', record ->> 'Modal_Price', reason_code, run_id "
        "FROM quarantine ORDER BY 1, 2"
    ).fetchall()
    assert quarantined == [
        ("Barasat", "NR", "bad_type", run_id),
        ("Habra", "1800", "conflicting_duplicate", run_id),
        ("Habra", "1900", "conflicting_duplicate", run_id),
    ]
    run = schema_conn.execute(
        "SELECT job_type, status, rows_in, rows_out, rows_quarantined FROM pipeline_runs WHERE run_id = %s",
        (run_id,),
    ).fetchone()
    assert run == ("validate_replay", "succeeded", 5, 1, 3)
    # Barasat reported day 0 and nothing usable on day 1; Habra's only day had conflicting rows, so it has no
    # reported day and therefore no status rows.
    assert day_status(schema_conn) == {0: ("reported", False)}

    tables = ["quarantine", "price_vintages", "prices_current", "series_day_status"]
    before = {table: count(schema_conn, table) for table in tables}
    _second_run_id, second = replay_snapshots(schema_conn)

    assert (second.rows_failed, second.rows_quarantined, second.vintages_added) == (3, 0, 0)
    assert (second.prices_current_changed, second.day_status_changed) == (0, 0)
    assert {table: count(schema_conn, table) for table in tables} == before


def test_a_failed_replay_changes_nothing_and_is_recorded(schema_conn):
    add_series(schema_conn, "Barasat")
    load_run = start_run(schema_conn, "test_m03_style_load")
    store_snapshot(schema_conn, load_run, [record("Barasat", 0, 2000), {"Market": "Barasat"}])  # no Commodity

    with pytest.raises(KeyError):
        replay_snapshots(schema_conn)

    status, error_message = schema_conn.execute(
        "SELECT status, error_message FROM pipeline_runs WHERE job_type = 'validate_replay'"
    ).fetchone()
    assert status == "failed"
    assert error_message.startswith("KeyError")
    assert count(schema_conn, "price_vintages") == 0


def test_gap_view_flags_dates_where_half_the_series_are_unreported(schema_conn):
    add_series(schema_conn, "Barasat", "Habra", "Sealdah Koley Market")
    run_id = start_run(schema_conn, "test")
    # Day 1: two of three series missing. Day 2: one of three missing.
    missing = {("Habra", 1), ("Sealdah Koley Market", 1), ("Habra", 2)}
    records = []
    for market in ["Barasat", "Habra", "Sealdah Koley Market"]:
        for day in range(4):
            if (market, day) not in missing:
                records.append(record(market, day, 2000))

    process_snapshot(
        schema_conn, run_id, VARIETY_WISE_SOURCE, records, RECEIVED_AT, tracked_series_keys(schema_conn)
    )
    refresh_prices_current(schema_conn)
    refresh_series_day_status(schema_conn)

    rows = schema_conn.execute(
        "SELECT arrival_date, series_tracked, series_unreported, is_widespread_gap "
        "FROM reporting_gaps_by_date ORDER BY 1"
    ).fetchall()
    assert [
        (arrival_date.day, tracked, unreported, gap) for arrival_date, tracked, unreported, gap in rows
    ] == [
        (1, 3, 0, False),
        (2, 3, 2, True),  # 2 of 3 unreported
        (3, 3, 1, False),  # 1 of 3 unreported: not widespread
        (4, 3, 0, False),
    ]
