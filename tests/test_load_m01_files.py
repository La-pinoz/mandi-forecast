"""Loading M01-format history files end to end (needs TEST_DATABASE_URL).

The files are small hand-made copies of the M01 format (research/day1/backfill_ogd.py), written to a
temporary folder, so each expected count can be worked out by reading the test.
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mandi.ingest.load_m01_files import SHORTLIST, load_history
from mandi.ingest.records import VARIETY_WISE_SOURCE

IST = timezone(timedelta(hours=5, minutes=30))
FETCHED_AT = "2026-11-22T21:40:45+05:30"


def record(market: str, commodity: str, variety: str, grade: str, prices: tuple, arrival: str) -> dict:
    min_price, max_price, modal_price = prices
    return {
        "Arrival_Date": arrival,
        "Commodity": commodity,
        "Commodity_Code": "78",
        "District": "North 24 Parganas",
        "Grade": grade,
        "Market": market,
        "Max_Price": max_price,
        "Min_Price": min_price,
        "Modal_Price": modal_price,
        "State": "West Bengal",
        "Variety": variety,
    }


def write_file(history_dir: Path, commodity: str, iso_date: str, records: list[dict]) -> None:
    day, month, year = iso_date[8:10], iso_date[5:7], iso_date[0:4]
    content = {
        "resource_id": VARIETY_WISE_SOURCE,
        "request_params": {
            "filters[State]": "West Bengal",
            "filters[Commodity]": commodity,
            "filters[Arrival_Date]": f"{day}/{month}/{year}",
        },
        "fetched_at_ist": FETCHED_AT,
        "total": len(records),
        "pages": 1,
        "updated_date": "2026-11-21T23:42:40Z",
        "records": records,
    }
    folder = history_dir / commodity
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{iso_date}.json").write_text(json.dumps(content), encoding="utf-8")


@pytest.fixture
def history_dir(tmp_path: Path) -> Path:
    write_file(
        tmp_path,
        "Tomato",
        "2025-11-01",
        [
            record("Barasat", "Tomato", "Other", "FAQ", ("2800", "3200", "3000"), "01/11/2025"),
            # Rename-overlap duplicate with identical prices (numbers this time): collapsed into one.
            record("Barasat APMC", "Tomato", "Other", "FAQ", (2800, 3200, 3000), "01/11/2025"),
            # Conflicting rows for a tracked key: neither is loaded.
            record("Habra", "Tomato", "Other", "FAQ", ("1700", "1900", "1800"), "01/11/2025"),
            record("Habra", "Tomato", "Other", "FAQ", ("1800", "2000", "1900"), "01/11/2025"),
            # Not a shortlisted market: kept in raw_snapshots only.
            record("Kalipur", "Tomato", "Deshi", "FAQ", ("1200", "1400", "1300"), "01/11/2025"),
        ],
    )
    write_file(
        tmp_path,
        "Onion",
        "2025-11-01",
        [
            record("Asansol APMC", "Onion", "Nasik", "FAQ", (1500, 1700, 1600), "01/11/2025"),
            # Shortlisted market and variety but another grade: not tracked.
            record("Asansol APMC", "Onion", "Nasik", "Non-FAQ", (1200, 1400, 1300), "01/11/2025"),
        ],
    )
    # A date the API returned no rows for: still stored as a snapshot.
    write_file(tmp_path, "Tomato", "2025-11-02", [])
    return tmp_path


def count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_first_load_stores_snapshots_series_and_vintages(schema_conn, history_dir):
    run_id, summary = load_history(schema_conn, history_dir)

    assert summary.files_loaded == 3
    assert summary.files_skipped == 0
    assert summary.tracked_records == 5  # Barasat x2, Habra x2, Asansol FAQ
    assert summary.duplicate_rows_collapsed == 1
    assert summary.conflicting_keys == [
        (VARIETY_WISE_SOURCE, "Habra", "Tomato", "Other", "FAQ", date(2025, 11, 1))
    ]
    assert summary.vintages_added == 2
    assert summary.prices_current_changed == 2

    assert count(schema_conn, "raw_snapshots") == 3
    assert count(schema_conn, "series") == len(SHORTLIST)
    assert count(schema_conn, "prices_current") == 2

    vintages = schema_conn.execute(
        """
        SELECT market, market_as_received, commodity, modal_price, first_seen_at, run_id
        FROM price_vintages ORDER BY commodity
        """
    ).fetchall()
    fetched_at = datetime(2026, 11, 22, 21, 40, 45, tzinfo=IST)
    assert vintages == [
        ("Asansol", "Asansol APMC", "Onion", 1600, fetched_at, run_id),
        ("Barasat", "Barasat", "Tomato", 3000, fetched_at, run_id),
    ]

    ingested = schema_conn.execute("SELECT DISTINCT ingested_at FROM raw_snapshots").fetchall()
    assert ingested == [(fetched_at,)]

    run = schema_conn.execute(
        "SELECT job_type, status, rows_in, rows_out, finished_at IS NOT NULL FROM pipeline_runs"
    ).fetchone()
    assert run == ("backfill_m01_files", "succeeded", 5, 2, True)


def test_rerun_adds_nothing(schema_conn, history_dir):
    load_history(schema_conn, history_dir)
    tables = ["raw_snapshots", "series", "price_vintages", "prices_current"]
    before = {table: count(schema_conn, table) for table in tables}

    second_run_id, summary = load_history(schema_conn, history_dir)

    assert summary.files_loaded == 0
    assert summary.files_skipped == 3
    assert summary.vintages_added == 0
    assert summary.prices_current_changed == 0
    assert {table: count(schema_conn, table) for table in tables} == before
    statuses = schema_conn.execute("SELECT run_id, status FROM pipeline_runs ORDER BY run_id").fetchall()
    assert statuses[-1] == (second_run_id, "succeeded")
    assert len(statuses) == 2


def test_a_broken_file_fails_the_run_and_keeps_finished_files(schema_conn, history_dir):
    # Sorted after Onion/2025-11-01.json and Tomato/2025-11-01.json, before Tomato/2025-11-02.json.
    (history_dir / "Tomato" / "2025-11-01b.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_history(schema_conn, history_dir)

    status, error_message = schema_conn.execute("SELECT status, error_message FROM pipeline_runs").fetchone()
    assert status == "failed"
    assert error_message.startswith("JSONDecodeError")
    assert count(schema_conn, "raw_snapshots") == 2  # the two files before the broken one

    # After the broken file is removed, a new run loads only what's missing.
    (history_dir / "Tomato" / "2025-11-01b.json").unlink()
    _, summary = load_history(schema_conn, history_dir)
    assert (summary.files_loaded, summary.files_skipped) == (1, 2)
    assert count(schema_conn, "prices_current") == 2


def test_an_empty_folder_is_an_error_not_an_empty_success(schema_conn, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_history(schema_conn, tmp_path)
    assert count(schema_conn, "pipeline_runs") == 0
