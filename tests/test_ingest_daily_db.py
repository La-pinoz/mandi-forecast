"""End to end on saved M01 responses: fetch -> raw_snapshots -> validate -> vintages.

Needs TEST_DATABASE_URL.

Fixtures (tests/fixtures/ogd/) are real responses for arrival date 21/11/2026, West Bengal, copied unchanged:
- first_fetch/: fetched 22/11 23:27 IST. Tracked rows: Tomato at Sealdah Koley Market 3200, Barasat 3000,
  Karsiyang(Matigara) 3100; Onion at Barasat 5500.
- recheck/: the same request at 23/11 10:41 IST. Late rows for two more tracked series (Habra Tomato 2700,
  Bankura Sadar Onion 4700); the four prices above are unchanged.

Only the network call is replaced: `fetch_date` reads the fixture file instead of calling the API.
"""

import json
from datetime import date, datetime
from pathlib import Path

import psycopg
import pytest

from mandi.ingest import ogd
from mandi.ingest.load_m01_files import add_shortlisted_series

FIXTURES = Path(__file__).parent / "fixtures" / "ogd"
ARRIVAL_DATE = date(2026, 11, 21)


def use_saved_responses(monkeypatch: pytest.MonkeyPatch, folder: str) -> None:
    def fetch_from_file(commodity: str, arrival_date: date, api_key: str) -> ogd.FetchedSnapshot:
        content = json.loads((FIXTURES / folder / commodity / f"{arrival_date}.json").read_text("utf-8"))
        return ogd.FetchedSnapshot(
            request_params=content["request_params"],
            fetched_at=datetime.fromisoformat(content["fetched_at_ist"]),
            payload={
                "total": content["total"],
                "pages": content["pages"],
                "updated_date": content["updated_date"],
                "records": content["records"],
            },
        )

    monkeypatch.setattr(ogd, "fetch_date", fetch_from_file)


def table_counts(conn: psycopg.Connection) -> dict[str, int]:
    counts = {}
    for table in ["raw_snapshots", "price_vintages", "prices_current", "quarantine", "series_day_status"]:
        counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return counts


def modal_prices(conn: psycopg.Connection) -> dict[tuple[str, str], int]:
    rows = conn.execute("SELECT market, commodity, modal_price FROM prices_current").fetchall()
    return {(market, commodity): int(modal) for market, commodity, modal in rows}


@pytest.fixture
def conn(schema_conn: psycopg.Connection) -> psycopg.Connection:
    add_shortlisted_series(schema_conn)
    schema_conn.commit()
    return schema_conn


def test_ingest_stores_validates_and_records_the_run(conn, monkeypatch) -> None:
    use_saved_responses(monkeypatch, "first_fetch")

    run_id, summary = ogd.ingest_dates(conn, "daily_ingest", [ARRIVAL_DATE], api_key="unused")

    assert summary.responses == 2
    assert summary.responses_already_stored == 0
    assert summary.tracked_records == 4
    assert summary.vintages_added == 4
    assert modal_prices(conn) == {
        ("Sealdah Koley Market", "Tomato"): 3200,
        ("Barasat", "Tomato"): 3000,
        ("Karsiyang(Matigara)", "Tomato"): 3100,
        ("Barasat", "Onion"): 5500,
    }
    assert table_counts(conn) == {
        "raw_snapshots": 2,
        "price_vintages": 4,
        "prices_current": 4,
        "quarantine": 0,
        "series_day_status": 4,
    }
    # The market name as received is kept next to the canonical one.
    received = conn.execute("SELECT DISTINCT market_as_received FROM price_vintages WHERE market = 'Barasat'")
    assert received.fetchall() == [("Barasat APMC",)]
    # first_seen_at is the time each response was fetched (from the fixture files), not the run time.
    first_seen = dict(conn.execute("SELECT DISTINCT commodity, first_seen_at FROM price_vintages").fetchall())
    assert first_seen == {
        "Tomato": datetime.fromisoformat("2026-11-22T23:27:37+05:30"),
        "Onion": datetime.fromisoformat("2026-11-23T00:19:41+05:30"),
    }

    run = conn.execute(
        "SELECT job_type, status, rows_in, rows_out, rows_quarantined, finished_at IS NOT NULL "
        "FROM pipeline_runs WHERE run_id = %s",
        (run_id,),
    ).fetchone()
    assert run == ("daily_ingest", "succeeded", 4, 4, 0, True)


def test_running_the_same_day_twice_adds_nothing(conn, monkeypatch) -> None:
    use_saved_responses(monkeypatch, "first_fetch")
    ogd.ingest_dates(conn, "daily_ingest", [ARRIVAL_DATE], api_key="unused")
    counts_after_first_run = table_counts(conn)

    run_id, summary = ogd.ingest_dates(conn, "daily_ingest", [ARRIVAL_DATE], api_key="unused")

    assert summary.responses == 2
    assert summary.responses_already_stored == 2
    assert summary.vintages_added == 0
    assert summary.prices_current_changed == 0
    assert summary.day_status_changed == 0
    assert table_counts(conn) == counts_after_first_run
    run = conn.execute("SELECT status, rows_out FROM pipeline_runs WHERE run_id = %s", (run_id,)).fetchone()
    assert run == ("succeeded", 0)


def test_recheck_turns_late_rows_into_new_vintages_and_keeps_unchanged_prices(conn, monkeypatch) -> None:
    use_saved_responses(monkeypatch, "first_fetch")
    ogd.ingest_dates(conn, "daily_ingest", [ARRIVAL_DATE], api_key="unused")

    use_saved_responses(monkeypatch, "recheck")
    recheck_run_id, summary = ogd.ingest_dates(conn, "recheck", [ARRIVAL_DATE], api_key="unused")

    # A new response (more rows, new updated_date) is stored; only the two late rows are new versions.
    assert summary.responses_already_stored == 0
    assert summary.tracked_records == 6
    assert summary.vintages_added == 2
    counts = table_counts(conn)
    assert counts["raw_snapshots"] == 4
    assert counts["price_vintages"] == 6
    assert counts["prices_current"] == 6
    assert modal_prices(conn)[("Habra", "Tomato")] == 2700
    assert modal_prices(conn)[("Bankura Sadar", "Onion")] == 4700

    late_rows = conn.execute(
        "SELECT market, commodity FROM price_vintages WHERE run_id = %s ORDER BY market", (recheck_run_id,)
    ).fetchall()
    assert late_rows == [("Bankura Sadar", "Onion"), ("Habra", "Tomato")]
    # No price was edited, so no observation has a second version.
    revised = conn.execute(
        "SELECT count(*) FROM (SELECT 1 FROM price_vintages "
        "GROUP BY market, commodity, variety, grade, arrival_date HAVING count(*) > 1) AS keys"
    ).fetchone()[0]
    assert revised == 0


def test_empty_responses_for_both_crops_in_one_run_are_both_stored(conn, monkeypatch) -> None:
    # The first live run failed on this (2026-11-23): today's Tomato and Onion answers were identical.
    def empty_answer(commodity: str, arrival_date: date, api_key: str) -> ogd.FetchedSnapshot:
        filters = {
            "filters[State]": "West Bengal",
            "filters[Commodity]": commodity,
            "filters[Arrival_Date]": arrival_date.strftime("%d/%m/%Y"),
        }
        payload = {"total": 0, "pages": 1, "updated_date": "2026-11-22T23:34:48Z", "records": []}
        return ogd.FetchedSnapshot(filters, datetime.fromisoformat("2026-11-23T16:00:00+05:30"), payload)

    monkeypatch.setattr(ogd, "fetch_date", empty_answer)
    _run_id, summary = ogd.ingest_dates(conn, "daily_ingest", [date(2026, 11, 23)], api_key="unused")

    assert summary.responses == 2
    assert table_counts(conn)["raw_snapshots"] == 2
    assert table_counts(conn)["price_vintages"] == 0


def test_a_failed_fetch_marks_the_run_failed_and_keeps_earlier_responses(conn, monkeypatch) -> None:
    use_saved_responses(monkeypatch, "first_fetch")
    fetch_from_file = ogd.fetch_date

    def fail_for_tomato(commodity: str, arrival_date: date, api_key: str) -> ogd.FetchedSnapshot:
        if commodity == "Tomato":
            raise ogd.FetchError("failed after 3 retries: HTTP 502")
        return fetch_from_file(commodity, arrival_date, api_key)

    monkeypatch.setattr(ogd, "fetch_date", fail_for_tomato)
    with pytest.raises(ogd.FetchError):
        ogd.ingest_dates(conn, "daily_ingest", [ARRIVAL_DATE], api_key="unused")

    status, error_message = conn.execute("SELECT status, error_message FROM pipeline_runs").fetchone()
    assert status == "failed"
    assert error_message == "FetchError: failed after 3 retries: HTTP 502"
    # Onion is fetched first (commodities in name order) and was committed before Tomato failed. The
    # final refresh never ran, so prices_current is filled by the next successful run.
    counts = table_counts(conn)
    assert counts["raw_snapshots"] == 1
    assert counts["price_vintages"] == 1
    assert counts["prices_current"] == 0


def test_no_transaction_is_left_open_while_waiting_for_the_api(conn, monkeypatch) -> None:
    # Regression (2026-11-24): the first fetch waited ~2.5 min in retries inside the transaction opened by
    # the series SELECT; Neon ended the idle-in-transaction session, so the run could not be marked failed.
    use_saved_responses(monkeypatch, "first_fetch")
    fetch_from_file = ogd.fetch_date
    status_during_fetch = []

    def record_status(commodity: str, arrival_date: date, api_key: str) -> ogd.FetchedSnapshot:
        status_during_fetch.append(conn.info.transaction_status)
        return fetch_from_file(commodity, arrival_date, api_key)

    monkeypatch.setattr(ogd, "fetch_date", record_status)
    ogd.ingest_dates(conn, "daily_ingest", [ARRIVAL_DATE], api_key="unused")

    idle = psycopg.pq.TransactionStatus.IDLE
    assert status_during_fetch == [idle, idle]


def test_no_active_series_is_an_error(schema_conn, monkeypatch) -> None:
    use_saved_responses(monkeypatch, "first_fetch")
    with pytest.raises(RuntimeError, match="No active series"):
        ogd.ingest_dates(schema_conn, "daily_ingest", [ARRIVAL_DATE], api_key="unused")
    assert schema_conn.execute("SELECT status FROM pipeline_runs").fetchone() == ("failed",)
