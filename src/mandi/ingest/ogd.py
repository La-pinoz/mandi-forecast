"""M05: fetch the data.gov.in variety-wise resource and store it through the same steps as history.

Run from the project root (uses DATA_GOV_IN_API_KEY and DATABASE_URL from the environment or .env):
    uv run python -m mandi.ingest.ogd ingest                                    # yesterday and today (IST)
    uv run python -m mandi.ingest.ogd recheck                                   # the 7 dates before those
    uv run python -m mandi.ingest.ogd backfill --start 2026-11-01 --end 2026-11-07

Each command is one pipeline_runs row (job types 'daily_ingest', 'recheck', 'backfill'). Flow:
1. Read the active series; the commodities to request come from them (Tomato, Onion).
2. For each arrival date and commodity, one request per page (all West Bengal markets), with retries. The
   records are sorted into a fixed order, because the API returns them in a varying order.
3. If this exact response (same request, same payload hash) is already in raw_snapshots, skip it: it was
   stored and validated by an earlier run. Otherwise store it, then `process_snapshot` (M04): contract
   failures go to quarantine, new or changed prices become vintages with first_seen_at = the fetch time.
   Each response is committed on its own.
4. Refresh prices_current and series_day_status, then mark the run succeeded.

Any error stops the run and marks it failed (responses already committed stay). Running the same command
again re-requests everything, but stores and adds only what changed, so a re-run is safe.

Why two daily jobs: Agmarknet lets markets edit the last 7 days, and M01 saw a date still gaining rows 1.5
days later. `ingest` gets the newest dates; `recheck` asks for the 7 dates before them again, so late rows
and edited prices become new vintages. Keeping them as separate runs shows how much each one finds.
"""

import argparse
import http.client
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psycopg
from dotenv import load_dotenv

from mandi.ingest.records import VARIETY_WISE_SOURCE
from mandi.ingest.snapshot import IST, process_snapshot, tracked_series_keys
from mandi.store.db import get_connection
from mandi.store.raw_snapshots import payload_hash, save_raw_snapshot, snapshot_already_stored
from mandi.store.runs import finish_run, start_run
from mandi.store.vintages import refresh_prices_current
from mandi.validate.day_status import refresh_series_day_status

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "https://api.data.gov.in/resource/"
STATE = "West Bengal"
# The server answers requests with Python's default User-Agent with HTTP 502 after ~61 s (M01), so send
# a neutral one of our own.
USER_AGENT = "mandi-price-pipeline/0.1"
PAGE_SIZE = 1000  # verified in M01; limit=10000 had its connection reset
SECONDS_BETWEEN_CALLS = 1
RETRY_WAITS_SECONDS = [10, 30, 90]
REQUEST_TIMEOUT_SECONDS = 120

INGEST_DAYS = 2  # yesterday and today
RECHECK_DAYS = 7  # the 7 dates before the ingest window


class FetchError(Exception):
    """A response could not be fetched completely and correctly. The message never contains the API key."""


@dataclass
class FetchedSnapshot:
    """All pages of one request, in the same shape as the M01 history files."""

    request_params: dict  # filters only; no API key, no paging
    fetched_at: datetime
    payload: dict  # total, pages, updated_date, records


@dataclass
class IngestSummary:
    responses: int = 0
    responses_already_stored: int = 0
    tracked_records: int = 0
    rows_quarantined: int = 0
    vintages_added: int = 0
    prices_current_changed: int = 0
    day_status_changed: int = 0


def load_api_key() -> str:
    """DATA_GOV_IN_API_KEY from the environment (call load_dotenv first to read .env)."""
    api_key = os.environ.get("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "DATA_GOV_IN_API_KEY is not set. Add it to .env locally or as a GitHub Actions secret."
        )
    return api_key


def request_once(params: dict) -> tuple[int, str]:
    """One HTTP GET to the variety-wise resource. Returns (status code, body text).

    Raises OSError (URLError, TimeoutError, ConnectionResetError, ...) or http.client.HTTPException (e.g.
    IncompleteRead) when no complete HTTP response arrives.
    """
    url = BASE_URL + VARIETY_WISE_SOURCE + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as error:
        # urllib raises for 4xx/5xx; the status and body are still needed to decide whether to retry.
        return error.code, error.read().decode("utf-8", errors="replace")


def get_page(params: dict, api_key: str) -> dict:
    """Request one page, retrying on HTTP 429/5xx, timeouts, connection errors and malformed bodies.

    Raises FetchError when every retry fails, and RuntimeError at once on other HTTP errors (e.g. 403 for a
    bad key), because those would fail for every request.
    """
    request_params = {"api-key": api_key, "format": "json", **params}
    last_error = ""
    for attempt in range(len(RETRY_WAITS_SECONDS) + 1):
        if attempt > 0:
            wait = RETRY_WAITS_SECONDS[attempt - 1]
            print(f"    retry {attempt} in {wait} s after: {last_error}")
            time.sleep(wait)
        time.sleep(SECONDS_BETWEEN_CALLS)

        try:
            status, body = request_once(request_params)
        except (OSError, http.client.HTTPException) as error:
            # Network failures: a timeout, or a connection reset while connecting (wrapped in URLError) or
            # while reading the body (not wrapped). Scrub the key in case a message ever includes the URL.
            last_error = f"{type(error).__name__}: {str(error).replace(api_key, '<API_KEY>')}"
            continue

        if status == 429 or status >= 500:
            last_error = f"HTTP {status}"
            continue
        if status != 200:
            raise RuntimeError(f"HTTP {status} (not retryable): {body.replace(api_key, '<API_KEY>')[:300]}")

        try:
            page = json.loads(body)
        except ValueError:
            last_error = "HTTP 200 with a non-JSON body"
            continue
        if "total" not in page:
            last_error = f"HTTP 200 without 'total': {body.replace(api_key, '<API_KEY>')[:200]}"
            continue
        return page

    raise FetchError(f"failed after {len(RETRY_WAITS_SECONDS)} retries: {last_error}")


def fetch_date(commodity: str, arrival_date: date, api_key: str) -> FetchedSnapshot:
    """Fetch every West Bengal row for one commodity and arrival date, paging until `total` rows arrived.

    Raises FetchError if the pages disagree with each other or a row doesn't match the filters: everything
    downstream relies on a response holding exactly the requested state, commodity and date.
    """
    arrival_text = arrival_date.strftime("%d/%m/%Y")
    filters = {
        "filters[State]": STATE,
        "filters[Commodity]": commodity,
        "filters[Arrival_Date]": arrival_text,
    }
    first_page = get_page({"limit": PAGE_SIZE, "offset": 0, **filters}, api_key)
    total = int(first_page["total"])
    records = list(first_page.get("records", []))
    pages = 1
    while len(records) < total:
        page = get_page({"limit": PAGE_SIZE, "offset": len(records), **filters}, api_key)
        pages += 1
        if int(page["total"]) != total:
            raise FetchError(
                f"{commodity} {arrival_text}: total changed while paging ({total}, then {page['total']})"
            )
        page_records = page.get("records", [])
        if not page_records:
            raise FetchError(f"{commodity} {arrival_text}: empty page at offset {len(records)} of {total}")
        records.extend(page_records)
    fetched_at = datetime.now(IST)

    if len(records) != total:
        raise FetchError(f"{commodity} {arrival_text}: received {len(records)} rows but total={total}")
    for record in records:
        if (
            record.get("State") != STATE
            or record.get("Commodity") != commodity
            or record.get("Arrival_Date") != arrival_text
        ):
            raise FetchError(f"{commodity} {arrival_text}: a row does not match the filters: {record}")

    # The API returns the same rows in a different order from one request to the next (seen 2026-11-23:
    # Onion 22/11, two fetches 4 minutes apart). Sorting gives the same content the same payload hash, so
    # an unchanged answer is recognised as already stored. Row order carries no information.
    sorted_records = sorted(records, key=lambda record: json.dumps(record, sort_keys=True))
    payload = {
        "total": total,
        "pages": pages,
        "updated_date": first_page.get("updated_date"),
        "records": sorted_records,
    }
    return FetchedSnapshot(request_params=filters, fetched_at=fetched_at, payload=payload)


def store_snapshot(
    conn: psycopg.Connection,
    run_id: int,
    snapshot: FetchedSnapshot,
    tracked: set[tuple[str, str, str, str, str]],
    summary: IngestSummary,
) -> str:
    """Store one fetched response and its validated rows (not committed). Returns a one-line log message."""
    summary.responses += 1
    hash_value = payload_hash(snapshot.payload)
    if snapshot_already_stored(conn, VARIETY_WISE_SOURCE, snapshot.request_params, hash_value):
        summary.responses_already_stored += 1
        return "same response already stored, skipped"

    save_raw_snapshot(
        conn, run_id, VARIETY_WISE_SOURCE, snapshot.request_params, snapshot.payload, snapshot.fetched_at
    )
    result = process_snapshot(
        conn, run_id, VARIETY_WISE_SOURCE, snapshot.payload["records"], snapshot.fetched_at, tracked
    )
    summary.tracked_records += result.tracked_records
    summary.rows_quarantined += result.rows_quarantined
    summary.vintages_added += result.vintages_added
    return (
        f"{result.tracked_records} rows of tracked series, {result.vintages_added} new vintages, "
        f"{result.rows_quarantined} newly quarantined"
    )


def ingest_dates(
    conn: psycopg.Connection, job_type: str, arrival_dates: list[date], api_key: str
) -> tuple[int, IngestSummary]:
    """Fetch and store every tracked commodity for each arrival date. Returns (run_id, summary).

    Any error marks the run 'failed' (with the error message) and is raised again.
    """
    run_id = start_run(conn, job_type)
    summary = IngestSummary()
    try:
        tracked = tracked_series_keys(conn)
        if not tracked:
            raise RuntimeError("No active series in `series`; load them first (M03).")
        commodities = sorted({series_key[2] for series_key in tracked})
        for arrival_date in arrival_dates:
            for commodity in commodities:
                snapshot = fetch_date(commodity, arrival_date, api_key)
                message = store_snapshot(conn, run_id, snapshot, tracked, summary)
                conn.commit()
                print(f"{commodity} {arrival_date}: {snapshot.payload['total']} rows; {message}")
        summary.prices_current_changed = refresh_prices_current(conn)
        summary.day_status_changed = refresh_series_day_status(conn)
        conn.commit()
    except Exception as error:
        conn.rollback()
        finish_run(
            conn,
            run_id,
            "failed",
            rows_in=summary.tracked_records,
            rows_out=summary.vintages_added,
            rows_quarantined=summary.rows_quarantined,
            error_message=f"{type(error).__name__}: {error}",
        )
        raise

    finish_run(
        conn,
        run_id,
        "succeeded",
        rows_in=summary.tracked_records,
        rows_out=summary.vintages_added,
        rows_quarantined=summary.rows_quarantined,
    )
    return run_id, summary


def date_range(start: date, end: date) -> list[date]:
    """Every calendar date from start to end, both included."""
    days = []
    day = start
    while day <= end:
        days.append(day)
        day += timedelta(days=1)
    return days


def dates_for_command(command: str, today: date, start: date | None, end: date | None) -> list[date]:
    """Arrival dates for a command. `today` is the IST date the job runs on.

    ingest: today-1 and today. recheck: today-8 to today-2 (the 7 dates before the ingest window).
    """
    if command == "ingest":
        return date_range(today - timedelta(days=INGEST_DAYS - 1), today)
    if command == "recheck":
        recheck_end = today - timedelta(days=INGEST_DAYS)
        return date_range(recheck_end - timedelta(days=RECHECK_DAYS - 1), recheck_end)
    if start is None or end is None or start > end:
        raise ValueError("backfill needs --start and --end with start <= end")
    return date_range(start, end)


JOB_TYPES = {"ingest": "daily_ingest", "recheck": "recheck", "backfill": "backfill"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch data.gov.in prices into the database.")
    parser.add_argument("command", choices=sorted(JOB_TYPES))
    parser.add_argument(
        "--start", type=date.fromisoformat, help="backfill only: first arrival date, YYYY-MM-DD"
    )
    parser.add_argument("--end", type=date.fromisoformat, help="backfill only: last arrival date, YYYY-MM-DD")
    args = parser.parse_args()

    today = datetime.now(IST).date()
    try:
        arrival_dates = dates_for_command(args.command, today, args.start, args.end)
    except ValueError as error:
        parser.error(str(error))

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = load_api_key()
    print(f"{args.command}: arrival dates {arrival_dates[0]} to {arrival_dates[-1]} (today is {today} IST)")
    with get_connection() as conn:
        run_id, summary = ingest_dates(conn, JOB_TYPES[args.command], arrival_dates, api_key)

    print(f"\nRun {run_id} succeeded.")
    print(f"Responses: {summary.responses}; already stored, skipped: {summary.responses_already_stored}")
    print(f"Rows of tracked series in new responses: {summary.tracked_records}")
    print(f"Newly quarantined: {summary.rows_quarantined}")
    print(f"Vintages added: {summary.vintages_added}")
    print(f"prices_current rows inserted or changed: {summary.prices_current_changed}")
    print(f"series_day_status rows inserted or changed: {summary.day_status_changed}")


if __name__ == "__main__":
    main()
