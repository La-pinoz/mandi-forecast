"""M01 history backfill from the variety-wise resource, one file per state, commodity and date (throwaway research code).

For each date in [start, end] it fetches every row for one commodity in one state (all its districts and markets)
and writes <out-root>/<state>/<commodity>/<YYYY-MM-DD>.json. A date whose file already exists is skipped, so the
job can be stopped and re-run. A date that still fails after retries is appended to
<out-root>/<state>/<commodity>/failed_dates.csv and gets no data file, so the next run tries it again.
A date with 0 rows gets a file with "total": 0.

Run from the project root, for example:
    python research/day1/backfill_ogd.py --start 01/01/2023 --end 31/01/2023 --state "West Bengal" --commodity Tomato

The same script re-fetches recent dates for the revision and cron-time checks (backfill_plan.md, Step 7) by
pointing --out-root at a new folder, e.g. --out-root data/raw/ogd_recheck/2026-11-23.

See backfill_plan.md (Steps 2-4) and docs/M01-day1-experiment.md (Step 0 results).
"""

import argparse
import csv
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from ogd_probe import BASE_URL, HEADERS, IST, PROJECT_ROOT, VARIETY_WISE_PRICE, load_api_key

DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "ogd_history"
PAGE_SIZE = 1000  # limit=10000 had its connection reset; 1000 is verified to work.
SECONDS_BETWEEN_CALLS = 1
RETRY_WAITS_SECONDS = [10, 30, 90]
REQUEST_TIMEOUT_SECONDS = 120


class FetchError(Exception):
    """A date could not be fetched completely and correctly. The message never contains the API key."""


def get_page(params: dict, api_key: str) -> dict:
    """Request one page, retrying on server errors, rate limiting, timeouts and connection resets.

    Raises FetchError if all retries fail. Raises RuntimeError at once on other HTTP errors (for example
    403 "Key not authorised"), because those would fail for every date and the run should stop.
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
            response = requests.get(
                BASE_URL + VARIETY_WISE_PRICE, params=request_params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except (requests.Timeout, requests.ConnectionError) as error:
            # requests puts the full URL (including the key) in its error text, so scrub it.
            last_error = f"{type(error).__name__}: {str(error).replace(api_key, '<API_KEY>')}"
            continue

        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            continue
        if response.status_code != 200:
            body = response.text.replace(api_key, "<API_KEY>")[:300]
            raise RuntimeError(f"HTTP {response.status_code} (not retryable, stopping the run): {body}")

        try:
            payload = response.json()
        except ValueError:
            last_error = "HTTP 200 with a non-JSON body"
            continue
        if "total" not in payload:
            last_error = f"HTTP 200 without 'total': {response.text.replace(api_key, '<API_KEY>')[:200]}"
            continue
        return payload

    raise FetchError(f"failed after {len(RETRY_WAITS_SECONDS)} retries: {last_error}")


def fetch_date(state: str, commodity: str, arrival_date: str, api_key: str) -> dict:
    """Fetch every row for one state and commodity on one date (dd/mm/yyyy), paging until all `total` rows arrive.

    Returns the record to save. Raises FetchError if the pages are inconsistent, because the date-loop
    design relies on getting exactly the rows for the requested state, date and commodity.
    """
    filters = {"filters[State]": state, "filters[Commodity]": commodity, "filters[Arrival_Date]": arrival_date}
    records = []
    pages = 0
    first_page = get_page({"limit": PAGE_SIZE, "offset": 0, **filters}, api_key)
    total = int(first_page["total"])
    page = first_page
    while True:
        pages += 1
        page_records = page.get("records", [])
        records.extend(page_records)
        if int(page["total"]) != total:
            raise FetchError(f"total changed while paging: {total} then {page['total']}")
        if len(records) >= total:
            break
        if not page_records:
            raise FetchError(f"empty page at offset {len(records)} with total={total}")
        page = get_page({"limit": PAGE_SIZE, "offset": len(records), **filters}, api_key)

    if len(records) != total:
        raise FetchError(f"received {len(records)} rows but total={total}")
    wrong_rows = [
        r for r in records
        if r.get("Arrival_Date") != arrival_date or r.get("Commodity") != commodity or r.get("State") != state
    ]
    if wrong_rows:
        raise FetchError(f"{len(wrong_rows)} rows do not match the requested filters, e.g. {wrong_rows[0]}")

    return {
        "resource_id": VARIETY_WISE_PRICE,
        "request_params": filters,
        "fetched_at_ist": datetime.now(IST).isoformat(timespec="seconds"),
        "total": total,
        "pages": pages,
        "updated_date": first_page.get("updated_date"),
        "records": records,
    }


def append_failure(failed_file: Path, arrival_date: date, error: str) -> None:
    is_new_file = not failed_file.exists()
    with failed_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new_file:
            writer.writerow(["arrival_date", "error", "failed_at_ist"])
        writer.writerow([arrival_date.isoformat(), error, datetime.now(IST).isoformat(timespec="seconds")])


def backfill(state: str, commodity: str, start: date, end: date, out_root: Path) -> None:
    """Fetch and save every date from start to end (inclusive) for one state and commodity, then print a summary."""
    api_key = load_api_key()
    out_dir = out_root / state / commodity
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_file = out_dir / "failed_dates.csv"

    fetched = skipped = zero_rows = failed = 0
    day = start
    while day <= end:
        out_file = out_dir / f"{day.isoformat()}.json"
        if out_file.exists():
            skipped += 1
        else:
            try:
                result = fetch_date(state, commodity, day.strftime("%d/%m/%Y"), api_key)
            except FetchError as error:
                failed += 1
                append_failure(failed_file, day, str(error))
                print(f"{commodity} {day}: FAILED, {error}")
            else:
                # Write to a temporary file first so a crash never leaves a partial file that a re-run would skip.
                temp_file = out_file.with_suffix(".json.tmp")
                temp_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                temp_file.replace(out_file)
                fetched += 1
                if result["total"] == 0:
                    zero_rows += 1
                print(f"{commodity} {day}: {result['total']} rows, {result['pages']} page(s)")
        day += timedelta(days=1)

    print(
        f"\n{commodity} {start} to {end}: fetched {fetched} (of which 0 rows: {zero_rows}), "
        f"skipped (already on disk) {skipped}, failed {failed}"
    )
    if failed:
        print(f"Failed dates are listed in {failed_file}. Re-run the same command to retry them.")


def parse_dd_mm_yyyy(text: str) -> date:
    return datetime.strptime(text, "%d/%m/%Y").date()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=parse_dd_mm_yyyy, help="first arrival date, DD/MM/YYYY")
    parser.add_argument("--end", required=True, type=parse_dd_mm_yyyy, help="last arrival date, DD/MM/YYYY")
    parser.add_argument("--state", required=True, help='exact State string, e.g. "West Bengal"')
    parser.add_argument("--commodity", required=True, help='exact Commodity string, e.g. "Tomato"')
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help=f"default: {DEFAULT_OUT_ROOT}")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start is after --end")
    backfill(args.state, args.commodity, args.start, args.end, args.out_root)
