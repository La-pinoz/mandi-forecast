"""M01 Day-1 probes of the data.gov.in OGD API (throwaway research code).

Each probe is one API call. Its request parameters (without the API key), status, headers and
full JSON response are saved to data/raw/ogd/<IST date>/<IST time>_<probe name>.json.

Run from the project root:  python research/day1/ogd_probe.py <probe-set>
Probe sets are defined in PROBE_SETS at the bottom; start with "basic".
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ogd"
BASE_URL = "https://api.data.gov.in/resource/"
CURRENT_DAILY_PRICE = "9ef84268-d588-465a-a308-a864a43d0070"
VARIETY_WISE_PRICE = "35985678-0d79-46b4-9ed6-6f13308a1d24"
IST = timezone(timedelta(hours=5, minutes=30))
SECONDS_BETWEEN_CALLS = 2
# Observed 2026-11-22: with the default "python-requests/x.y" User-Agent every data query hung
# and returned HTTP 502 after ~61 s, while curl with the same key got HTTP 200 in < 1 s.
# A neutral, non-default User-Agent avoids this.
HEADERS = {"User-Agent": "research-data-client/0.1"}


def load_api_key() -> str:
    """Read DATA_GOV_IN_API_KEY from the environment, falling back to the project .env file."""
    key = os.environ.get("DATA_GOV_IN_API_KEY", "").strip()
    env_file = PROJECT_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DATA_GOV_IN_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("DATA_GOV_IN_API_KEY is not set (environment or .env).")
    return key


def probe(name: str, resource_id: str, params: dict) -> dict:
    """Make one API call, save it, print a one-line summary, and return the parsed response."""
    api_key = load_api_key()
    request_params = {"api-key": api_key, "format": "json", **params}
    time.sleep(SECONDS_BETWEEN_CALLS)
    try:
        response = requests.get(BASE_URL + resource_id, params=request_params, headers=HEADERS, timeout=120)
    except requests.RequestException as error:
        # requests puts the full URL (including the key) in its error text, so scrub it.
        raise RuntimeError(f"{name}: request failed: {str(error).replace(api_key, '<API_KEY>')}") from None

    body_text = response.text.replace(api_key, "<API_KEY>")
    try:
        payload = json.loads(body_text)
    except ValueError:
        payload = {"non_json_body": body_text[:5000]}

    now_ist = datetime.now(IST)
    record = {
        "probe": name,
        "resource_id": resource_id,
        "params": params,  # without the API key
        "fetched_at_ist": now_ist.isoformat(timespec="seconds"),
        "status_code": response.status_code,
        "elapsed_seconds": round(response.elapsed.total_seconds(), 1),
        "response_headers": dict(response.headers),
        "response": payload,
    }
    out_file = RAW_DIR / now_ist.strftime("%Y-%m-%d") / f"{now_ist.strftime('%H%M%S')}_{name}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")

    records = payload.get("records", []) if isinstance(payload, dict) else []
    # The current-daily resource uses "arrival_date"; the variety-wise one uses "Arrival_Date".
    arrival_dates = sorted({r.get("arrival_date", r.get("Arrival_Date", "?")) for r in records})
    print(
        f"{name}: HTTP {response.status_code}, total={payload.get('total')}, "
        f"returned={len(records)}, arrival_dates={arrival_dates[:10]} -> {out_file.name}"
    )
    return payload


def ist_date(days_ago: int) -> str:
    """Date in the dd/mm/yyyy format the dataset uses for arrival_date (verify in 'basic')."""
    return (datetime.now(IST) - timedelta(days=days_ago)).strftime("%d/%m/%Y")


PROBE_SETS = {
    # What does the resource look like? Fields, total row count, date format.
    "basic": [
        ("no_filter_limit10", CURRENT_DAILY_PRICE, {"limit": 10}),
        ("tomato_all_states", CURRENT_DAILY_PRICE, {"limit": 1000, "filters[commodity]": "Tomato"}),
        ("west_bengal_all", CURRENT_DAILY_PRICE, {"limit": 1000, "filters[state]": "West Bengal"}),
    ],
    # Can past arrival dates be fetched? (M01 A2 and B5)
    "date_filter": [
        ("date_1_day_ago", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": ist_date(1)}),
        ("date_2_days_ago", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": ist_date(2)}),
        ("date_7_days_ago", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": ist_date(7)}),
        ("date_2023_06_15", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": "15/06/2023"}),
        ("date_today", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": ist_date(0)}),
    ],
    # Hypothesis from "date_filter": the arrival_date filter only matches the year.
    # If true, 01/01/2026 returns all rows and 22/11/2025 returns none. Also try ISO format.
    "date_year_check": [
        ("date_2026_01_01", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": "01/01/2026"}),
        ("date_2025_11_22", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": "22/11/2025"}),
        ("date_iso_yesterday", CURRENT_DAILY_PRICE, {"limit": 10, "filters[arrival_date]": "2026-11-21"}),
    ],
    # Every row of the resource in pages of 1000 (limit=10000 had its connection reset).
    "all_rows": [
        (f"all_rows_offset_{offset}", CURRENT_DAILY_PRICE, {"limit": 1000, "offset": offset})
        for offset in range(0, 10000, 1000)
    ],
    # Step 0 of backfill_plan.md: is the variety-wise resource's Arrival_Date filter exact?
    "variety_wise": [
        ("vw_no_filter_limit10", VARIETY_WISE_PRICE, {"limit": 10}),
        ("vw_tomato_2021_01_02", VARIETY_WISE_PRICE,
         {"limit": 1000, "filters[Commodity]": "Tomato", "filters[Arrival_Date]": "02/01/2021"}),
        ("vw_onion_2005_01_15", VARIETY_WISE_PRICE,
         {"limit": 1000, "filters[Commodity]": "Onion", "filters[Arrival_Date]": "15/01/2005"}),
        ("vw_tomato_1990_01_01", VARIETY_WISE_PRICE,
         {"limit": 1000, "filters[Commodity]": "Tomato", "filters[Arrival_Date]": "01/01/1990"}),
        ("vw_tomato_future", VARIETY_WISE_PRICE,
         {"limit": 1000, "filters[Commodity]": "Tomato", "filters[Arrival_Date]": ist_date(-30)}),
        ("vw_tomato_yesterday", VARIETY_WISE_PRICE,
         {"limit": 1000, "filters[Commodity]": "Tomato", "filters[Arrival_Date]": ist_date(1)}),
        ("vw_potato_yesterday", VARIETY_WISE_PRICE,
         {"limit": 1000, "filters[Commodity]": "Potato", "filters[Arrival_Date]": ist_date(1)}),
    ],
    # Step 0, continued: undocumented Market filter, and whether offset paging with filters
    # reproduces the single-page result (Tomato 02/01/2021 had 438 rows in one page).
    "variety_wise_paging": [
        ("vw_market_sirsa_yesterday", VARIETY_WISE_PRICE,
         {"limit": 1000, "filters[Commodity]": "Tomato", "filters[Arrival_Date]": ist_date(1),
          "filters[Market]": "Sirsa APMC"}),
    ]
    + [
        (f"vw_tomato_2021_01_02_offset_{offset}", VARIETY_WISE_PRICE,
         {"limit": 200, "offset": offset, "filters[Commodity]": "Tomato", "filters[Arrival_Date]": "02/01/2021"})
        for offset in range(0, 600, 200)
    ],
}


if __name__ == "__main__":
    set_name = sys.argv[1] if len(sys.argv) > 1 else ""
    if set_name not in PROBE_SETS:
        sys.exit(f"Usage: python research/day1/ogd_probe.py {{{'|'.join(PROBE_SETS)}}}")
    for probe_name, resource, probe_params in PROBE_SETS[set_name]:
        probe(probe_name, resource, probe_params)
