"""M03: load the saved M01 history files into the database, through the same vintage logic the daily job uses.

Run from the project root (uses DATABASE_URL from the environment or .env):
    uv run python -m mandi.ingest.load_m01_files

Input: data/raw/ogd_history/West Bengal/<Commodity>/<YYYY-MM-DD>.json, one saved API response per state,
commodity and arrival date, written by research/day1/backfill_ogd.py (docs/M01-day1-experiment.md).

Flow (one pipeline_runs row per run):
1. Add the 8 shortlisted series to `series` (if they aren't there yet).
2. For each file, in its own transaction:
   a. skip it if an earlier run already stored it (same payload hash and same fetch time);
   b. store the response in raw_snapshots, with ingested_at = the file's fetched_at_ist;
   c. parse the records of the shortlisted series, set aside keys whose rows disagree, and record vintages
      with first_seen_at = fetched_at_ist.
3. Refresh prices_current from price_vintages, then mark the run succeeded.

A crash leaves every finished file committed and the run marked failed; running again skips those files.

History loaded this way is a LATEST-SNAPSHOT vintage (what the API returned on 22-23/11/2026), not the
values first published on each date.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from mandi.ingest.records import VARIETY_WISE_SOURCE, canonical_market_name, parse_record, split_duplicates
from mandi.store.db import get_connection
from mandi.store.raw_snapshots import payload_hash, save_raw_snapshot
from mandi.store.runs import finish_run, start_run
from mandi.store.vintages import record_vintages, refresh_prices_current

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = PROJECT_ROOT / "data" / "raw" / "ogd_history" / "West Bengal"
JOB_TYPE = "backfill_m01_files"

# The owner-approved M01 shortlist (docs/M01-day1-experiment.md, section 3.10): market (canonical name,
# spelled exactly as the source does), commodity, variety, grade, district.
SHORTLIST = [
    ("Barasat", "Tomato", "Other", "FAQ", "North 24 Parganas"),
    ("Habra", "Tomato", "Other", "FAQ", "North 24 Parganas"),
    ("Sealdah Koley Market", "Tomato", "Other", "FAQ", "Kolkata"),
    ("Karsiyang(Matigara)", "Tomato", "Other", "FAQ", "Darjeeling"),
    ("Asansol", "Onion", "Nasik", "FAQ", "Paschim Bardhaman"),
    ("Barasat", "Onion", "Other", "FAQ", "North 24 Parganas"),
    ("Medinipur(West)", "Onion", "Onion", "FAQ", "Medinipur(W)"),
    ("Bankura Sadar", "Onion", "Nasik", "FAQ", "Bankura"),
]


@dataclass
class LoadSummary:
    files_loaded: int = 0
    files_skipped: int = 0
    tracked_records: int = 0
    duplicate_rows_collapsed: int = 0
    # Keys with two rows that disagree on price. Not loaded; M04's rules decide what happens to them.
    conflicting_keys: list[tuple] = field(default_factory=list)
    # (file name, error) for tracked records whose price or date couldn't be parsed. Not loaded.
    unparseable_records: list[tuple[str, str]] = field(default_factory=list)
    vintages_added: int = 0
    prices_current_changed: int = 0


def add_shortlisted_series(conn: psycopg.Connection) -> None:
    """Insert the shortlist into `series`; series already there are left unchanged."""
    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO series (source, market, commodity, variety, grade, district)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, market, commodity, variety, grade) DO NOTHING
            """,
            [(VARIETY_WISE_SOURCE, *series) for series in SHORTLIST],
        )


def tracked_series_keys(conn: psycopg.Connection) -> set[tuple[str, str, str, str, str]]:
    """(source, market, commodity, variety, grade) of every active series."""
    rows = conn.execute(
        "SELECT source, market, commodity, variety, grade FROM series WHERE is_active"
    ).fetchall()
    return set(rows)


def stored_snapshots(conn: psycopg.Connection) -> set[tuple[str, datetime]]:
    """(payload_hash, ingested_at) of every stored raw snapshot: how a file loaded earlier is recognised."""
    rows = conn.execute("SELECT payload_hash, ingested_at FROM raw_snapshots").fetchall()
    return set(rows)


def load_file(
    conn: psycopg.Connection,
    run_id: int,
    path: Path,
    tracked: set[tuple[str, str, str, str, str]],
    already_stored: set[tuple[str, datetime]],
    summary: LoadSummary,
) -> None:
    """Store one M01 file as a raw snapshot and its tracked observations as vintages (not committed)."""
    content = json.loads(path.read_text(encoding="utf-8"))
    source = content["resource_id"]
    fetched_at = datetime.fromisoformat(content["fetched_at_ist"])
    if fetched_at.tzinfo is None:
        raise ValueError(f"{path}: fetched_at_ist has no time zone: {content['fetched_at_ist']}")
    # The response part of the file; resource_id, request_params and fetched_at_ist have their own columns.
    payload = {
        "total": content["total"],
        "pages": content["pages"],
        "updated_date": content["updated_date"],
        "records": content["records"],
    }

    if (payload_hash(payload), fetched_at) in already_stored:
        summary.files_skipped += 1
        return
    save_raw_snapshot(conn, run_id, source, content["request_params"], payload, fetched_at)

    observations = []
    for record in content["records"]:
        series_key = (
            source,
            canonical_market_name(record["Market"]),
            record["Commodity"],
            record["Variety"],
            record["Grade"],
        )
        if series_key not in tracked:
            continue
        summary.tracked_records += 1
        try:
            observations.append(parse_record(record, source))
        except ValueError as error:
            summary.unparseable_records.append((path.name, str(error)))

    unique, conflicting_keys = split_duplicates(observations)
    rows_in_conflicts = 0
    for observation in observations:
        if observation.key() in conflicting_keys:
            rows_in_conflicts += 1
    summary.duplicate_rows_collapsed += len(observations) - len(unique) - rows_in_conflicts
    summary.conflicting_keys.extend(conflicting_keys)
    summary.vintages_added += record_vintages(conn, unique, first_seen_at=fetched_at, run_id=run_id)
    summary.files_loaded += 1


def load_history(conn: psycopg.Connection, history_dir: Path = HISTORY_DIR) -> tuple[int, LoadSummary]:
    """Load every <commodity>/<date>.json file under `history_dir`. Returns (run_id, summary).

    Any error marks the run 'failed' (with the error message) and is raised again.
    """
    paths = sorted(history_dir.glob("*/*.json"))
    if not paths:
        raise FileNotFoundError(f"No <commodity>/<date>.json files under {history_dir}")

    run_id = start_run(conn, JOB_TYPE)
    summary = LoadSummary()
    try:
        add_shortlisted_series(conn)
        conn.commit()
        tracked = tracked_series_keys(conn)
        already_stored = stored_snapshots(conn)
        for path in paths:
            load_file(conn, run_id, path, tracked, already_stored, summary)
            conn.commit()
        summary.prices_current_changed = refresh_prices_current(conn)
        conn.commit()
    except Exception as error:
        conn.rollback()
        finish_run(
            conn,
            run_id,
            "failed",
            rows_in=summary.tracked_records,
            rows_out=summary.vintages_added,
            error_message=f"{type(error).__name__}: {error}",
        )
        raise

    finish_run(conn, run_id, "succeeded", rows_in=summary.tracked_records, rows_out=summary.vintages_added)
    return run_id, summary


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    with get_connection() as conn:
        run_id, summary = load_history(conn)

    print(f"Run {run_id} succeeded.")
    print(f"Files loaded: {summary.files_loaded}; already loaded, skipped: {summary.files_skipped}")
    print(f"Records of shortlisted series: {summary.tracked_records}")
    print(f"Duplicate rows with identical prices, collapsed: {summary.duplicate_rows_collapsed}")
    print(f"Unparseable records, not loaded: {len(summary.unparseable_records)}")
    for file_name, error in summary.unparseable_records:
        print(f"  {file_name}: {error}")
    print(f"Keys with conflicting rows, not loaded (left for M04): {len(summary.conflicting_keys)}")
    for key in summary.conflicting_keys:
        _source, market, commodity, variety, grade, arrival_date = key
        print(f"  {arrival_date} {market} / {commodity} / {variety} / {grade}")
    print(f"Vintages added: {summary.vintages_added}")
    print(f"prices_current rows inserted or changed: {summary.prices_current_changed}")


if __name__ == "__main__":
    main()
