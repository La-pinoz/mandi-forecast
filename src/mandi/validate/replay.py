"""M04: run every stored raw snapshot through validation again, so history follows the same rules as new data.

Run from the project root (uses DATABASE_URL from the environment or .env):
    uv run python -m mandi.validate.replay

Flow (one pipeline_runs row, job type 'validate_replay'):
1. Read the tracked series and every raw snapshot, oldest first (vintages must be compared in time order).
2. For each snapshot, `process_snapshot`: contract failures go to quarantine (unless an earlier run already
   quarantined the same record for the same reason); passing observations become vintages only if they
   differ from the latest stored version, with first_seen_at = the snapshot's ingested_at.
3. Refresh prices_current, then rebuild series_day_status (status + anomaly flags).
4. Mark the run succeeded and print the data-quality report.

Everything is one transaction: a failure leaves the database unchanged and the run marked 'failed'.
Running it again adds nothing (only its own pipeline_runs row).
"""

from dataclasses import dataclass
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from mandi.ingest.snapshot import process_snapshot, tracked_series_keys
from mandi.store.db import get_connection
from mandi.store.runs import finish_run, start_run
from mandi.store.vintages import refresh_prices_current
from mandi.validate.day_status import refresh_series_day_status
from mandi.validate.report import print_data_quality_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOB_TYPE = "validate_replay"

ALL_SNAPSHOTS = """
SELECT source, payload -> 'records', ingested_at
FROM raw_snapshots
ORDER BY ingested_at, run_id, payload_hash
"""


@dataclass
class ReplaySummary:
    snapshots: int = 0
    tracked_records: int = 0
    rows_failed: int = 0
    rows_quarantined: int = 0
    vintages_added: int = 0
    prices_current_changed: int = 0
    day_status_changed: int = 0


def replay_snapshots(conn: psycopg.Connection) -> tuple[int, ReplaySummary]:
    """Validate every raw snapshot again and store the outcome. Returns (run_id, summary).

    Any error rolls back all of the replay's work, marks the run 'failed' and is raised again.
    """
    run_id = start_run(conn, JOB_TYPE)
    summary = ReplaySummary()
    try:
        tracked = tracked_series_keys(conn)
        for source, records, ingested_at in conn.execute(ALL_SNAPSHOTS).fetchall():
            result = process_snapshot(conn, run_id, source, records, ingested_at, tracked)
            summary.snapshots += 1
            summary.tracked_records += result.tracked_records
            summary.rows_failed += result.rows_failed
            summary.rows_quarantined += result.rows_quarantined
            summary.vintages_added += result.vintages_added
        summary.prices_current_changed = refresh_prices_current(conn)
        summary.day_status_changed = refresh_series_day_status(conn)
        conn.commit()
    except Exception as error:
        conn.rollback()
        finish_run(conn, run_id, "failed", error_message=f"{type(error).__name__}: {error}")
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


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    with get_connection() as conn:
        run_id, summary = replay_snapshots(conn)
        print(f"Run {run_id} succeeded.")
        print(
            f"Snapshots replayed: {summary.snapshots}; records of tracked series: {summary.tracked_records}"
        )
        print(f"Contract failures (record, reason): {summary.rows_failed}")
        print(f"Newly quarantined (not already in quarantine): {summary.rows_quarantined}")
        print(f"Vintages added: {summary.vintages_added}")
        print(f"prices_current rows inserted or changed: {summary.prices_current_changed}")
        print(f"series_day_status rows inserted or changed: {summary.day_status_changed}")
        print_data_quality_report(conn)


if __name__ == "__main__":
    main()
