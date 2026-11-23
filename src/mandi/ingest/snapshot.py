"""Validate the records of one raw snapshot and store the outcome: the step every load path shares.

Used by the M01 file loader (ingest/load_m01_files.py), the replay of stored snapshots (validate/replay.py)
and, from M05, the daily job, so history and live data follow exactly the same rules.

Flow for one snapshot: contracts (validate/contracts.py) -> failures into `quarantine` -> passing
observations into `price_vintages` with first_seen_at = the snapshot's ingested_at.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import psycopg

from mandi.store.quarantine import save_quarantined
from mandi.store.vintages import record_vintages
from mandi.validate.contracts import check_records

# India has no daylight saving time, so a fixed offset is exact (and needs no tzdata package on Windows).
IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class SnapshotResult:
    tracked_records: int
    duplicate_rows_collapsed: int
    rows_failed: int  # (record, reason) pairs that failed a contract in this snapshot
    rows_quarantined: int  # of those, the ones not already in quarantine from an earlier run
    vintages_added: int


def tracked_series_keys(conn: psycopg.Connection) -> set[tuple[str, str, str, str, str]]:
    """(source, market, commodity, variety, grade) of every active series."""
    rows = conn.execute(
        "SELECT source, market, commodity, variety, grade FROM series WHERE is_active"
    ).fetchall()
    return set(rows)


def process_snapshot(
    conn: psycopg.Connection,
    run_id: int,
    source: str,
    records: list[dict],
    ingested_at: datetime,
    tracked: set[tuple[str, str, str, str, str]],
) -> SnapshotResult:
    """Validate one snapshot's records and store quarantined rows and new vintages (not committed).

    The future-date rule compares arrival dates with the IST date of `ingested_at`, so a replay judges each
    snapshot by the day it was received, not by today.
    """
    if ingested_at.tzinfo is None:
        raise ValueError("ingested_at must be timezone-aware")
    snapshot_date = ingested_at.astimezone(IST).date()

    result = check_records(records, source, tracked, snapshot_date)
    rows_quarantined = save_quarantined(conn, run_id, source, result.quarantined)
    vintages_added = record_vintages(conn, result.observations, first_seen_at=ingested_at, run_id=run_id)
    return SnapshotResult(
        tracked_records=result.tracked_records,
        duplicate_rows_collapsed=result.duplicate_rows_collapsed,
        rows_failed=len(result.quarantined),
        rows_quarantined=rows_quarantined,
        vintages_added=vintages_added,
    )
