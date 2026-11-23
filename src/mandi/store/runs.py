"""Start and finish `pipeline_runs` rows. Both functions commit, so a run's row is saved even if the
run's own work is later rolled back."""

from datetime import datetime

import psycopg


def start_run(conn: psycopg.Connection, job_type: str, cutoff_at: datetime | None = None) -> int:
    """Insert a 'running' row and commit. Returns its run_id."""
    row = conn.execute(
        "INSERT INTO pipeline_runs (job_type, cutoff_at, status) VALUES (%s, %s, 'running') RETURNING run_id",
        (job_type, cutoff_at),
    ).fetchone()
    conn.commit()
    return row[0]


def finish_run(
    conn: psycopg.Connection,
    run_id: int,
    status: str,
    rows_in: int | None = None,
    rows_out: int | None = None,
    rows_quarantined: int | None = None,
    error_message: str | None = None,
) -> None:
    """Set the final status ('succeeded' or 'failed'), counts and finish time, and commit."""
    conn.execute(
        """
        UPDATE pipeline_runs
        SET status = %s, finished_at = now(), rows_in = %s, rows_out = %s,
            rows_quarantined = %s, error_message = %s
        WHERE run_id = %s
        """,
        (status, rows_in, rows_out, rows_quarantined, error_message, run_id),
    )
    conn.commit()
