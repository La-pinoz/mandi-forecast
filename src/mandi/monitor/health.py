"""M10: data-health tables for the dashboard (blueprint 14.3).

- Last run of each job type, with its status and error, and when that job last succeeded.
- Quarantined rows by reason, days per series by status, widespread-gap dates: the same queries as the M04
  data-quality report (`mandi.validate.report`), returned as tables instead of printed.
"""

from dataclasses import dataclass

import pandas as pd
import psycopg

from mandi.validate.report import GAP_DATES, QUARANTINE_BY_REASON, STATUS_BY_SERIES

DAY_STATUSES = ["reported", "no_transaction", "unreported"]

LAST_RUNS = """
SELECT latest.job_type, latest.status, latest.started_at, latest.error_message, succeeded.last_succeeded_at
FROM (
    SELECT DISTINCT ON (job_type) job_type, status, started_at, error_message
    FROM pipeline_runs
    ORDER BY job_type, started_at DESC, run_id DESC
) latest
LEFT JOIN (
    SELECT job_type, max(finished_at) AS last_succeeded_at
    FROM pipeline_runs
    WHERE status = 'succeeded'
    GROUP BY job_type
) succeeded ON succeeded.job_type = latest.job_type
ORDER BY latest.job_type
"""


@dataclass
class DataHealth:
    last_runs: pd.DataFrame  # job_type, latest_status, latest_started_at, latest_error, last_succeeded_at
    quarantine_by_reason: pd.DataFrame  # reason_code, rows
    day_status: pd.DataFrame  # crop, market, one column per DAY_STATUSES, anomaly_flags
    widespread_gap_dates: pd.DataFrame  # arrival_date, series_unreported, series_tracked


def load_data_health(conn: psycopg.Connection) -> DataHealth:
    """Read the four data-health tables."""
    last_runs = pd.DataFrame(
        conn.execute(LAST_RUNS).fetchall(),
        columns=["job_type", "latest_status", "latest_started_at", "latest_error", "last_succeeded_at"],
    )
    quarantine = pd.DataFrame(conn.execute(QUARANTINE_BY_REASON).fetchall(), columns=["reason_code", "rows"])
    gap_dates = pd.DataFrame(
        conn.execute(GAP_DATES).fetchall(), columns=["arrival_date", "series_unreported", "series_tracked"]
    )

    status_rows = pd.DataFrame(
        conn.execute(STATUS_BY_SERIES).fetchall(), columns=["crop", "market", "status", "days", "anomalies"]
    )
    # One row per series; a status that never occurs (the source has no no-transaction marker, M01) shows 0.
    day_status = status_rows.pivot_table(
        index=["crop", "market"], columns="status", values="days", aggfunc="sum", fill_value=0
    )
    day_status = day_status.reindex(columns=DAY_STATUSES, fill_value=0)
    day_status["anomaly_flags"] = status_rows.groupby(["crop", "market"])["anomalies"].sum()
    day_status = day_status.reset_index()
    day_status.columns.name = None
    return DataHealth(last_runs, quarantine, day_status, gap_dates)
