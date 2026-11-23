"""Print the data-quality report: day status counts, quarantine by reason, anomaly flags, gap dates.

Run from the project root (uses DATABASE_URL from the environment or .env):
    uv run python -m mandi.validate.report
"""

from pathlib import Path

import psycopg
from dotenv import load_dotenv

from mandi.store.db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STATUS_BY_SERIES = """
SELECT s.commodity, s.market, d.status, count(*), count(*) FILTER (WHERE d.is_anomaly)
FROM series_day_status d JOIN series s USING (series_id)
GROUP BY s.commodity, s.market, d.status
ORDER BY s.commodity, s.market, d.status
"""

QUARANTINE_BY_REASON = (
    "SELECT reason_code, count(*) FROM quarantine GROUP BY reason_code ORDER BY reason_code"
)

QUARANTINE_ROWS = """
SELECT record ->> 'Arrival_Date', record ->> 'Market', record ->> 'Commodity', reason_code, detail
FROM quarantine
ORDER BY to_date(record ->> 'Arrival_Date', 'DD/MM/YYYY'), record ->> 'Market', detail
"""

ANOMALY_KINDS = """
SELECT split_part(anomaly_reason, ':', 1) AS kind, count(*)
FROM series_day_status WHERE is_anomaly
GROUP BY kind ORDER BY kind
"""

GAP_DATES = """
SELECT arrival_date, series_unreported, series_tracked
FROM reporting_gaps_by_date WHERE is_widespread_gap
ORDER BY arrival_date
"""


def print_data_quality_report(conn: psycopg.Connection) -> None:
    print("\nDays per series by status (anomaly flags among reported days):")
    for commodity, market, status, days, anomalies in conn.execute(STATUS_BY_SERIES).fetchall():
        flags = f", {anomalies} flagged" if status == "reported" else ""
        print(f"  {commodity:6} {market:22} {status:12} {days:5}{flags}")

    print("\nQuarantine rows by reason:")
    for reason_code, rows in conn.execute(QUARANTINE_BY_REASON).fetchall():
        print(f"  {reason_code:22} {rows}")
    for arrival_date, market, commodity, reason_code, detail in conn.execute(QUARANTINE_ROWS).fetchall():
        print(f"    {arrival_date} {market} {commodity}: {reason_code} ({detail})")

    print("\nAnomaly flags by kind:")
    for kind, flags in conn.execute(ANOMALY_KINDS).fetchall():
        print(f"  {kind:14} {flags}")

    gap_dates = conn.execute(GAP_DATES).fetchall()
    print(f"\nWidespread-gap dates (at least half the tracked series unreported): {len(gap_dates)}")
    for arrival_date, unreported, tracked in gap_dates:
        print(f"  {arrival_date}: {unreported} of {tracked} unreported")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    with get_connection() as conn:
        print_data_quality_report(conn)


if __name__ == "__main__":
    main()
