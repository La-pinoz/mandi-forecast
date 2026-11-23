"""Fill `series_day_status` (blueprint 5.3): one row per tracked series per calendar day.

    reported        a valid price is in prices_current for that series and date
    unreported      no usable observation: no row in the source, or only quarantined rows
    no_transaction  never assigned: the source has no NIL marker (M01), so a missing row can't be told
                    apart from a missing report. Missing data is never read as "market closed".

Each series runs from its first reported date to the latest date any tracked series reported. Anomaly flags
(validate/anomalies.py) are added to reported days. Dates where many series are unreported at once are
shown by the view `reporting_gaps_by_date` (migration 008).

The table is rebuilt from prices_current on every run, so late rows and revisions update earlier days.
"""

import pandas as pd
import psycopg

from mandi.validate.anomalies import flag_anomalies

TRACKED_PRICES = """
SELECT s.series_id, s.market || ' ' || s.commodity AS series_name, s.commodity,
       p.arrival_date, p.modal_price
FROM prices_current p
JOIN series s USING (source, market, commodity, variety, grade)
WHERE s.is_active
"""

UPSERT_DAY_STATUS = """
INSERT INTO series_day_status (series_id, arrival_date, status, is_anomaly, anomaly_reason)
VALUES (%(series_id)s, %(arrival_date)s, %(status)s, %(is_anomaly)s, %(anomaly_reason)s)
ON CONFLICT (series_id, arrival_date) DO UPDATE
SET status = EXCLUDED.status,
    is_anomaly = EXCLUDED.is_anomaly,
    anomaly_reason = EXCLUDED.anomaly_reason,
    updated_at = now()
WHERE (series_day_status.status, series_day_status.is_anomaly, series_day_status.anomaly_reason)
      IS DISTINCT FROM (EXCLUDED.status, EXCLUDED.is_anomaly, EXCLUDED.anomaly_reason)
"""


def build_day_status(prices: pd.DataFrame) -> pd.DataFrame:
    """Status rows for every series and calendar day. Input: one row per reported series-day with columns
    series_id, series_name, commodity, arrival_date (datetime64), modal_price (float).

    Output columns: series_id, arrival_date, status, is_anomaly, anomaly_reason.
    """
    flagged = flag_anomalies(prices)
    end_date = flagged["arrival_date"].max()

    parts = []
    for series_id, reported in flagged.groupby("series_id"):
        all_days = pd.DataFrame(
            {"arrival_date": pd.date_range(reported["arrival_date"].min(), end_date, freq="D")}
        )
        reported_days = reported[["arrival_date", "is_anomaly", "anomaly_reason"]].assign(status="reported")
        # Left join: calendar days with no reported price get NaN, i.e. no usable observation.
        days = all_days.merge(reported_days, on="arrival_date", how="left")
        days["series_id"] = series_id
        days["status"] = days["status"].fillna("unreported")
        days["is_anomaly"] = days["is_anomaly"].fillna(False).astype(bool)
        parts.append(days)
    return pd.concat(parts, ignore_index=True)[
        ["series_id", "arrival_date", "status", "is_anomaly", "anomaly_reason"]
    ]


def refresh_series_day_status(conn: psycopg.Connection) -> int:
    """Rebuild the status of every tracked series-day from prices_current (not committed).

    Returns the number of rows inserted or changed; rows already up to date are not rewritten.
    """
    rows = conn.execute(TRACKED_PRICES).fetchall()
    if not rows:
        return 0
    prices = pd.DataFrame(
        rows, columns=["series_id", "series_name", "commodity", "arrival_date", "modal_price"]
    )
    prices["arrival_date"] = pd.to_datetime(prices["arrival_date"])
    prices["modal_price"] = prices["modal_price"].astype(float)

    status = build_day_status(prices)
    params = []
    for row in status.itertuples(index=False):
        params.append(
            {
                "series_id": int(row.series_id),
                "arrival_date": row.arrival_date.date(),
                "status": row.status,
                "is_anomaly": bool(row.is_anomaly),
                "anomaly_reason": row.anomaly_reason if isinstance(row.anomaly_reason, str) else None,
            }
        )
    with conn.cursor() as cursor:
        cursor.executemany(UPSERT_DAY_STATUS, params)
        return cursor.rowcount
