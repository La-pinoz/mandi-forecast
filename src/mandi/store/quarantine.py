"""Store records that failed a hard contract in `quarantine` (blueprint 5.4)."""

import psycopg
from psycopg.types.json import Jsonb

from mandi.validate.contracts import QuarantinedRow

# A record quarantined for a reason once is not stored again by later runs (a replay, or the daily recheck
# fetching the same bad record again), so the table holds each distinct problem once, with the run that
# first found it.
INSERT_IF_NEW = """
INSERT INTO quarantine (run_id, row_hash, source, record, reason_code, detail)
SELECT %(run_id)s, %(row_hash)s, %(source)s, %(record)s, %(reason_code)s, %(detail)s
WHERE NOT EXISTS (
    SELECT 1 FROM quarantine
    WHERE source = %(source)s AND row_hash = %(row_hash)s AND reason_code = %(reason_code)s
)
"""


def save_quarantined(conn: psycopg.Connection, run_id: int, source: str, rows: list[QuarantinedRow]) -> int:
    """Insert the rows not already quarantined for the same reason (not committed). Returns rows inserted."""
    if not rows:
        return 0
    params = []
    for row in rows:
        params.append(
            {
                "run_id": run_id,
                "row_hash": row.row_hash,
                "source": source,
                "record": Jsonb(row.record),
                "reason_code": row.reason_code,
                "detail": row.detail,
            }
        )
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_IF_NEW, params)
        return cursor.rowcount
