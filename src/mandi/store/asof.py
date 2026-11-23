"""As-of reconstruction (blueprint 5.2): what did the pipeline know at a given moment?

For each observation key, the answer is the latest vintage with `first_seen_at <= cutoff`. A key first
seen after the cutoff is absent, and a revision made after the cutoff is ignored.
"""

from datetime import datetime

import psycopg
from psycopg.rows import dict_row

PRICES_AS_OF = """
SELECT DISTINCT ON (source, market, commodity, variety, grade, arrival_date)
       source, market, commodity, variety, grade, arrival_date, district,
       min_price, max_price, modal_price, vintage_id, first_seen_at
FROM price_vintages
WHERE first_seen_at <= %s
ORDER BY source, market, commodity, variety, grade, arrival_date, first_seen_at DESC, vintage_id DESC
"""


def prices_as_of(conn: psycopg.Connection, cutoff: datetime) -> list[dict]:
    """Return one dict per key: the version known at `cutoff`, sorted by key.

    `cutoff` must include a time zone; a naive datetime would be read in the database session's zone,
    which could silently shift the cutoff by hours.
    """
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware, e.g. datetime(..., tzinfo=ZoneInfo('Asia/Kolkata'))")
    with conn.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(PRICES_AS_OF, (cutoff,)).fetchall()
