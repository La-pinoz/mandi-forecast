"""Record versions of observations in `price_vintages` and keep `prices_current` in step (blueprint 5.2).

Rule: an incoming observation is compared with the LATEST stored vintage of its key (latest =
largest `first_seen_at`, then largest `vintage_id`). Same row hash -> nothing is added. Different hash, or
no vintage yet -> a new vintage. So 3000 -> 3200 -> 3000 gives three vintages, while seeing 3000 twice in a
row gives one.
"""

import hashlib
from datetime import datetime

import psycopg

from mandi.ingest.records import Observation

# The WHERE clause does the "compare with the latest vintage" step inside the database, so each
# observation needs no separate SELECT round trip. `IS DISTINCT FROM` is true when the subquery returns
# no row (NULL), i.e. the key has no vintage yet.
INSERT_IF_CHANGED = """
INSERT INTO price_vintages (
    source, market, commodity, variety, grade, arrival_date, market_as_received, district,
    min_price, max_price, modal_price, row_hash, first_seen_at, run_id
)
SELECT %(source)s, %(market)s, %(commodity)s, %(variety)s, %(grade)s, %(arrival_date)s,
       %(market_as_received)s, %(district)s, %(min_price)s, %(max_price)s, %(modal_price)s,
       %(row_hash)s, %(first_seen_at)s, %(run_id)s
WHERE %(row_hash)s IS DISTINCT FROM (
    SELECT row_hash
    FROM price_vintages
    WHERE source = %(source)s AND market = %(market)s AND commodity = %(commodity)s
      AND variety = %(variety)s AND grade = %(grade)s AND arrival_date = %(arrival_date)s
    ORDER BY first_seen_at DESC, vintage_id DESC
    LIMIT 1
)
"""

# Copy the latest vintage of every key into prices_current. Rows whose vintage is already current are
# left alone (the WHERE on DO UPDATE), so re-running it changes nothing.
REFRESH_PRICES_CURRENT = """
INSERT INTO prices_current (
    source, market, commodity, variety, grade, arrival_date, district,
    min_price, max_price, modal_price, vintage_id
)
SELECT DISTINCT ON (source, market, commodity, variety, grade, arrival_date)
       source, market, commodity, variety, grade, arrival_date, district,
       min_price, max_price, modal_price, vintage_id
FROM price_vintages
ORDER BY source, market, commodity, variety, grade, arrival_date, first_seen_at DESC, vintage_id DESC
ON CONFLICT (source, market, commodity, variety, grade, arrival_date) DO UPDATE
SET district = EXCLUDED.district,
    min_price = EXCLUDED.min_price,
    max_price = EXCLUDED.max_price,
    modal_price = EXCLUDED.modal_price,
    vintage_id = EXCLUDED.vintage_id
WHERE prices_current.vintage_id <> EXCLUDED.vintage_id
"""


def row_hash(observation: Observation) -> str:
    """SHA-256 of min|max|modal, each written with 2 decimals (the column precision).

    So "1800" (string, older rows) and 1800 (number, newer rows) hash the same. Market name as received
    and district are not hashed: a version is a change in the prices.
    """
    text = f"{observation.min_price:.2f}|{observation.max_price:.2f}|{observation.modal_price:.2f}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_vintages(
    conn: psycopg.Connection, observations: list[Observation], first_seen_at: datetime, run_id: int
) -> int:
    """Add a vintage for each observation whose prices differ from its key's latest vintage (not committed).

    Returns the number of vintages added. Observations must have unique keys. If this run already stored a
    DIFFERENT value for a key, the unique constraint (key, run_id) raises UniqueViolation: one run seeing
    two values for one observation is an error, not something to drop silently.
    """
    if not observations:
        return 0
    params = []
    for observation in observations:
        params.append(
            {
                "source": observation.source,
                "market": observation.market,
                "commodity": observation.commodity,
                "variety": observation.variety,
                "grade": observation.grade,
                "arrival_date": observation.arrival_date,
                "market_as_received": observation.market_as_received,
                "district": observation.district,
                "min_price": observation.min_price,
                "max_price": observation.max_price,
                "modal_price": observation.modal_price,
                "row_hash": row_hash(observation),
                "first_seen_at": first_seen_at,
                "run_id": run_id,
            }
        )
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_IF_CHANGED, params)
        # After executemany, psycopg's rowcount is the total number of rows inserted.
        return cursor.rowcount


def refresh_prices_current(conn: psycopg.Connection) -> int:
    """Bring prices_current up to date with the latest vintage of every key (not committed).

    Returns the number of prices_current rows inserted or changed. It looks at all vintages, so a run that
    crashed before calling it is repaired by the next run.
    """
    with conn.cursor() as cursor:
        cursor.execute(REFRESH_PRICES_CURRENT)
        return cursor.rowcount
