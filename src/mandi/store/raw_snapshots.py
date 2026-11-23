"""Store API responses, as received, in the append-only `raw_snapshots` table."""

import hashlib
import json
from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb


def payload_hash(payload: dict) -> str:
    """SHA-256 of the payload as JSON with sorted keys, so the same content always gives the same hash."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot_already_stored(
    conn: psycopg.Connection, source: str, request_params: dict, hash_value: str
) -> bool:
    """True if any run already stored this exact response (same source, request and payload hash).

    The payload includes the dataset's `updated_date`, which changes once a day, so the same request on a
    later day is stored again even when its records are unchanged: that is the evidence that we checked.
    """
    row = conn.execute(
        """
        SELECT 1 FROM raw_snapshots
        WHERE source = %s AND request_params = %s AND payload_hash = %s
        LIMIT 1
        """,
        (source, Jsonb(request_params), hash_value),
    ).fetchone()
    return row is not None


def save_raw_snapshot(
    conn: psycopg.Connection,
    run_id: int,
    source: str,
    request_params: dict,
    payload: dict,
    ingested_at: datetime,
) -> str:
    """Insert one response (not committed). Returns its payload hash.

    `request_params` must not contain the API key. Saving the same request and payload twice in one run
    raises UniqueViolation (primary key run_id + source + request_params + payload_hash, migration 009).
    """
    hash_value = payload_hash(payload)
    conn.execute(
        """
        INSERT INTO raw_snapshots (run_id, payload_hash, source, request_params, payload, ingested_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (run_id, hash_value, source, Jsonb(request_params), Jsonb(payload), ingested_at),
    )
    return hash_value
