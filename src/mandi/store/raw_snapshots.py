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


def save_raw_snapshot(
    conn: psycopg.Connection,
    run_id: int,
    source: str,
    request_params: dict,
    payload: dict,
    ingested_at: datetime,
) -> str:
    """Insert one response (not committed). Returns its payload hash.

    `request_params` must not contain the API key. Saving the same payload twice in one run raises
    UniqueViolation (primary key run_id + payload_hash).
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
