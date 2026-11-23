"""Shared test fixtures.

Database tests need TEST_DATABASE_URL (set in CI; locally in .env or the shell) and are skipped
without it. Each test gets its own new, empty schema that is dropped afterwards, so tests never see
or change existing tables, even if the URL points at a real database.

The URL must be a direct connection, not Neon's pooled one (host containing "-pooler"). The pooler
(PgBouncer, transaction mode) may send each transaction to a different server connection and does not
reset session settings, so the fixture's `SET search_path` could apply to the wrong statements and
leak to other clients. This happened once during M02 (see docs/M02-repo-ci-postgres.md).
"""

import math
import os
import uuid
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import psycopg
import pytest
from dotenv import load_dotenv
from psycopg import sql

from mandi.features.build import build_feature_table
from mandi.models.lgbm import train
from mandi.store.migrate import apply_migrations

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@pytest.fixture
def db_conn() -> Iterator[psycopg.Connection]:
    """A connection whose search_path is a fresh, empty schema (a stand-in for a fresh database)."""
    database_url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set")
    if "-pooler" in database_url:
        pytest.fail("TEST_DATABASE_URL is a pooled Neon URL; use the direct one (host without '-pooler').")

    schema = sql.Identifier(f"test_{uuid.uuid4().hex[:12]}")
    conn = psycopg.connect(database_url)
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
    conn.execute(sql.SQL("SET search_path TO {}").format(schema))
    conn.commit()
    try:
        yield conn
    finally:
        conn.rollback()
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
        conn.commit()
        conn.close()


@pytest.fixture
def schema_conn(db_conn: psycopg.Connection) -> psycopg.Connection:
    """`db_conn` with all project migrations applied: a fresh database holding the current schema."""
    apply_migrations(db_conn)
    return db_conn


@pytest.fixture(scope="session")
def small_lgbm_models() -> dict[int, lgb.Booster]:
    """Per-horizon LightGBM models trained on 120 days of one synthetic series (series 1, Barasat Tomato)
    whose price grows 1% a day. Only the plumbing around the model is tested with it, not its accuracy."""
    series_info = pd.DataFrame({"series_id": [1], "market": ["Barasat"], "crop": ["Tomato"]})
    prices = {}
    spreads = {}
    for offset in range(120):
        day = date(2024, 1, 1) + timedelta(days=offset)
        prices[day] = 2000.0 * math.exp(0.01 * offset)
        spreads[day] = 0.1
    origins = [date(2024, 1, 2) + timedelta(days=offset) for offset in range(110)]
    table = build_feature_table({1: prices}, {1: spreads}, series_info, origins)
    return train(table[table["actual"].notna()], per_horizon=True)
