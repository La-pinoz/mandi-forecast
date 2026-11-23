"""Shared test fixtures.

Database tests need TEST_DATABASE_URL (set in CI; locally in .env or the shell) and are skipped
without it. Each test gets its own new, empty schema that is dropped afterwards, so tests never see
or change existing tables, even if the URL points at a real database.

The URL must be a direct connection, not Neon's pooled one (host containing "-pooler"). The pooler
(PgBouncer, transaction mode) may send each transaction to a different server connection and does not
reset session settings, so the fixture's `SET search_path` could apply to the wrong statements and
leak to other clients. This happened once during M02 (see docs/M02-repo-ci-postgres.md).
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg import sql

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
