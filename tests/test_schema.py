"""Checks that the core tables enforce the rules the blueprint relies on (needs TEST_DATABASE_URL)."""

import psycopg
import pytest

from mandi.store.migrate import apply_migrations

SOURCE = "35985678-0d79-46b4-9ed6-6f13308a1d24"

INSERT_VINTAGE = """
INSERT INTO price_vintages (
    source, market, commodity, variety, grade, arrival_date, market_as_received, district,
    min_price, max_price, modal_price, row_hash, first_seen_at, run_id
) VALUES (
    %(source)s, 'Barasat', 'Tomato', 'Other', 'FAQ', '2026-11-21', 'Barasat APMC', 'North 24 Parganas',
    %(min)s, %(max)s, %(modal)s, 'hash-1', '2026-11-22 21:00+05:30', %(run_id)s
)
"""


@pytest.fixture
def schema_conn(db_conn):
    apply_migrations(db_conn)
    return db_conn


def new_run(conn: psycopg.Connection) -> int:
    row = conn.execute(
        "INSERT INTO pipeline_runs (job_type, status) VALUES ('test', 'running') RETURNING run_id"
    ).fetchone()
    return row[0]


def test_same_observation_twice_in_one_run_is_rejected(schema_conn):
    run_id = new_run(schema_conn)
    values = {"source": SOURCE, "min": 1800, "max": 2200, "modal": 2000, "run_id": run_id}
    schema_conn.execute(INSERT_VINTAGE, values)

    with pytest.raises(psycopg.errors.UniqueViolation):
        schema_conn.execute(INSERT_VINTAGE, values)


def test_same_observation_in_a_later_run_is_allowed(schema_conn):
    first_run = new_run(schema_conn)
    second_run = new_run(schema_conn)
    schema_conn.execute(
        INSERT_VINTAGE, {"source": SOURCE, "min": 1800, "max": 2200, "modal": 2000, "run_id": first_run}
    )
    schema_conn.execute(
        INSERT_VINTAGE, {"source": SOURCE, "min": 1800, "max": 2400, "modal": 2100, "run_id": second_run}
    )

    count = schema_conn.execute("SELECT count(*) FROM price_vintages").fetchone()[0]
    assert count == 2


@pytest.mark.parametrize(
    ("min_price", "max_price", "modal_price"),
    [(0, 2200, 2000), (2100, 2200, 2000), (1800, 1900, 2000)],
    ids=["zero_min", "min_above_modal", "modal_above_max"],
)
def test_impossible_prices_are_rejected(schema_conn, min_price, max_price, modal_price):
    run_id = new_run(schema_conn)
    values = {"source": SOURCE, "min": min_price, "max": max_price, "modal": modal_price, "run_id": run_id}

    with pytest.raises(psycopg.errors.CheckViolation):
        schema_conn.execute(INSERT_VINTAGE, values)


@pytest.mark.parametrize(
    "statement", ["UPDATE price_vintages SET modal_price = 2100", "DELETE FROM price_vintages"]
)
def test_price_vintages_are_append_only(schema_conn, statement):
    run_id = new_run(schema_conn)
    schema_conn.execute(
        INSERT_VINTAGE, {"source": SOURCE, "min": 1800, "max": 2200, "modal": 2000, "run_id": run_id}
    )

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        schema_conn.execute(statement)


@pytest.mark.parametrize(
    "statement", ["UPDATE raw_snapshots SET payload = '{}'", "DELETE FROM raw_snapshots"]
)
def test_raw_snapshots_are_append_only(schema_conn, statement):
    run_id = new_run(schema_conn)
    schema_conn.execute(
        "INSERT INTO raw_snapshots (run_id, payload_hash, source, request_params, payload, ingested_at)"
        " VALUES (%s, 'h1', %s, '{}', '{\"total\": 0, \"records\": []}', now())",
        (run_id, SOURCE),
    )

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        schema_conn.execute(statement)


def test_unknown_day_status_is_rejected(schema_conn):
    series_id = schema_conn.execute(
        "INSERT INTO series (source, market, commodity, variety, grade, district)"
        " VALUES (%s, 'Barasat', 'Tomato', 'Other', 'FAQ', 'North 24 Parganas') RETURNING series_id",
        (SOURCE,),
    ).fetchone()[0]

    # "closed" is deliberately not a status: missing data must never be read as a market closure.
    with pytest.raises(psycopg.errors.CheckViolation):
        schema_conn.execute(
            "INSERT INTO series_day_status (series_id, arrival_date, status)"
            " VALUES (%s, '2026-11-21', 'closed')",
            (series_id,),
        )
