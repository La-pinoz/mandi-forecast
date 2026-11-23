"""Vintage recording, prices_current refresh and as-of queries (needs TEST_DATABASE_URL)."""

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import psycopg
import pytest

from mandi.ingest.records import VARIETY_WISE_SOURCE, Observation, parse_price
from mandi.store.asof import prices_as_of
from mandi.store.runs import start_run
from mandi.store.vintages import record_vintages, refresh_prices_current, row_hash

IST = timezone(timedelta(hours=5, minutes=30))
DAY_1 = date(2026, 11, 20)
DAY_2 = date(2026, 11, 21)


def observation(
    modal: int | str, arrival_date: date = DAY_1, min_price: int | str | None = None
) -> Observation:
    """A Barasat tomato observation; min and max are 200 either side of the modal unless given."""
    modal_price = parse_price(modal)
    return Observation(
        source=VARIETY_WISE_SOURCE,
        market="Barasat",
        commodity="Tomato",
        variety="Other",
        grade="FAQ",
        arrival_date=arrival_date,
        market_as_received="Barasat APMC",
        district="North 24 Parganas",
        min_price=parse_price(min_price) if min_price is not None else modal_price - 200,
        max_price=modal_price + 200,
        modal_price=modal_price,
    )


def seen_at(day: int, hour: int) -> datetime:
    return datetime(2026, 11, day, hour, 0, tzinfo=IST)


def stored_modal_prices(conn: psycopg.Connection, arrival_date: date = DAY_1) -> list[Decimal]:
    rows = conn.execute(
        "SELECT modal_price FROM price_vintages WHERE arrival_date = %s ORDER BY first_seen_at",
        (arrival_date,),
    ).fetchall()
    return [row[0] for row in rows]


def test_unchanged_changed_and_changed_back(schema_conn):
    runs = [start_run(schema_conn, "test") for _ in range(4)]

    added = [
        record_vintages(schema_conn, [observation(3000)], seen_at(21, 21), runs[0]),
        record_vintages(schema_conn, [observation(3000)], seen_at(22, 21), runs[1]),  # unchanged
        record_vintages(schema_conn, [observation(3200)], seen_at(23, 21), runs[2]),  # changed
        record_vintages(schema_conn, [observation(3000)], seen_at(24, 21), runs[3]),  # changed back
    ]

    assert added == [1, 0, 1, 1]
    assert stored_modal_prices(schema_conn) == [Decimal("3000"), Decimal("3200"), Decimal("3000")]


def test_rerunning_the_same_run_id_adds_no_rows(schema_conn):
    run_id = start_run(schema_conn, "test")
    batch = [observation(3000, DAY_1), observation(3100, DAY_2)]

    first = record_vintages(schema_conn, batch, seen_at(21, 21), run_id)
    second = record_vintages(schema_conn, batch, seen_at(21, 21), run_id)

    assert (first, second) == (2, 0)
    assert schema_conn.execute("SELECT count(*) FROM price_vintages").fetchone()[0] == 2


def test_a_different_value_for_one_key_in_one_run_raises(schema_conn):
    run_id = start_run(schema_conn, "test")
    record_vintages(schema_conn, [observation(3000)], seen_at(21, 21), run_id)

    with pytest.raises(psycopg.errors.UniqueViolation):
        record_vintages(schema_conn, [observation(3200)], seen_at(21, 21), run_id)


def test_string_and_number_prices_are_the_same_version(schema_conn):
    as_string = observation("1800", min_price="1600")
    as_number = observation(1800.0, min_price=1600)
    assert row_hash(as_string) == row_hash(as_number)

    first_run = start_run(schema_conn, "test")
    second_run = start_run(schema_conn, "test")
    record_vintages(schema_conn, [as_string], seen_at(21, 21), first_run)

    assert record_vintages(schema_conn, [as_number], seen_at(22, 21), second_run) == 0


def test_any_price_change_is_a_new_version(schema_conn):
    first_run = start_run(schema_conn, "test")
    second_run = start_run(schema_conn, "test")
    record_vintages(schema_conn, [observation(3000)], seen_at(21, 21), first_run)

    # Same modal, different min: still a revision.
    assert record_vintages(schema_conn, [observation(3000, min_price=2700)], seen_at(22, 21), second_run) == 1


def test_prices_current_follows_the_latest_vintage(schema_conn):
    first_run = start_run(schema_conn, "test")
    second_run = start_run(schema_conn, "test")
    record_vintages(
        schema_conn, [observation(3000, DAY_1), observation(2500, DAY_2)], seen_at(21, 21), first_run
    )

    assert refresh_prices_current(schema_conn) == 2
    assert refresh_prices_current(schema_conn) == 0  # nothing new: no row is touched

    record_vintages(
        schema_conn, [observation(3200, DAY_1), observation(2500, DAY_2)], seen_at(22, 21), second_run
    )
    assert refresh_prices_current(schema_conn) == 1

    rows = schema_conn.execute(
        """
        SELECT c.arrival_date, c.modal_price, v.run_id
        FROM prices_current c JOIN price_vintages v USING (vintage_id)
        ORDER BY c.arrival_date
        """
    ).fetchall()
    assert rows == [(DAY_1, Decimal("3200.00"), second_run), (DAY_2, Decimal("2500.00"), first_run)]


@pytest.fixture
def two_keys_with_a_revision(schema_conn):
    """DAY_1: 3000 seen 21/11 21:00 IST, revised to 3200 seen 23/11 21:00. DAY_2: 2500 seen 22/11 21:00."""
    runs = [start_run(schema_conn, "test") for _ in range(3)]
    record_vintages(schema_conn, [observation(3000, DAY_1)], seen_at(21, 21), runs[0])
    record_vintages(schema_conn, [observation(2500, DAY_2)], seen_at(22, 21), runs[1])
    record_vintages(schema_conn, [observation(3200, DAY_1)], seen_at(23, 21), runs[2])
    return schema_conn


def modal_by_date(rows: list[dict]) -> dict[date, Decimal]:
    return {row["arrival_date"]: row["modal_price"] for row in rows}


@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        (seen_at(21, 20), {}),  # before anything was seen
        (seen_at(21, 21), {DAY_1: Decimal("3000")}),  # exactly at first_seen_at counts as seen
        (seen_at(22, 22), {DAY_1: Decimal("3000"), DAY_2: Decimal("2500")}),  # revision not yet seen
        (seen_at(24, 0), {DAY_1: Decimal("3200"), DAY_2: Decimal("2500")}),  # revision seen
    ],
    ids=["before_first", "at_first", "before_revision", "after_revision"],
)
def test_as_of_returns_the_version_known_at_the_cutoff(two_keys_with_a_revision, cutoff, expected):
    assert modal_by_date(prices_as_of(two_keys_with_a_revision, cutoff)) == expected


def test_as_of_compares_instants_not_clock_readings(two_keys_with_a_revision):
    # 15:30 UTC on 21/11 is 21:00 IST, the moment DAY_1 was first seen.
    cutoff_utc = datetime(2026, 11, 21, 15, 30, tzinfo=UTC)
    assert modal_by_date(prices_as_of(two_keys_with_a_revision, cutoff_utc)) == {DAY_1: Decimal("3000")}


def test_as_of_rejects_a_cutoff_without_time_zone(schema_conn):
    with pytest.raises(ValueError, match="timezone-aware"):
        prices_as_of(schema_conn, datetime(2026, 11, 22, 21, 0))
