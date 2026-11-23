"""Hard data contracts (validate/contracts.py): one test per quarantine rule, no database needed."""

from datetime import date
from decimal import Decimal

import pytest

from mandi.ingest.records import VARIETY_WISE_SOURCE
from mandi.store.raw_snapshots import payload_hash
from mandi.validate.contracts import (
    BAD_TYPE,
    CONFLICTING_DUPLICATE,
    FUTURE_ARRIVAL_DATE,
    MIN_ABOVE_MODAL,
    MODAL_ABOVE_MAX,
    NON_POSITIVE_PRICE,
    check_records,
)

TRACKED = {(VARIETY_WISE_SOURCE, "Barasat", "Tomato", "Other", "FAQ")}
SNAPSHOT_DATE = date(2026, 11, 22)

VALID = {
    "Arrival_Date": "21/11/2026",
    "Commodity": "Tomato",
    "District": "North 24 Parganas",
    "Grade": "FAQ",
    "Market": "Barasat APMC",
    "Max_Price": 2200,
    "Min_Price": "1800",
    "Modal_Price": 2000,
    "State": "West Bengal",
    "Variety": "Other",
}


def check(records: list[dict]):
    return check_records(records, VARIETY_WISE_SOURCE, TRACKED, SNAPSHOT_DATE)


def reasons(result) -> list[tuple[str, str]]:
    return sorted((row.reason_code, row.detail) for row in result.quarantined)


def test_a_valid_record_becomes_an_observation():
    result = check([VALID])

    assert result.quarantined == []
    assert result.tracked_records == 1
    [observation] = result.observations
    assert observation.market == "Barasat"
    assert (observation.min_price, observation.max_price, observation.modal_price) == (
        Decimal("1800"),
        Decimal("2200"),
        Decimal("2000"),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("Modal_Price", "NR"), ("Min_Price", "1,800"), ("Max_Price", None), ("Modal_Price", "")],
)
def test_unparseable_price_is_bad_type(field, value):
    result = check([{**VALID, field: value}])

    assert result.observations == []
    assert reasons(result) == [(BAD_TYPE, f"{field}={value!r}")]


def test_unparseable_date_is_bad_type():
    result = check([{**VALID, "Arrival_Date": "2026-11-21"}])

    assert reasons(result) == [(BAD_TYPE, "Arrival_Date='2026-11-21'")]


def test_missing_price_field_is_bad_type():
    record = dict(VALID)
    del record["Max_Price"]

    assert reasons(check([record])) == [(BAD_TYPE, "Max_Price=None")]


@pytest.mark.parametrize("price", ["0", -5])
def test_zero_or_negative_price_is_quarantined(price):
    # Min price only, so no ordering rule is broken as well.
    result = check([{**VALID, "Min_Price": price}])

    assert reasons(result) == [(NON_POSITIVE_PRICE, f"Min_Price={price!r}")]


def test_zero_max_with_positive_modal_breaks_two_rules():
    # The M01 case: Max_Price = 0 with a positive modal price. One quarantine row per rule.
    result = check([{**VALID, "Max_Price": 0}])

    assert reasons(result) == [
        (MODAL_ABOVE_MAX, "min/max/modal = 1800/0/2000"),
        (NON_POSITIVE_PRICE, "Max_Price=0"),
    ]


def test_min_above_modal_is_quarantined():
    result = check([{**VALID, "Min_Price": 2100}])

    assert reasons(result) == [(MIN_ABOVE_MODAL, "min/max/modal = 2100/2200/2000")]


def test_modal_above_max_is_quarantined():
    result = check([{**VALID, "Modal_Price": 2300}])

    assert reasons(result) == [(MODAL_ABOVE_MAX, "min/max/modal = 1800/2200/2300")]


def test_equal_prices_are_valid():
    result = check([{**VALID, "Min_Price": 2000, "Max_Price": 2000}])

    assert result.quarantined == []
    assert len(result.observations) == 1


def test_arrival_date_after_the_snapshot_date_is_quarantined():
    result = check([{**VALID, "Arrival_Date": "23/11/2026"}])

    assert reasons(result) == [
        (FUTURE_ARRIVAL_DATE, "Arrival_Date=23/11/2026 is after the snapshot date 2026-11-22")
    ]


def test_arrival_date_on_the_snapshot_date_is_allowed():
    assert check([{**VALID, "Arrival_Date": "22/11/2026"}]).quarantined == []


def test_conflicting_duplicates_are_both_quarantined():
    first = VALID
    second = {**VALID, "Market": "Barasat", "Min_Price": 1750}
    other_day = {**VALID, "Arrival_Date": "20/11/2026"}

    result = check([first, second, other_day])

    assert [observation.arrival_date for observation in result.observations] == [date(2026, 11, 20)]
    assert sorted((row.row_hash, row.reason_code) for row in result.quarantined) == sorted(
        [(payload_hash(first), CONFLICTING_DUPLICATE), (payload_hash(second), CONFLICTING_DUPLICATE)]
    )


def test_identical_duplicates_collapse_to_the_first_row():
    # Rename-overlap day: the same market under both names, same prices (string vs number).
    old_name = {**VALID, "Market": "Barasat"}
    new_name = {**VALID, "Max_Price": "2200.00"}

    result = check([old_name, new_name])

    assert result.quarantined == []
    assert result.duplicate_rows_collapsed == 1
    [observation] = result.observations
    assert observation.market_as_received == "Barasat"


def test_untracked_records_are_not_checked():
    other_grade = {**VALID, "Grade": "Non-FAQ", "Modal_Price": "NR"}
    other_market = {**VALID, "Market": "Kalipur", "Min_Price": 0}

    result = check([other_grade, other_market])

    assert result == check([])
    assert result.tracked_records == 0


def test_a_bad_record_does_not_affect_a_good_one_in_the_same_snapshot():
    bad = {**VALID, "Arrival_Date": "20/11/2026", "Modal_Price": "NR"}

    result = check([VALID, bad])

    assert [observation.arrival_date for observation in result.observations] == [date(2026, 11, 21)]
    assert [row.record for row in result.quarantined] == [bad]
