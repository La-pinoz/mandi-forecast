"""Parsing of variety-wise API records (no database needed)."""

from datetime import date
from decimal import Decimal

import pytest

from mandi.ingest.records import (
    VARIETY_WISE_SOURCE,
    canonical_market_name,
    parse_price,
    parse_record,
    split_duplicates,
)

RECORD = {
    "Arrival_Date": "21/11/2026",
    "Commodity": "Tomato",
    "Commodity_Code": "78",
    "District": "North 24 Parganas",
    "Grade": "FAQ",
    "Market": "Barasat APMC",
    "Max_Price": 2200,
    "Min_Price": "1800",
    "Modal_Price": 2000,
    "State": "West Bengal",
    "Variety": "Other",
}


@pytest.mark.parametrize(
    ("received", "canonical"),
    [
        ("Barasat APMC", "Barasat"),
        ("Barasat", "Barasat"),
        ("Karsiyang(Matigara) APMC", "Karsiyang(Matigara)"),
        # Only a trailing " APMC" is the rename; the letters elsewhere are part of the name.
        ("APMC Yard", "APMC Yard"),
        ("BarasatAPMC", "BarasatAPMC"),
    ],
)
def test_canonical_market_name_strips_only_a_trailing_apmc(received, canonical):
    assert canonical_market_name(received) == canonical


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1800", Decimal("1800")),
        (1800, Decimal("1800")),
        (1850.5, Decimal("1850.5")),
        (" 1800 ", Decimal("1800")),
    ],
)
def test_parse_price_accepts_strings_and_numbers(value, expected):
    assert parse_price(value) == expected


@pytest.mark.parametrize("value", ["", "NR", "1,800", "NaN", "Infinity"])
def test_parse_price_rejects_non_numbers(value):
    with pytest.raises(ValueError):
        parse_price(value)


def test_parse_record_uses_canonical_market_and_keeps_name_as_received():
    observation = parse_record(RECORD, VARIETY_WISE_SOURCE)

    assert observation.market == "Barasat"
    assert observation.market_as_received == "Barasat APMC"
    assert observation.arrival_date == date(2026, 11, 21)
    assert observation.prices() == (Decimal("1800"), Decimal("2200"), Decimal("2000"))
    assert observation.key() == (VARIETY_WISE_SOURCE, "Barasat", "Tomato", "Other", "FAQ", date(2026, 11, 21))


def test_parse_record_rejects_a_bad_date():
    with pytest.raises(ValueError):
        parse_record({**RECORD, "Arrival_Date": "2026-11-21"}, VARIETY_WISE_SOURCE)


def test_identical_duplicates_collapse_to_the_first_row():
    # Rename-overlap day: the same market under both names, same prices.
    old_name = parse_record({**RECORD, "Market": "Barasat"}, VARIETY_WISE_SOURCE)
    new_name = parse_record({**RECORD, "Max_Price": "2200.00"}, VARIETY_WISE_SOURCE)

    unique, conflicting = split_duplicates([old_name, new_name])

    assert unique == [old_name]
    assert conflicting == []


def test_conflicting_duplicates_are_set_aside_entirely():
    first = parse_record(RECORD, VARIETY_WISE_SOURCE)
    second = parse_record({**RECORD, "Min_Price": "1750"}, VARIETY_WISE_SOURCE)
    other_day = parse_record({**RECORD, "Arrival_Date": "20/11/2026"}, VARIETY_WISE_SOURCE)

    unique, conflicting = split_duplicates([first, second, other_day])

    assert unique == [other_day]
    assert conflicting == [first.key()]
