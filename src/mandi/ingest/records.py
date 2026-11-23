"""Turn records from the data.gov.in variety-wise resource into observations for `price_vintages`.

One API record looks like (prices are strings in older rows, numbers in newer ones):
    {"State": "West Bengal", "District": "North 24 Parganas", "Market": "Barasat APMC",
     "Commodity": "Tomato", "Variety": "Other", "Grade": "FAQ", "Arrival_Date": "21/11/2026",
     "Min_Price": 1800, "Max_Price": "2200", "Modal_Price": 2000, ...}
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

# data.gov.in "Variety-wise Daily Market Prices Data of Commodity", the project's only source (M01).
VARIETY_WISE_SOURCE = "35985678-0d79-46b4-9ed6-6f13308a1d24"

APMC_SUFFIX = " APMC"


@dataclass(frozen=True)
class Observation:
    """One parsed price row. Observation key: (source, market, commodity, variety, grade, arrival_date)."""

    source: str
    market: str
    commodity: str
    variety: str
    grade: str
    arrival_date: date
    market_as_received: str
    district: str
    min_price: Decimal
    max_price: Decimal
    modal_price: Decimal

    def key(self) -> tuple[str, str, str, str, str, date]:
        return (self.source, self.market, self.commodity, self.variety, self.grade, self.arrival_date)


def canonical_market_name(market_as_received: str) -> str:
    """Remove a trailing " APMC": West Bengal markets were renamed `X` -> `X APMC` on 01-05/11/2025, with
    identical prices on the overlap days, so both names are one market (owner decision, M01)."""
    if market_as_received.endswith(APMC_SUFFIX):
        return market_as_received[: -len(APMC_SUFFIX)]
    return market_as_received


def parse_price(value: str | int | float) -> Decimal:
    """Parse a price sent as a string ("1800") or a number (1800). Raises ValueError if it isn't one."""
    try:
        price = Decimal(str(value).strip())
    except InvalidOperation:
        raise ValueError(f"price is not a number: {value!r}") from None
    if not price.is_finite():
        raise ValueError(f"price is not a finite number: {value!r}")
    return price


def parse_record(record: dict, source: str) -> Observation:
    """Parse one API record. Raises ValueError for a bad price or date, KeyError for a missing field."""
    return Observation(
        source=source,
        market=canonical_market_name(record["Market"]),
        commodity=record["Commodity"],
        variety=record["Variety"],
        grade=record["Grade"],
        arrival_date=datetime.strptime(record["Arrival_Date"], "%d/%m/%Y").date(),
        market_as_received=record["Market"],
        district=record["District"],
        min_price=parse_price(record["Min_Price"]),
        max_price=parse_price(record["Max_Price"]),
        modal_price=parse_price(record["Modal_Price"]),
    )
