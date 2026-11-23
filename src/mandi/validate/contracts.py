"""Hard data contracts for records of the variety-wise resource (blueprint 5.4), as a pandera schema.

Input: the records of one raw snapshot (one API response). Only records of tracked series are checked; the
others stay in `raw_snapshots` only. Output: the observations that passed, and one `QuarantinedRow` per
(record, failed rule). A record that breaks several rules gets one row per rule.

NIL / no-transaction records: M01 found that the source has no NIL marker; a no-trade day simply has no row.
So there is no NIL record to route away before the price rules run. A day without a row becomes `unreported`
in `series_day_status` (validate/day_status.py) and can never reach quarantine. If the source ever adds a
marker, recognising it belongs at the top of `check_records`, before the schema runs.

Reason codes:
    bad_type               a price or the arrival date can't be parsed (e.g. "NR", "1,800", missing)
    non_positive_price     min, max or modal price is 0 or negative
    min_above_modal        min price > modal price
    modal_above_max        modal price > max price
    future_arrival_date    arrival date after the snapshot's date (IST)
    conflicting_duplicate  another row in the same snapshot has the same observation key and other prices
"""

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandera.pandas as pa

from mandi.ingest.records import Observation, canonical_market_name, parse_price, parse_record
from mandi.store.raw_snapshots import payload_hash

BAD_TYPE = "bad_type"
NON_POSITIVE_PRICE = "non_positive_price"
MIN_ABOVE_MODAL = "min_above_modal"
MODAL_ABOVE_MAX = "modal_above_max"
FUTURE_ARRIVAL_DATE = "future_arrival_date"
CONFLICTING_DUPLICATE = "conflicting_duplicate"

# Columns built from the record -> the source field they come from (used in quarantine details).
SOURCE_FIELDS = {
    "arrival_date": "Arrival_Date",
    "min_price": "Min_Price",
    "max_price": "Max_Price",
    "modal_price": "Modal_Price",
}
PRICE_COLUMNS = ["min_price", "max_price", "modal_price"]
KEY_COLUMNS = ["market", "commodity", "variety", "grade", "arrival_date"]


@dataclass(frozen=True)
class QuarantinedRow:
    row_hash: str  # SHA-256 of the record as received
    record: dict
    reason_code: str
    detail: str


@dataclass
class ContractResult:
    observations: list[Observation]  # passed every rule; one per observation key
    quarantined: list[QuarantinedRow]
    tracked_records: int
    duplicate_rows_collapsed: int  # rows dropped because an identical row of the same key came first


def price_or_nan(value: object) -> float:
    """`parse_price` as a float, or NaN when the value isn't a number (NaN then fails as bad_type)."""
    try:
        return float(parse_price(value))
    except ValueError:
        return math.nan


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """One row per record with the parsed values the contracts check. Unparseable values become NaN / NaT.

    Prices are floats here only for the checks; stored observations keep exact Decimals (`parse_record`).
    """
    frame = pd.DataFrame(
        {
            "market": [canonical_market_name(record["Market"]) for record in records],
            "commodity": [record["Commodity"] for record in records],
            "variety": [record["Variety"] for record in records],
            "grade": [record["Grade"] for record in records],
            "arrival_date": pd.to_datetime(
                [record.get("Arrival_Date") for record in records], format="%d/%m/%Y", errors="coerce"
            ),
        }
    )
    for column in PRICE_COLUMNS:
        frame[column] = [price_or_nan(record.get(SOURCE_FIELDS[column])) for record in records]
    return frame


def conflicting_duplicate_mask(frame: pd.DataFrame) -> pd.Series:
    """True for rows whose observation key has another row with different prices.

    Rows whose date couldn't be parsed have no key and are never marked (they are quarantined as bad_type).
    """
    # map(str) writes NaN as "nan" (astype(str) would keep it missing in pandas 3).
    price_text = (
        frame["min_price"].map(str) + "|" + frame["max_price"].map(str) + "|" + frame["modal_price"].map(str)
    )
    key_groups = [frame[column] for column in KEY_COLUMNS]
    distinct_prices_per_key = price_text.groupby(key_groups).transform("nunique")
    return distinct_prices_per_key > 1


def contract_schema(snapshot_date: date) -> pa.DataFrameSchema:
    """The hard contracts. Each check is named after the reason code it produces when it fails.

    `nullable=False` is the type rule: a value that couldn't be parsed is NaN/NaT. Each row-level check is
    written as "not <violation>", so a NaN comparison counts as no violation (NaN is caught as bad_type).
    """
    last_allowed_date = pd.Timestamp(snapshot_date)
    price_column = pa.Column(
        float, pa.Check(lambda prices: prices > 0, name=NON_POSITIVE_PRICE), nullable=False
    )
    return pa.DataFrameSchema(
        columns={
            "arrival_date": pa.Column(nullable=False),
            "min_price": price_column,
            "max_price": price_column,
            "modal_price": price_column,
        },
        checks=[
            pa.Check(
                lambda df: ~(df["min_price"] > df["modal_price"]), name=MIN_ABOVE_MODAL, ignore_na=False
            ),
            pa.Check(
                lambda df: ~(df["modal_price"] > df["max_price"]), name=MODAL_ABOVE_MAX, ignore_na=False
            ),
            pa.Check(
                lambda df: ~(df["arrival_date"] > last_allowed_date),
                name=FUTURE_ARRIVAL_DATE,
                ignore_na=False,
            ),
            pa.Check(lambda df: ~conflicting_duplicate_mask(df), name=CONFLICTING_DUPLICATE, ignore_na=False),
        ],
    )


def failure_detail(reason_code: str, column: str, record: dict, snapshot_date: date) -> str:
    """A short human-readable explanation of one failure, using the values as received."""
    if reason_code in (BAD_TYPE, NON_POSITIVE_PRICE):
        field = SOURCE_FIELDS[column]
        return f"{field}={record.get(field)!r}"
    if reason_code == FUTURE_ARRIVAL_DATE:
        return f"Arrival_Date={record['Arrival_Date']} is after the snapshot date {snapshot_date.isoformat()}"
    return f"min/max/modal = {record['Min_Price']}/{record['Max_Price']}/{record['Modal_Price']}"


def find_failures(records: list[dict], snapshot_date: date) -> dict[int, dict[str, str]]:
    """Run the schema. Returns {record position: {reason_code: detail}} for records that failed a rule."""
    frame = records_to_frame(records)
    try:
        contract_schema(snapshot_date).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as errors:
        failure_cases = errors.failure_cases
    else:
        return {}

    known_codes = {
        NON_POSITIVE_PRICE,
        MIN_ABOVE_MODAL,
        MODAL_ABOVE_MAX,
        FUTURE_ARRIVAL_DATE,
        CONFLICTING_DUPLICATE,
    }
    details: dict[int, dict[str, list[str]]] = {}
    for case in failure_cases.itertuples():
        reason_code = BAD_TYPE if case.check == "not_nullable" else case.check
        # An unknown check (e.g. a dtype failure) or a failure without a row (a check that raised an error)
        # is a bug in this module, not a data problem, so fail loudly instead of quarantining anything.
        if (reason_code != BAD_TYPE and reason_code not in known_codes) or case.index is None:
            raise RuntimeError(f"Unexpected contract failure: {case}")
        position = int(case.index)
        detail = failure_detail(reason_code, case.column, records[position], snapshot_date)
        # Row-level checks report one failure per column with the same detail, so keep each detail once.
        # Column checks (bad_type, non_positive_price) can fail on several columns; all are listed.
        reason_details = details.setdefault(position, {}).setdefault(reason_code, [])
        if detail not in reason_details:
            reason_details.append(detail)

    failures = {}
    for position, reasons in details.items():
        failures[position] = {reason_code: "; ".join(parts) for reason_code, parts in reasons.items()}
    return failures


def check_records(
    records: list[dict],
    source: str,
    tracked: set[tuple[str, str, str, str, str]],
    snapshot_date: date,
) -> ContractResult:
    """Apply the contracts to the records of tracked series in one snapshot.

    `tracked` holds (source, market, commodity, variety, grade) of the tracked series; `snapshot_date` is the
    IST date the snapshot was received, for the future-date rule. Rows of one key with identical prices are
    one observation (the first row is kept, e.g. `X` and `X APMC` on the rename-overlap days).
    """
    tracked_records = []
    for record in records:
        series_key = (
            source,
            canonical_market_name(record["Market"]),
            record["Commodity"],
            record["Variety"],
            record["Grade"],
        )
        if series_key in tracked:
            tracked_records.append(record)
    if not tracked_records:
        return ContractResult([], [], 0, 0)

    failures = find_failures(tracked_records, snapshot_date)

    quarantined = []
    observations = []
    seen_keys = set()
    duplicate_rows_collapsed = 0
    for position, record in enumerate(tracked_records):
        if position in failures:
            for reason_code, detail in failures[position].items():
                quarantined.append(QuarantinedRow(payload_hash(record), record, reason_code, detail))
            continue
        observation = parse_record(record, source)
        # Conflicting rows were quarantined above, so a key seen again here has identical prices.
        if observation.key() in seen_keys:
            duplicate_rows_collapsed += 1
            continue
        seen_keys.add(observation.key())
        observations.append(observation)
    return ContractResult(observations, quarantined, len(tracked_records), duplicate_rows_collapsed)
