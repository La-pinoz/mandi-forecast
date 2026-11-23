"""Flatten the raw backfill files into one derived CSV (throwaway research code, backfill_plan.md Step 5).

Reads data/raw/ogd_history/<state>/<commodity>/<YYYY-MM-DD>.json (written by backfill_ogd.py) and writes
data/interim/prices_flat.csv, one row per raw record. The CSV is derived: delete it and re-run at any time.

No rows are dropped. A price that is not a number is written as an empty cell and counted in the summary.

Run from the project root:  python research/day1/flatten_history.py
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ogd_history"
OUT_FILE = PROJECT_ROOT / "data" / "interim" / "prices_flat.csv"

COLUMNS = [
    "state", "district", "market", "commodity", "commodity_code", "variety", "grade",
    "arrival_date", "min_price", "max_price", "modal_price", "source_file",
]
PRICE_FIELDS = {"min_price": "Min_Price", "max_price": "Max_Price", "modal_price": "Modal_Price"}
# One price observation per market, product and day. State and district are included because a market
# name alone is not guaranteed to be unique across India.
KEY_COLUMNS = ["state", "district", "market", "commodity", "variety", "grade", "arrival_date"]


def to_float(value: object) -> float | None:
    """Convert a price to float. Older rows send strings ("680"), recent rows numbers (2300)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten(raw_dir: Path, out_file: Path) -> None:
    """Write every record from every raw file to out_file and print data-quality counts."""
    raw_files = sorted(raw_dir.glob("*/*/*.json"))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    price_failures = 0
    price_failure_examples = []
    non_positive_price_rows = 0
    prices_by_key = defaultdict(list)

    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for raw_file in raw_files:
            source_file = raw_file.relative_to(PROJECT_ROOT).as_posix()
            snapshot = json.loads(raw_file.read_text(encoding="utf-8"))
            for record in snapshot["records"]:
                row = {
                    "state": record["State"],
                    "district": record["District"],
                    "market": record["Market"],
                    "commodity": record["Commodity"],
                    "commodity_code": str(record["Commodity_Code"]),
                    "variety": record["Variety"],
                    "grade": record["Grade"],
                    "arrival_date": datetime.strptime(record["Arrival_Date"], "%d/%m/%Y").date().isoformat(),
                    "source_file": source_file,
                }
                for column, raw_field in PRICE_FIELDS.items():
                    price = to_float(record.get(raw_field))
                    if price is None:
                        price_failures += 1
                        if len(price_failure_examples) < 5:
                            price_failure_examples.append((source_file, raw_field, record.get(raw_field)))
                    row[column] = price

                prices = [row[column] for column in PRICE_FIELDS]
                if any(price is not None and price <= 0 for price in prices):
                    non_positive_price_rows += 1
                key = tuple(row[column] for column in KEY_COLUMNS)
                prices_by_key[key].append(tuple(prices))

                writer.writerow(row)
                rows_written += 1

    duplicated_keys = [prices for prices in prices_by_key.values() if len(prices) > 1]
    conflicting_keys = [prices for prices in duplicated_keys if len(set(prices)) > 1]

    print(f"Read {len(raw_files)} raw files, wrote {rows_written} rows to {out_file}")
    print(f"Prices that are not numbers (left empty): {price_failures}")
    for example in price_failure_examples:
        print(f"  e.g. {example}")
    print(f"Rows with a zero or negative price (kept; possible NIL marker): {non_positive_price_rows}")
    print(f"Keys {KEY_COLUMNS} with more than one row: {len(duplicated_keys)}")
    print(f"  of which the rows have different prices: {len(conflicting_keys)}")


if __name__ == "__main__":
    flatten(RAW_DIR, OUT_FILE)
