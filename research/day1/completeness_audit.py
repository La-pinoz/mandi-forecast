"""Completeness audit per price series (throwaway research code, backfill_plan.md Step 6).

Reads data/interim/prices_flat.csv (from flatten_history.py) and writes docs/completeness_audit.csv with one
row per series (state, district, market, commodity, variety, grade):

- market: the market name with a trailing " APMC" removed. Around 01-05/11/2025 the source renamed West Bengal
  markets (e.g. "Asansol" -> "Asansol APMC"); on the overlap days both names carry identical prices, so they are
  one market. Without this, every series looks like two half-length series. See docs/M01-day1-experiment.md.

- first_date, last_date: over all fetched data
- days_reported, pct_days_reported: distinct reported dates in the audit window / calendar days in the window
- longest_gap_days: longest run of consecutive calendar days with no row in the window (window edges count)
- pct_week_ago_available: % of reported dates in the window whose date 7 calendar days earlier is also
  reported. Seasonal-naive needs that exact observation.
- conflicting_days: dates (any time) with more than one row for the series and different modal prices.

The window is the 3 years ending on the latest arrival date in the file. "Reported" means a row exists; a
missing row is "unreported", not proof that the market was closed.

Run from the project root:  python research/day1/completeness_audit.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLAT_FILE = PROJECT_ROOT / "data" / "interim" / "prices_flat.csv"
OUT_FILE = PROJECT_ROOT / "docs" / "completeness_audit.csv"
SERIES_COLUMNS = ["state", "district", "market", "commodity", "variety", "grade"]
WINDOW_YEARS = 3


def audit_series(rows: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> dict:
    """Completeness numbers for one series, given its rows (columns arrival_date and modal_price)."""
    reported = set(rows["arrival_date"])
    window_dates = sorted(d for d in reported if window_start <= d <= window_end)
    window_days = (window_end - window_start).days + 1

    # Put a virtual reported day just outside each edge so gaps at the start and end of the window count.
    boundaries = [window_start - pd.Timedelta(days=1)] + window_dates + [window_end + pd.Timedelta(days=1)]
    longest_gap = 0
    for previous, current in zip(boundaries, boundaries[1:]):
        longest_gap = max(longest_gap, (current - previous).days - 1)

    week_ago_count = sum(1 for d in window_dates if d - pd.Timedelta(days=7) in reported)
    pct_week_ago = 100 * week_ago_count / len(window_dates) if window_dates else 0.0

    modal_prices_per_date = rows.groupby("arrival_date")["modal_price"].nunique()
    conflicting_days = int((modal_prices_per_date > 1).sum())

    return {
        "first_date": min(reported).date(),
        "last_date": max(reported).date(),
        "days_reported": len(window_dates),
        "pct_days_reported": round(100 * len(window_dates) / window_days, 1),
        "longest_gap_days": longest_gap,
        "pct_week_ago_available": round(pct_week_ago, 1),
        "conflicting_days": conflicting_days,
    }


def main() -> None:
    prices = pd.read_csv(
        FLAT_FILE, usecols=SERIES_COLUMNS + ["arrival_date", "modal_price"], parse_dates=["arrival_date"]
    )
    prices["market"] = prices["market"].str.removesuffix(" APMC")
    window_end = prices["arrival_date"].max()
    window_start = window_end - pd.DateOffset(years=WINDOW_YEARS) + pd.Timedelta(days=1)
    print(f"Audit window: {window_start.date()} to {window_end.date()}")

    rows = []
    for series_key, group in prices.groupby(SERIES_COLUMNS):
        row = dict(zip(SERIES_COLUMNS, series_key))
        row.update(audit_series(group, window_start, window_end))
        rows.append(row)

    audit = pd.DataFrame(rows).sort_values(["commodity", "pct_days_reported"], ascending=[True, False])
    audit.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(audit)} series to {OUT_FILE}")

    show_columns = ["state", "market", "variety", "grade", "pct_days_reported", "longest_gap_days",
                    "pct_week_ago_available", "conflicting_days", "last_date"]
    for commodity, group in audit.groupby("commodity"):
        print(f"\n{commodity}: 10 most complete of {len(group)} series")
        print(group[show_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
