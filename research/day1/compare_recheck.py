"""Compare two fetches of the same dates to see whether published data changes (backfill_plan.md Step 7).

Each folder has the layout written by backfill_ogd.py: <folder>/<State>/<Commodity>/<YYYY-MM-DD>.json.
For every date present in both folders it reports, per commodity:
- rows in the earlier and the later fetch;
- added: rows whose key is only in the later fetch (late-arriving markets);
- removed: rows whose key is only in the earlier fetch;
- changed: keys in both fetches whose prices differ (true revisions).
Key = (district, market, variety, grade). A key with conflicting duplicate rows (see docs/M01-day1-experiment.md) is
compared as a sorted list of its price triples, so a duplicate appearing or vanishing counts as "changed".

Run from the project root, earlier folder first:
    python research/day1/compare_recheck.py data/raw/ogd_history data/raw/ogd_recheck/2026-11-23T1041
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_prices_by_key(json_file: Path) -> tuple[dict, str]:
    """Return ({key: sorted list of (min, max, modal)}, fetched_at_ist) for one saved date file."""
    snapshot = json.loads(json_file.read_text(encoding="utf-8"))
    prices_by_key = defaultdict(list)
    for record in snapshot["records"]:
        key = (record["District"], record["Market"], record["Variety"], record["Grade"])
        # Prices may be strings or numbers depending on the year, so compare them as floats.
        prices = (float(record["Min_Price"]), float(record["Max_Price"]), float(record["Modal_Price"]))
        prices_by_key[key].append(prices)
    for key in prices_by_key:
        prices_by_key[key].sort()
    return dict(prices_by_key), snapshot["fetched_at_ist"]


def compare(earlier_root: Path, later_root: Path) -> None:
    later_files = sorted(later_root.glob("*/*/*.json"))
    if not later_files:
        sys.exit(f"No date files found under {later_root}")

    print("commodity  date        fetched (earlier -> later)                     rows  added removed changed")
    for later_file in later_files:
        relative_path = later_file.relative_to(later_root)
        earlier_file = earlier_root / relative_path
        if not earlier_file.exists():
            continue
        earlier, earlier_time = load_prices_by_key(earlier_file)
        later, later_time = load_prices_by_key(later_file)

        added = [key for key in later if key not in earlier]
        removed = [key for key in earlier if key not in later]
        changed = [key for key in later if key in earlier and later[key] != earlier[key]]
        earlier_rows = sum(len(prices) for prices in earlier.values())
        later_rows = sum(len(prices) for prices in later.values())

        commodity = relative_path.parts[1]
        print(
            f"{commodity:<10} {later_file.stem}  {earlier_time} -> {later_time}  "
            f"{earlier_rows:>3}->{later_rows:<3} {len(added):>5} {len(removed):>7} {len(changed):>7}"
        )
        for key in changed:
            print(f"    changed {key}: {earlier[key]} -> {later[key]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python research/day1/compare_recheck.py <earlier folder> <later folder>")
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
