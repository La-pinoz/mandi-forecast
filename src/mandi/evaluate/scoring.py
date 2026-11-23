"""M06: provisional scores for live forecasts (blueprint 6.1, 7.1).

Run from the project root (uses DATABASE_URL from the environment or .env):
    uv run python -m mandi.evaluate.scoring

Flow (one pipeline_runs row, job type 'score'):
1. Find every available forecast whose target date has a reported price, i.e. a row in prices_current for
   its series and date. Only validated prices reach prices_current, so quarantined rows are never actuals;
   an unreported date has no row and is not scored. (A prices_current row for a tracked series is exactly
   what series_day_status calls 'reported'.)
2. Insert a provisional score using that price's vintage, unless a score for this forecast and this vintage
   already exists. So a re-run adds nothing, and a revised actual (a new vintage) adds a second score next
   to the first instead of overwriting it.
3. Count, per model, the forecasts that are unavailable, scored, still waiting for their target date, or
   past their target date with no reported price ("unreported so far": a late row may still arrive).
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from mandi.ingest.snapshot import IST
from mandi.store.db import get_connection
from mandi.store.runs import finish_run, start_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INSERT_PROVISIONAL_SCORES = """
INSERT INTO forecast_scores (forecast_id, status, actual_vintage_id, actual_price, error, score_run_id)
SELECT f.forecast_id, 'provisional', p.vintage_id, p.modal_price, f.point_forecast - p.modal_price,
       %(run_id)s
FROM forecasts f
JOIN series s ON s.series_id = f.series_id
JOIN prices_current p
  ON p.source = s.source AND p.market = s.market AND p.commodity = s.commodity
 AND p.variety = s.variety AND p.grade = s.grade AND p.arrival_date = f.target_date
WHERE f.point_forecast IS NOT NULL
ON CONFLICT (forecast_id, status, actual_vintage_id) DO NOTHING
"""

SCORING_COUNTS = """
SELECT f.model_version,
       count(*) AS forecasts,
       count(*) FILTER (WHERE f.point_forecast IS NULL) AS unavailable,
       count(*) FILTER (WHERE f.point_forecast IS NOT NULL AND scored.forecast_id IS NOT NULL) AS scored,
       count(*) FILTER (WHERE f.point_forecast IS NOT NULL AND scored.forecast_id IS NULL
                          AND f.target_date >= %(today)s) AS awaiting_target_date,
       count(*) FILTER (WHERE f.point_forecast IS NOT NULL AND scored.forecast_id IS NULL
                          AND f.target_date < %(today)s) AS unreported_so_far
FROM forecasts f
LEFT JOIN (SELECT DISTINCT forecast_id FROM forecast_scores WHERE status = 'provisional') scored
       ON scored.forecast_id = f.forecast_id
GROUP BY f.model_version
ORDER BY f.model_version
"""


@dataclass
class ModelCounts:
    model_version: str
    forecasts: int
    unavailable: int  # no value: the model couldn't forecast this target
    scored: int  # has at least one provisional score
    awaiting_target_date: int  # target date is today or later, no actual yet
    unreported_so_far: int  # target date has passed with no reported price


def score_provisional(conn: psycopg.Connection) -> tuple[int, int]:
    """Add provisional scores for every forecast with a reported actual. Returns (run_id, scores added).

    Any error rolls back, marks the run 'failed' and is raised again.
    """
    run_id = start_run(conn, "score")
    try:
        with conn.cursor() as cursor:
            cursor.execute(INSERT_PROVISIONAL_SCORES, {"run_id": run_id})
            added = cursor.rowcount
        conn.commit()
    except Exception as error:
        conn.rollback()
        finish_run(conn, run_id, "failed", error_message=f"{type(error).__name__}: {error}")
        raise
    finish_run(conn, run_id, "succeeded", rows_out=added)
    return run_id, added


def scoring_counts(conn: psycopg.Connection, today: date) -> list[ModelCounts]:
    """Per model: how many forecasts are unavailable, scored, waiting, or unreported so far.

    `today` is the IST date; a target date before it with no reported price counts as unreported so far.
    """
    rows = conn.execute(SCORING_COUNTS, {"today": today}).fetchall()
    return [ModelCounts(*row) for row in rows]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    today = datetime.now(IST).date()
    with get_connection() as conn:
        run_id, added = score_provisional(conn)
        counts = scoring_counts(conn, today)

    print(f"Run {run_id} succeeded. Provisional scores added: {added}")
    print(f"\nForecasts by model (today is {today} IST):")
    header = (
        f"{'model':<20}{'forecasts':>10}{'unavailable':>13}{'scored':>8}{'awaiting':>10}{'unreported':>12}"
    )
    print(header)
    for model in counts:
        print(
            f"{model.model_version:<20}{model.forecasts:>10}{model.unavailable:>13}{model.scored:>8}"
            f"{model.awaiting_target_date:>10}{model.unreported_so_far:>12}"
        )


if __name__ == "__main__":
    main()
