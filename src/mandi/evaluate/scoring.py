"""M06 + M09: provisional and settled scores for live forecasts (blueprint 6.1, 7.1).

Run from the project root (uses DATABASE_URL from the environment or .env):
    uv run python -m mandi.evaluate.scoring                    # settling window 7 days
    uv run python -m mandi.evaluate.scoring --settling-days 10

Flow (one pipeline_runs row, job type 'score', one transaction):
1. Provisional: every available forecast whose target date has a reported price (a row in prices_current;
   only validated prices get there, so quarantined rows are never actuals) gets a score against that
   price's vintage, unless it already has a score for that vintage. A revised actual (a new vintage) adds a
   second provisional score next to the first instead of overwriting it. Settled forecasts are skipped.
2. Settled: a forecast whose target date is more than `settling_days` days before today (IST) and that has
   a reported price gets its one settled score, against the vintage in prices_current now. With 7 days, a
   target date D settles on D + 8: that morning's recheck (dates today-2 .. today-8) has looked at D for
   the last time, just before this step.
3. Revisions: if a settled score's actual has since been revised (prices_current holds another vintage), a
   row is added to settled_score_revisions. The settled score itself is never changed.
4. Count, per model, the forecasts that are unavailable, scored, settled, still waiting for their target
   date, or past their target date with no reported price ("unreported so far").
"""

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from mandi.ingest.snapshot import IST
from mandi.store.db import get_connection
from mandi.store.runs import finish_run, start_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTLING_DAYS = 7  # the source's edit window (blueprint 6.1); M14 checks it against observed revisions

# Joins a forecast to the latest validated price of its series on its target date.
FORECAST_ACTUALS = """
FROM forecasts f
JOIN series s ON s.series_id = f.series_id
JOIN prices_current p
  ON p.source = s.source AND p.market = s.market AND p.commodity = s.commodity
 AND p.variety = s.variety AND p.grade = s.grade AND p.arrival_date = f.target_date
"""

INSERT_PROVISIONAL_SCORES = f"""
INSERT INTO forecast_scores (forecast_id, status, actual_vintage_id, actual_price, error, score_run_id)
SELECT f.forecast_id, 'provisional', p.vintage_id, p.modal_price, f.point_forecast - p.modal_price,
       %(run_id)s
{FORECAST_ACTUALS}
WHERE f.point_forecast IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM forecast_scores settled
      WHERE settled.forecast_id = f.forecast_id AND settled.status = 'settled'
  )
ON CONFLICT (forecast_id, status, actual_vintage_id) DO NOTHING
"""

INSERT_SETTLED_SCORES = f"""
INSERT INTO forecast_scores (
    forecast_id, status, actual_vintage_id, actual_price, error, settling_days, score_run_id
)
SELECT f.forecast_id, 'settled', p.vintage_id, p.modal_price, f.point_forecast - p.modal_price,
       %(settling_days)s, %(run_id)s
{FORECAST_ACTUALS}
WHERE f.point_forecast IS NOT NULL
  AND f.target_date < %(settle_before)s
ON CONFLICT (forecast_id) WHERE status = 'settled' DO NOTHING
"""

INSERT_SETTLED_REVISIONS = f"""
INSERT INTO settled_score_revisions (score_id, revised_vintage_id, revised_price, revised_error, run_id)
SELECT sc.score_id, p.vintage_id, p.modal_price, f.point_forecast - p.modal_price, %(run_id)s
{FORECAST_ACTUALS}
JOIN forecast_scores sc ON sc.forecast_id = f.forecast_id AND sc.status = 'settled'
WHERE p.vintage_id <> sc.actual_vintage_id
ON CONFLICT (score_id, revised_vintage_id) DO NOTHING
"""

SCORING_COUNTS = """
SELECT f.model_version,
       count(*) AS forecasts,
       count(*) FILTER (WHERE f.point_forecast IS NULL) AS unavailable,
       count(*) FILTER (WHERE f.point_forecast IS NOT NULL AND scored.forecast_id IS NOT NULL) AS scored,
       count(*) FILTER (WHERE settled.forecast_id IS NOT NULL) AS settled,
       count(*) FILTER (WHERE f.point_forecast IS NOT NULL AND scored.forecast_id IS NULL
                          AND f.target_date >= %(today)s) AS awaiting_target_date,
       count(*) FILTER (WHERE f.point_forecast IS NOT NULL AND scored.forecast_id IS NULL
                          AND f.target_date < %(today)s) AS unreported_so_far
FROM forecasts f
LEFT JOIN (SELECT DISTINCT forecast_id FROM forecast_scores WHERE status = 'provisional') scored
       ON scored.forecast_id = f.forecast_id
LEFT JOIN (SELECT forecast_id FROM forecast_scores WHERE status = 'settled') settled
       ON settled.forecast_id = f.forecast_id
GROUP BY f.model_version
ORDER BY f.model_version
"""


@dataclass
class ScoringResult:
    provisional_added: int
    settled_added: int
    revisions_recorded: int  # settled actuals found revised in this run


@dataclass
class ModelCounts:
    model_version: str
    forecasts: int
    unavailable: int  # no value: the model couldn't forecast this target
    scored: int  # has at least one provisional score
    settled: int  # has its settled score
    awaiting_target_date: int  # target date is today or later, no actual yet
    unreported_so_far: int  # target date has passed with no reported price


def score_forecasts(
    conn: psycopg.Connection, today: date, settling_days: int = SETTLING_DAYS
) -> tuple[int, ScoringResult]:
    """Add provisional and settled scores and record revisions of settled actuals. Returns (run_id, result).

    `today` is the IST date; target dates before today - `settling_days` are settled. Any error rolls back
    all three steps, marks the run 'failed' and is raised again.
    """
    if settling_days < 1:
        raise ValueError("settling_days must be at least 1")
    params = {"settling_days": settling_days, "settle_before": today - timedelta(days=settling_days)}
    run_id = start_run(conn, "score")
    params["run_id"] = run_id
    try:
        with conn.cursor() as cursor:
            cursor.execute(INSERT_PROVISIONAL_SCORES, params)
            provisional_added = cursor.rowcount
            cursor.execute(INSERT_SETTLED_SCORES, params)
            settled_added = cursor.rowcount
            cursor.execute(INSERT_SETTLED_REVISIONS, params)
            revisions_recorded = cursor.rowcount
        conn.commit()
    except Exception as error:
        conn.rollback()
        finish_run(conn, run_id, "failed", error_message=f"{type(error).__name__}: {error}")
        raise
    result = ScoringResult(provisional_added, settled_added, revisions_recorded)
    finish_run(conn, run_id, "succeeded", rows_out=provisional_added + settled_added + revisions_recorded)
    return run_id, result


def scoring_counts(conn: psycopg.Connection, today: date) -> list[ModelCounts]:
    """Per model: how many forecasts are unavailable, scored, settled, waiting, or unreported so far.

    `today` is the IST date; a target date before it with no reported price counts as unreported so far.
    """
    rows = conn.execute(SCORING_COUNTS, {"today": today}).fetchall()
    return [ModelCounts(*row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score live forecasts against reported prices.")
    parser.add_argument(
        "--settling-days",
        type=int,
        default=SETTLING_DAYS,
        help=f"settle targets older than this many days (default {SETTLING_DAYS})",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    today = datetime.now(IST).date()
    with get_connection() as conn:
        run_id, result = score_forecasts(conn, today, args.settling_days)
        counts = scoring_counts(conn, today)

    print(f"Run {run_id} succeeded (settling window {args.settling_days} days).")
    print(
        f"Provisional scores added: {result.provisional_added}; settled scores added: {result.settled_added}"
    )
    print(f"Revisions of settled actuals recorded: {result.revisions_recorded}")
    print(f"\nForecasts by model (today is {today} IST):")
    header = (
        f"{'model':<20}{'forecasts':>10}{'unavailable':>13}{'scored':>8}{'settled':>9}"
        f"{'awaiting':>10}{'unreported':>12}"
    )
    print(header)
    for model in counts:
        print(
            f"{model.model_version:<20}{model.forecasts:>10}{model.unavailable:>13}{model.scored:>8}"
            f"{model.settled:>9}{model.awaiting_target_date:>10}{model.unreported_so_far:>12}"
        )


if __name__ == "__main__":
    main()
