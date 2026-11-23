"""Live scorecard and rolling performance on hand-made score tables (no database)."""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from mandi.evaluate.live_report import SCORE_COLUMNS, price_history, rolling_scores, scorecard

FIRST_DAY = date(2026, 12, 1)
COLUMNS = [*SCORE_COLUMNS, "mase_scale"]


def score_rows(model: str, errors: list[float], horizon: int = 1, scale: float = 100.0) -> list[dict]:
    """One score per day from FIRST_DAY on, actual 1000, forecast 1000 + error, all for series 1."""
    rows = []
    for offset, error in enumerate(errors):
        origin = FIRST_DAY + timedelta(days=offset)
        rows.append(
            {
                "forecast_id": None,
                "status": "settled",
                "series_id": 1,
                "crop": "Tomato",
                "market": "Barasat",
                "origin_date": origin,
                "horizon_days": horizon,
                "target_date": origin + timedelta(days=horizon),
                "model_version": model,
                "point_forecast": 1000.0 + error,
                "actual": 1000.0,
                "mase_scale": scale,
            }
        )
    return rows


def test_scorecard_compares_models_on_shared_rows_and_keeps_the_losing_model():
    # Naive has a 4th scored day that LightGBM lacks; it must not enter the comparison.
    scores = pd.DataFrame(
        score_rows("naive.v1", [100, -100, 200, 5000]) + score_rows("lgbm.v1", [300, 300, -300])
    )

    table = scorecard(scores, ["crop", "horizon_days"])

    assert len(table) == 1
    row = table.iloc[0]
    assert (row["crop"], row["horizon_days"], row["n"]) == ("Tomato", 1, 3)
    assert row["MAE naive.v1"] == pytest.approx(133.3)
    assert row["MASE naive.v1"] == pytest.approx(1.333)
    assert row["MAE lgbm.v1"] == pytest.approx(300.0)
    assert row["MASE lgbm.v1"] == pytest.approx(3.0)
    assert row["lowest MASE"] == "naive.v1"


def test_scorecard_marks_groups_below_the_minimum_count_but_still_shows_them():
    scores = pd.DataFrame(
        score_rows("naive.v1", [10] * 3, horizon=1)
        + score_rows("lgbm.v1", [20] * 3, horizon=1)
        + score_rows("naive.v1", [10] * 5, horizon=2)
        + score_rows("lgbm.v1", [20] * 5, horizon=2)
    )

    table = scorecard(scores, ["crop", "horizon_days"], min_n=5)

    assert table["horizon_days"].tolist() == [1, 2]
    assert table["n"].tolist() == [3, 5]
    assert table["note"].tolist() == ["insufficient data (n < 5)", ""]
    assert table["MAE naive.v1"].tolist() == [10.0, 10.0]


def test_scorecard_of_no_shared_rows_is_empty():
    assert scorecard(
        pd.DataFrame(score_rows("naive.v1", []), columns=COLUMNS), ["crop", "horizon_days"]
    ).empty


def test_rolling_scores_use_the_window_and_hide_thin_windows():
    # Target dates are 2/12, 3/12, 4/12. Window 2 days, at least 2 scores per model.
    scores = pd.DataFrame(score_rows("naive.v1", [100, 300, 500]) + score_rows("lgbm.v1", [200, 200, 200]))

    rolling = rolling_scores(scores, window_days=2, min_n=2)

    naive = rolling[rolling["model_version"] == "naive.v1"]
    assert naive["end_date"].tolist() == [date(2026, 12, 2), date(2026, 12, 3), date(2026, 12, 4)]
    assert naive["n"].tolist() == [1, 2, 2]
    assert math.isnan(naive["MAE"].iloc[0])  # one score is not drawn
    assert naive["MAE"].iloc[1:].tolist() == [200.0, 400.0]  # (100 + 300) / 2, (300 + 500) / 2
    assert naive["MASE"].iloc[1:].tolist() == [2.0, 4.0]
    lgbm = rolling[rolling["model_version"] == "lgbm.v1"]
    assert lgbm["MAE"].iloc[1:].tolist() == [200.0, 200.0]


def test_rolling_scores_of_nothing_is_an_empty_table():
    rolling = rolling_scores(pd.DataFrame(score_rows("naive.v1", []), columns=COLUMNS))
    assert rolling.empty
    assert list(rolling.columns) == ["end_date", "model_version", "n", "MAE", "MASE"]


def test_price_history_keeps_gaps_and_starts_at_the_first_day():
    prices = {date(2026, 11, 5): 3000.0, date(2026, 11, 1): 2800.0, date(2026, 11, 3): 2900.0}

    history = price_history(prices, date(2026, 11, 2))

    # 4/11 was not reported and stays absent: nothing is filled in.
    assert history.to_dict("records") == [
        {"date": date(2026, 11, 3), "price": 2900.0},
        {"date": date(2026, 11, 5), "price": 3000.0},
    ]
