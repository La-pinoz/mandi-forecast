"""LightGBM backtest: fold training rows, layout choice and the error-analysis labels (no database)."""

from datetime import date, timedelta

import pandas as pd

from mandi.evaluate.backtest import make_folds
from mandi.evaluate.backtest_lgbm import (
    PER_HORIZON,
    POOLED,
    backtest_lgbm,
    choose_layout,
    completeness_bin,
    shock_phase,
    staleness_bin,
    target_vs_last_price,
    training_rows,
)
from mandi.features.build import build_feature_table

SERIES_INFO = pd.DataFrame({"series_id": [1], "market": ["Barasat"], "crop": ["Tomato"]})
CUTOFF = date(2024, 3, 1)


def small_table() -> pd.DataFrame:
    """One complete series, 01/01-31/03/2024, price rising by 1 a day; every origin from 02/01."""
    prices = {}
    spreads = {}
    for offset in range(91):
        day = date(2024, 1, 1) + timedelta(days=offset)
        prices[day] = 1000.0 + offset
        spreads[day] = 0.1
    del prices[date(2024, 2, 20)]  # one unreported target date
    del spreads[date(2024, 2, 20)]
    origins = [date(2024, 1, 2) + timedelta(days=offset) for offset in range(83)]  # 02/01 to 24/03
    return build_feature_table({1: prices}, {1: spreads}, SERIES_INFO, origins)


def test_fold_trains_only_on_rows_whose_target_is_before_the_cutoff_and_reported():
    table = small_table()
    training = training_rows(table, CUTOFF)

    assert (training["target_date"] < CUTOFF).all()
    assert training["actual"].notna().all()
    # Origin 28/02 at horizon 1 targets 29/02: allowed. At horizon 2 it targets the cutoff day: excluded,
    # even though its origin is before the cutoff.
    origin = date(2024, 2, 28)
    assert set(training[training["origin_date"] == origin]["horizon_days"]) == {1}
    # The row whose target is the unreported 20/02 is not trained on.
    assert date(2024, 2, 20) not in set(training["target_date"])
    assert date(2024, 2, 20) in set(table["target_date"])


def test_backtest_forecasts_each_fold_origin_once_per_layout_with_training_before_the_cutoff():
    table = small_table()
    folds = make_folds(CUTOFF, last_data_date=date(2024, 3, 31))

    rows, last_models = backtest_lgbm(table, folds)

    assert set(rows["model_version"]) == {POOLED, PER_HORIZON}
    assert set(last_models) == {POOLED, PER_HORIZON}
    origins = sorted({origin for fold in folds for origin in fold.origin_dates})
    for layout in [POOLED, PER_HORIZON]:
        layout_rows = rows[rows["model_version"] == layout]
        assert sorted(set(layout_rows["origin_date"])) == origins
        assert len(layout_rows) == len(origins) * 7
        assert layout_rows["point_forecast"].notna().all()
    assert (rows["origin_date"] >= rows["train_cutoff"]).all()


def scored(model: str, origin: date, error: float) -> dict:
    return {
        "series_id": 1,
        "origin_date": origin,
        "horizon_days": 1,
        "model_version": model,
        "point_forecast": 1000.0 + error,
        "actual": 1000.0,
        "mase_scale": 100.0,
    }


def test_layout_is_chosen_on_development_origins_only():
    development = date(2024, 6, 1)
    held_out = date(2025, 6, 1)
    rows = pd.DataFrame(
        [
            scored(POOLED, development, 10),
            scored(PER_HORIZON, development, 50),
            # Held-out rows would reverse the choice if they were (wrongly) used.
            scored(POOLED, held_out, 900),
            scored(PER_HORIZON, held_out, 0),
        ]
    )
    assert choose_layout(rows) == POOLED


def test_shock_phases():
    origin = date(2024, 5, 20)
    target = origin + timedelta(days=2)

    def labels(target_label: str, volatile_days_before: list[int]) -> dict[date, str]:
        result = {target: target_label}
        for offset in volatile_days_before:
            result[origin - timedelta(days=offset)] = "volatile"
        return result

    assert shock_phase(origin, target, labels("volatile", [])) == "start"
    assert shock_phase(origin, target, labels("volatile", [9])) == "start"  # only older than 7 days
    assert shock_phase(origin, target, labels("volatile", [3])) == "during"
    assert shock_phase(origin, target, labels("normal", [12])) == "recovery"
    assert shock_phase(origin, target, labels("normal", [15])) == "calm"
    assert shock_phase(origin, target, {}) == "unlabelled"


def test_completeness_and_staleness_bins_edges():
    assert completeness_bin(19) == "a: < 70%"  # 67.9%
    assert completeness_bin(20) == "b: 70-90%"  # 71.4%
    assert completeness_bin(25) == "b: 70-90%"  # 89.3%
    assert completeness_bin(26) == "c: >= 90%"  # 92.9%
    assert [staleness_bin(days) for days in (1, 2, 3, 6, 7, 30)] == [
        "a: 1",
        "b: 2",
        "c: 3-6",
        "c: 3-6",
        "d: 7+",
        "d: 7+",
    ]
    assert completeness_bin(28) == "c: >= 90%"


def test_target_vs_last_price_labels_every_model_row_from_naives_forecast():
    origin = date(2025, 6, 1)
    rows = pd.DataFrame([scored("naive.v1", origin, 0), scored(PER_HORIZON, origin, 30)])
    rows.loc[2] = {**scored("naive.v1", origin, 20), "series_id": 2}
    labels = target_vs_last_price(rows).set_index("series_id")["target_vs_last_price"]
    assert labels.to_dict() == {1: "unchanged", 2: "changed"}
