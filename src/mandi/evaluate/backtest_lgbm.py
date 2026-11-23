"""M08: rolling-origin backtest of the global LightGBM against the baselines (blueprint 7.4, 8.1-8.4).

Run from the project root (reads DATABASE_URL from the environment or .env; writes nothing to the database):
    uv run python -m mandi.evaluate.backtest_lgbm

It uses the M07 harness unchanged: the same folds, origins, targets, "before the origin" information rule,
volatile labels, MASE scales and score tables. Fixed before any LightGBM result was seen (owner,
2026-11-23; see docs/M08-lightgbm.md):
- Per fold, LightGBM trains on the feature rows whose **target** date is before the fold cutoff and was
  reported. A row's origin being before the cutoff is not enough: its target could be on or after it.
- Two layouts are run, pooled (horizon as a feature) and per horizon. The one with the lower like-for-like
  MASE on the **development** origins (2024) becomes lgbm.v1. Nothing else is chosen from results.
- The headline comparison with the baselines uses the **held-out** origins, 2025-01-01 onwards.
- One Diebold-Mariano test: held-out, horizon 1, lgbm.v1 vs. naive, absolute-error loss.

Flow of `main()`:
1. Load each active series' prices and spreads as known now, and its market and crop.
2. `make_folds` and `backtest_baselines` (M07) give the folds and the baseline rows.
3. `build_feature_table` (features/build.py): one row per (series, origin, horizon) for every origin.
4. `backtest_lgbm`: per fold, train both layouts on the fold's training rows and forecast its origins.
5. `choose_layout` on the development origins; the winner is renamed lgbm.v1.
6. Write one row per LightGBM forecast to data/backtest/lgbm_forecasts.csv and print the report tables.
"""

from datetime import date, datetime, timedelta

import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv

from mandi.evaluate.backtest import (
    FIRST_FOLD_CUTOFF,
    MAX_HORIZON_DAYS,
    PROJECT_ROOT,
    Fold,
    availability_table,
    backtest_baselines,
    common_sample,
    make_folds,
    markdown_table,
    score_table,
    volatility_labels,
)
from mandi.evaluate.metrics import diebold_mariano
from mandi.features.build import build_feature_table
from mandi.ingest.snapshot import IST
from mandi.models.baselines import NAIVE_MODEL
from mandi.models.daily_forecast import prices_and_spreads_by_series, series_info_table
from mandi.models.lgbm import MODEL_VERSION, feature_importance, predict_prices, train
from mandi.store.db import get_connection

OUTPUT_CSV = PROJECT_ROOT / "data" / "backtest" / "lgbm_forecasts.csv"

POOLED = "lgbm_pooled"
PER_HORIZON = "lgbm_per_horizon"
LAYOUTS = {POOLED: False, PER_HORIZON: True}  # name -> per_horizon
HELD_OUT_START = date(2025, 1, 1)  # origins before: development (layout choice); from here: headline
DM_HORIZON = 1
DM_MAX_LAG = 7  # Newey-West lags, in days: errors of neighbouring days are correlated for about a week

KEY = ["series_id", "origin_date", "horizon_days"]


def training_rows(table: pd.DataFrame, train_cutoff: date) -> pd.DataFrame:
    """The feature rows a fold with this cutoff may train on: target date before the cutoff, and reported.

    Filtering on the origin would not be enough: origin cutoff - 1 at horizon 7 has its target 6 days after
    the cutoff, inside the fold's test period.
    """
    return table[(table["target_date"] < train_cutoff) & table["actual"].notna()]


def backtest_lgbm(
    table: pd.DataFrame, folds: list[Fold]
) -> tuple[pd.DataFrame, dict[str, dict[int, lgb.Booster]]]:
    """Forecast every fold's origins with both layouts, each trained on that fold's training rows.

    Returns (rows, last_models). Rows have KEY, crop, market, train_cutoff, target_date, model_version
    (POOLED or PER_HORIZON), point_forecast, actual, days_since_last and count_28. `last_models` holds each
    layout's models from the last fold (the most training data), for feature importance.
    """
    forecasts = []
    last_models = {}
    for fold in folds:
        training = training_rows(table, fold.train_cutoff)
        test = table[table["origin_date"].isin(fold.origin_dates)]
        for layout, per_horizon in LAYOUTS.items():
            models = train(training, per_horizon)
            fold_rows = test[
                [*KEY, "crop", "market", "target_date", "actual", "days_since_last", "count_28"]
            ].copy()
            fold_rows["train_cutoff"] = fold.train_cutoff
            fold_rows["model_version"] = layout
            fold_rows["point_forecast"] = predict_prices(models, test)
            forecasts.append(fold_rows)
            last_models[layout] = models
        print(
            f"fold {fold.train_cutoff}: {len(training)} training rows, {len(test)} forecast rows", flush=True
        )
    return pd.concat(forecasts, ignore_index=True), last_models


def with_period(results: pd.DataFrame) -> pd.DataFrame:
    """Add `period`: 'development' for origins before HELD_OUT_START, 'held-out' from it on."""
    results = results.copy()
    is_held_out = results["origin_date"] >= HELD_OUT_START
    results["period"] = "development"
    results.loc[is_held_out, "period"] = "held-out"
    return results


def choose_layout(lgbm_rows: pd.DataFrame) -> str:
    """The layout with the lower MASE on the development origins, on rows both layouts scored."""
    development = lgbm_rows[lgbm_rows["origin_date"] < HELD_OUT_START]
    scores = score_table(common_sample(development), []).set_index("model_version")
    return str(scores["MASE"].idxmin())


def shock_phase(origin: date, target: date, labels: dict[date, str]) -> str:
    """The shock phase of one forecast, from the M07 volatile labels of its series (analysis only).

    start     target volatile, no volatile date in the 7 days before the origin (not yet visible)
    during    target volatile, some volatile date in the 7 days before the origin
    recovery  target normal, some volatile date in the 14 days before the origin
    calm      target normal otherwise; unlabelled if the target has no label
    """
    target_label = labels.get(target)
    if target_label is None:
        return "unlabelled"
    volatile_days_before = []
    for offset in range(1, 15):
        if labels.get(origin - timedelta(days=offset)) == "volatile":
            volatile_days_before.append(offset)
    if target_label == "volatile":
        if any(offset <= 7 for offset in volatile_days_before):
            return "during"
        return "start"
    if volatile_days_before:
        return "recovery"
    return "calm"


def completeness_bin(reported_days_in_28: int) -> str:
    """Share of the 28 days before the origin that were reported, in three bins."""
    share = reported_days_in_28 / 28
    if share < 0.7:
        return "a: < 70%"
    if share < 0.9:
        return "b: 70-90%"
    return "c: >= 90%"


def staleness_bin(days_since_last: int) -> str:
    """Days from the last reported price to the origin (1 = the day before the origin was reported)."""
    if days_since_last == 1:
        return "a: 1"
    if days_since_last == 2:
        return "b: 2"
    if days_since_last <= 6:
        return "c: 3-6"
    return "d: 7+"


def target_vs_last_price(head_to_head: pd.DataFrame) -> pd.DataFrame:
    """KEY plus 'unchanged' if the target's price equals the last known price exactly, else 'changed'.

    Naive's forecast is the last known price, so its rows give the comparison. This splits rows by their
    outcome, for diagnosis only: it can't be known at the origin.
    """
    naive_rows = head_to_head[head_to_head["model_version"] == NAIVE_MODEL]
    labels = naive_rows[KEY].copy()
    labels["target_vs_last_price"] = "changed"
    labels.loc[naive_rows["point_forecast"] == naive_rows["actual"], "target_vs_last_price"] = "unchanged"
    return labels


def losing_cells(head_to_head: pd.DataFrame) -> pd.DataFrame:
    """Series x horizon cells where lgbm.v1's MASE is higher than naive's, worst first."""
    scores = score_table(head_to_head, ["market", "crop", "horizon_days"])
    by_model = scores.pivot_table(
        index=["market", "crop", "horizon_days"], columns="model_version", values="MASE"
    )
    counts = scores[scores["model_version"] == MODEL_VERSION].set_index(["market", "crop", "horizon_days"])[
        "n"
    ]
    cells = pd.DataFrame(
        {
            "n": counts,
            "naive MASE": by_model[NAIVE_MODEL],
            "lgbm MASE": by_model[MODEL_VERSION],
        }
    )
    cells["difference"] = (cells["lgbm MASE"] - cells["naive MASE"]).round(3)
    losing = cells[cells["difference"] > 0].sort_values("difference", ascending=False)
    return losing.reset_index()


def dm_test(head_to_head: pd.DataFrame) -> tuple[int, float, float, float]:
    """The one DM test: horizon DM_HORIZON, lgbm.v1 vs. naive on rows both scored.

    The loss difference of a day is the mean over series of |lgbm error| - |naive error| for that origin
    date. Returns (days, mean difference in rupees, statistic, p-value).
    """
    rows = head_to_head[head_to_head["horizon_days"] == DM_HORIZON].copy()
    rows["absolute_error"] = (rows["point_forecast"] - rows["actual"]).abs()
    errors = rows.pivot_table(
        index=["origin_date", "series_id"], columns="model_version", values="absolute_error"
    )
    errors["difference"] = errors[MODEL_VERSION] - errors[NAIVE_MODEL]
    daily = errors.groupby(level="origin_date")["difference"].mean().sort_index()
    statistic, p_value = diebold_mariano(daily, DM_MAX_LAG)
    return len(daily), float(daily.mean()), statistic, p_value


def print_report(
    results: pd.DataFrame,
    chosen: str,
    last_models: dict[str, dict[int, lgb.Booster]],
    labels: dict[int, dict[date, str]],
) -> None:
    """Print the LightGBM section's tables as markdown. `results` holds every model's rows with `period`."""
    other_layout = PER_HORIZON if chosen == POOLED else POOLED
    lgbm_models = [MODEL_VERSION, other_layout]

    print(f"### Layout choice (development origins, before {HELD_OUT_START})\n")
    print(f"Chosen: **{chosen}** (renamed {MODEL_VERSION}).\n")
    development = results[results["period"] == "development"]
    development_layouts = development[development["model_version"].isin([*lgbm_models, NAIVE_MODEL])]
    print(markdown_table(score_table(common_sample(development_layouts), [])))
    print("\nBy horizon:\n")
    print(markdown_table(score_table(common_sample(development_layouts), ["horizon_days"])))

    print("\n### Both layouts and naive, by period (like-for-like)\n")
    layouts = results[results["model_version"].isin([*lgbm_models, NAIVE_MODEL])]
    print(markdown_table(score_table(common_sample(layouts), ["period"])))

    held_out = results[(results["period"] == "held-out") & (results["model_version"] != other_layout)]
    print(f"\n### Held-out ({HELD_OUT_START} onwards): availability\n")
    print(markdown_table(availability_table(held_out, [])))

    head_to_head = common_sample(held_out[held_out["model_version"].isin([MODEL_VERSION, NAIVE_MODEL])])
    print("\n### Held-out, like-for-like: lgbm.v1 vs. naive\n")
    print(markdown_table(score_table(head_to_head, [])))
    print("\n### Held-out, all three models on rows all scored (horizons 1-6)\n")
    print(markdown_table(score_table(common_sample(held_out), [])))
    for title, by in [
        ("horizon", ["horizon_days"]),
        ("crop", ["crop"]),
        ("market", ["market"]),
        ("series", ["market", "crop"]),
        ("normal vs. volatile target dates", ["regime"]),
    ]:
        print(f"\n### Held-out, lgbm.v1 vs. naive, by {title}\n")
        print(markdown_table(score_table(head_to_head, by)))

    print("\n### Error analysis: series x horizon cells where lgbm.v1 loses to naive (MASE)\n")
    losing = losing_cells(head_to_head)
    cell_count = head_to_head.groupby(["series_id", "horizon_days"]).ngroups
    print(f"lgbm.v1 loses in {len(losing)} of {cell_count} cells.\n")
    if len(losing) > 0:
        print(markdown_table(losing))

    analysis = head_to_head.copy()
    phases = []
    for row in analysis.itertuples(index=False):
        phases.append(shock_phase(row.origin_date, row.target_date, labels[row.series_id]))
    analysis["shock_phase"] = phases
    analysis["completeness_28d"] = analysis["count_28"].apply(completeness_bin)
    analysis["days_since_last_bin"] = analysis["days_since_last"].apply(staleness_bin)
    print("\n### Error analysis: shock phase\n")
    print(markdown_table(score_table(analysis, ["shock_phase"])))
    print("\n### Error analysis: share of the 28 days before the origin reported\n")
    print(markdown_table(score_table(analysis, ["completeness_28d"])))
    print("\n### Error analysis: days since the last observation at the origin\n")
    print(markdown_table(score_table(analysis, ["days_since_last_bin"])))
    print("\n### Error analysis: target price equal to the last known price, or changed (by outcome)\n")
    analysis = analysis.merge(target_vs_last_price(head_to_head), on=KEY)
    print(markdown_table(score_table(analysis, ["target_vs_last_price"])))
    print("\nBy horizon:\n")
    print(markdown_table(score_table(analysis, ["target_vs_last_price", "horizon_days"])))

    print("\n### Feature importance (share of total gain, lgbm.v1, last fold)\n")
    importance = feature_importance(last_models[chosen]).round(1).reset_index()
    importance.columns = ["feature", "gain %"]
    print(markdown_table(importance))

    days, mean_difference, statistic, p_value = dm_test(head_to_head)
    print(f"\n### Diebold-Mariano test (held-out, horizon {DM_HORIZON}, lgbm.v1 vs. naive)\n")
    print(
        f"Days: {days}. Mean daily |error| difference (lgbm - naive): {mean_difference:.1f} rupees. "
        f"Newey-West lags: {DM_MAX_LAG}. Statistic: {statistic:.2f}. Two-sided p-value: {p_value:.4f}."
    )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    as_of = datetime.now(IST)
    with get_connection() as conn:
        prices, spreads = prices_and_spreads_by_series(conn, as_of)
        series_info = series_info_table(conn)

    first_data_date = min(min(series_prices) for series_prices in prices.values() if series_prices)
    last_data_date = max(max(series_prices) for series_prices in prices.values() if series_prices)
    folds = make_folds(FIRST_FOLD_CUTOFF, last_data_date)

    baseline_rows = backtest_baselines(prices, folds).merge(series_info, on="series_id", how="left")

    # Every origin from the day after the first price, so training rows start in 2023.
    last_origin = last_data_date - timedelta(days=MAX_HORIZON_DAYS)
    origins = []
    origin = first_data_date + timedelta(days=1)
    while origin <= last_origin:
        origins.append(origin)
        origin += timedelta(days=1)
    table = build_feature_table(prices, spreads, series_info, origins)

    lgbm_rows, last_models = backtest_lgbm(table, folds)
    # Same regime label and MASE scale as the baseline rows of the same (series, origin, horizon).
    labels_and_scales = baseline_rows[baseline_rows["model_version"] == NAIVE_MODEL][
        [*KEY, "regime", "mase_scale"]
    ]
    lgbm_rows = lgbm_rows.merge(labels_and_scales, on=KEY, how="left")
    lgbm_rows["unavailable_reason"] = None

    chosen = choose_layout(lgbm_rows)
    lgbm_rows.loc[lgbm_rows["model_version"] == chosen, "model_version"] = MODEL_VERSION

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    lgbm_rows.to_csv(OUTPUT_CSV, index=False)

    analysis_columns = table[[*KEY, "days_since_last", "count_28"]]
    baseline_rows = baseline_rows.merge(analysis_columns, on=KEY, how="left")
    results = with_period(pd.concat([baseline_rows, lgbm_rows], ignore_index=True))

    labels = {}
    for series_id, series_prices in prices.items():
        _, labels[series_id] = volatility_labels(series_prices)

    print(f"\nInputs: prices as known at {as_of:%Y-%m-%d %H:%M %Z}; latest arrival date {last_data_date}.")
    print(f"Feature rows: {len(table)} (origins {origins[0]} to {origins[-1]}). Folds: {len(folds)}.")
    print(f"LightGBM rows written to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}: {len(lgbm_rows)}.\n")
    print_report(results, chosen, last_models, labels)


if __name__ == "__main__":
    main()
