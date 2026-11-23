# Model card: lgbm.v1 (first draft, M08)

**Status:** registered as champion in `model_registry` and in the daily run from M09 (2026-11-23); retrained
on all data known on 2026-11-23 (71,653 rows). No live scores yet. **Backtest result: it does not beat
the naive baseline.** All numbers below are from `docs/backtest_report.md`, LightGBM section.

## Model

| Item | Value |
|---|---|
| Name | `lgbm.v1` |
| Type | Gradient-boosted trees, LightGBM 4.7.0. One **global** model family for all 8 series: seven models, one per horizon (the layout chosen on 2024 data) |
| Output | Point forecast (P50) of the daily modal price, ₹/quintal, for 1–7 calendar days after the origin. No intervals yet (M11) |
| Target | log(price(target) / p_last), where p_last is the last price known at the origin. Forecast = p_last · exp(prediction) |
| Objective | L1 (absolute error), so the prediction is a median |
| Parameters | Fixed, not tuned: 300 trees, learning rate 0.05, 15 leaves, ≥ 50 rows per leaf, 80% of features per tree, seed 42 |
| Code | `src/mandi/features/build.py`, `src/mandi/models/lgbm.py`, `src/mandi/evaluate/backtest_lgbm.py`; live: `src/mandi/models/registry.py`, `register_lgbm.py`, `daily_forecast.py` |

## Intended use

Daily forecasts of wholesale tomato and onion modal prices for 8 West Bengal Agmarknet market series, shown
next to two transparent baselines (naive, seasonal naive) and scored against observed prices. It is a data
science portfolio project, **not** advice for buying, selling or storing produce. Not intended for other
markets, crops or states, or for horizons beyond 7 days.

## Data

- **Source:** data.gov.in, Agmarknet variety-wise daily prices (the only source; M01). Modal, min and max
  price per market, crop and day. **No arrivals/volume field** exists.
- **Series:** 4 tomato and 4 onion series, shortlisted in M01 for completeness. Not representative of West
  Bengal markets in general.
- **History:** 2023-01-01 onwards, the latest snapshot of each price (`prices_current`). Quarantined rows
  (M04 hard failures) excluded; anomaly-flagged prices kept.
- **Missing days** are not filled: a lag of an unreported day is NaN.

## Features (all from prices dated before the origin)

Price lags 1, 2, 3, 7, 14, 28 days (relative to p_last, NaN if unreported); days since the last price; count
of reported days, mean, std, min and max over 7/14/28-day calendar windows; min–max spread (last day, 7-day
mean); log(p_last); target day of week and month; series, crop and market (categorical). The forecast
horizon is implicit in the per-horizon model.

## Evaluation (backtest)

Rolling origin, expanding window, retrained weekly, a forecast every day. Held-out origins
2025-01-01 to 2026-11-15 (none used for any choice), rows both models scored:

| Model | n | MAE (₹) | sMAPE % | MASE |
|---|---|---|---|---|
| lgbm.v1 | 36,401 | 223.7 | 8.17 | 0.646 |
| naive | 36,401 | 213.3 | 7.82 | 0.611 |

- Worse than naive at every horizon, for both crops and in every market (54 of 56 series × horizon cells).
- Better than seasonal naive (MASE 0.602 vs. 0.754 on rows all three scored, h = 1–6).
- Better than naive on rows where the price actually changed (MASE 0.973 vs. 0.997), and slightly better on
  volatile target dates (1.855 vs. 1.911). Worse on the 38.7% of rows where the price stayed exactly the same.
- Diebold–Mariano, h = 1, vs. naive: LightGBM's absolute error higher by ₹1.2/day, p < 0.0001.

## Limitations

1. **It does not beat naive on average.** Repeated ("sticky") modal prices make "no change" hard to beat.
2. **Vintage leakage:** the backtest uses revised history, not prices as first published (blueprint §8.1).
   Agmarknet allows edits for 7 days, so only the most recent inputs are affected. Live results will be
   reported separately.
3. **Late rows:** the backtest treats the day before the origin as complete; live, it often isn't at 07:17
   IST. Live accuracy is expected to be worse.
4. **No volume, weather or festival data.** Reporting completeness is used as a proxy for volume in the
   error analysis only.
5. **Shocks are not anticipated.** At the start of a price shock its MAE is ~₹736 (naive ~₹766).
6. **Fixed hyperparameters.** No search was done; a tuned model might differ (deliberately out of scope).
7. **Overlapping forecasts:** counts are forecasts, not independent samples.

## Ethical considerations

Farmers and traders could act on a price forecast. The model is not better than "same as the last price", and
it fails at the start of price shocks, which matter most to them. Any public display must show the baselines
and the measured accuracy next to the forecast.
