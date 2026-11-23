# Backtest report

Rolling-origin backtest results for every model, all run through the same harness
(`src/mandi/evaluate/backtest.py`). This page covers **backtest** results only. Live results (forecasts made
each morning and scored as actuals arrive) are a separate record and are never mixed into these tables.

Sections: [setup](#setup) · [limitations](#limitations) · [baselines](#baselines-m07) · [LightGBM](#lightgbm-m08).

---

## Setup

Fixed before any results were looked at (2026-11-23).

| Item | Choice |
|---|---|
| Series | The 8 active series in `series` (4 tomato, 4 onion, West Bengal) |
| Inputs | The latest history: each observation's latest vintage as of the run (what `prices_current` holds). Quarantined rows are not in it; anomaly-flagged prices are (they are real observations) |
| Folds | Rolling origin, expanding window. A fold starts every 7 days from 2024-01-01. Its training data is every price with an arrival date **before** its cutoff, so every fold trains on at least all of 2023 |
| Origins | Every day: the 7 days starting at each fold's cutoff. 150 folds, origins 2024-01-01 to 2026-11-15 (1,050 days) |
| Information at origin D | Prices with arrival date **before D**. The live run starts at 07:17 IST, when day D has no prices yet, so the backtest gets the same information as live |
| Targets | D + 1 … D + 7 **calendar** days. The last origin is the last data date minus 7, so every origin has all 7 targets |
| Same for every model | Same series, origins, targets and inputs. Each forecast is scored only if it has a value and its target date was reported. Unreported target dates are counted, not scored |
| No random splits | None anywhere |

**Metrics** (`src/mandi/evaluate/metrics.py`), on forecast F and actual A (modal price, ₹/quintal):

- **MAE** = mean |F − A| in ₹/quintal.
- **sMAPE** = mean of 200·|F − A| / (|F| + |A|), in percent (0–200).
- **MASE** = mean of |F − A| / scale. The scale is per series and per fold: the mean |price(t) − price(t − 7)|
  over training days t where **both** t and t − 7 calendar days are reported. A pair with an unreported day is
  skipped, never filled from a neighbouring day. MASE < 1 means smaller errors than the seasonal-naive rule
  made *in that fold's training period*; it is not a head-to-head comparison on the test rows.

**Volatile vs. normal** (defined before looking at results): for a target date t, the 7-day change is
|log(price(t) / price(t − 7))|. A target is **volatile** if its change is in the top decile (≥ the 90th
percentile) of that series' 7-day changes over its **full history**, and **normal** otherwise. If day t − 7 is
unreported, the target is **unlabelled** and shown on its own row. These labels describe target dates, not
model inputs, so using the full history to set the threshold is not leakage. Thresholds are listed in the
results.

**Caveat on the volatile split:** the label uses exactly the price pair that seasonal naive's error is made of
(its error is price(t − 7) − price(t)). So seasonal naive is bad on volatile dates *by construction*, and every
seasonal-naive forecast that can be scored is labelled. The split is fair for comparing models that are not
built from that pair (naive, LightGBM).

---

## Limitations

1. **Vintage leakage (blueprint §8.1).** The backtest uses each price as it reads *now*, after any revisions,
   not as first published. A simulated January 2025 forecast therefore sees January 2025 inputs as later
   corrected. Rolling origins prevent target leakage, not this. The effect is limited: Agmarknet allows edits for
   only 7 days, so only the most recent inputs of each origin can be affected. data.gov.in does not expose past
   versions, so this is **documented, not engineered around**. Live results, from the pipeline's own vintages,
   measure the real-time version and are reported separately.
2. **Late rows are also invisible.** Live, the previous day's rows are often incomplete at 07:17 (rows arrive up
   to ~2 days late). The backtest sees every day before the origin as complete, so it is more optimistic than
   live, especially for horizon 1 of naive.
3. **Overlapping forecasts.** Each target date is forecast from 7 origins, and neighbouring origins share
   inputs, so errors are strongly correlated. The counts `n` are numbers of forecasts, not independent samples;
   differences between models should not be read through naive standard errors.
4. **Selected series.** The 8 series were shortlisted for completeness (M01). Results do not describe
   West Bengal markets in general.

---

## Baselines (M07)

Run on 2026-11-23 with:

```bash
uv run python -m mandi.evaluate.backtest
```

Inputs: prices as known at 2026-11-23 19:34 IST; latest arrival date 2026-11-22. 8 series, 150 folds (cutoffs
2024-01-01 to 2026-11-09), origins 2024-01-01 to 2026-11-15, 117,600 forecast rows (8 series × 1,050 origins
× 7 horizons × 2 models). Per-forecast rows: `data/backtest/baseline_forecasts.csv` (not committed; re-created
by the command).

The models (`src/mandi/models/baselines.py`, the same functions the live run uses):

- **naive.v1**: the latest reported price before the origin, for every horizon.
- **seasonal_naive.v1**: the price exactly 7 calendar days before the target. Unavailable if that day is
  unreported or not yet known at the origin; no other day is used instead.

### Findings

1. **Naive is the baseline to beat.** On the rows both models scored (n = 44,847), naive has MAE ₹232 vs.
   ₹312, sMAPE 7.7% vs. 10.3%, MASE 0.66 vs. 0.88. It is better at every horizon from 1 to 5.
2. **At horizon 6 the two baselines are the same forecast.** Under the "before D" rule, naive uses day D − 1
   (when reported) and seasonal naive's lag day for h = 6 is also D − 1. Checked: identical on all 7,890 rows
   where seasonal naive had a value.
3. **Seasonal naive is never available at horizon 7**, because its lag day is the origin itself, which isn't
   known at 07:17. The same happens live (M06). Over horizons 1–6 it is available 93.9% of the time; 80.5%
   over all horizons.
4. **Naive error grows with horizon** (MASE 0.41 at h = 1 to 0.97 at h = 7). Seasonal naive's error is flat
   (MASE ≈ 0.88 at every horizon), because its error is always a 7-day price change whatever the horizon. Using
   last week's same weekday adds no information here: there is no useful day-of-week pattern for it to catch.
5. **Tomato is harder than onion in rupees and percent** (naive MAE ₹354 vs. ₹133, sMAPE 11.1% vs. 5.1%), but
   their MASE is similar (0.71 vs. 0.70) because MASE divides by each series' own typical weekly change. Habra
   tomato is the hardest series (naive sMAPE 15.7%).
6. **Volatile target dates are much harder:** naive's MAE is ₹712 there vs. ₹199 on normal dates (sMAPE 25.3%
   vs. 6.4%). 4,907 of naive's 55,247 scored forecasts (8.9%) fall on volatile dates. It is below 10% because
   the thresholds use the full history, including the 2023 shocks, which are not in the test period.
7. **Unreported targets:** 3,553 naive forecasts (6.0%) could not be scored because the target date was
   unreported.

### Volatile thresholds (90th percentile of |7-day log change|, full history)

| series_id | market | crop | threshold |
|---|---|---|---|
| 1 | Barasat | Tomato | 0.3365 |
| 2 | Habra | Tomato | 0.4055 |
| 3 | Sealdah Koley Market | Tomato | 0.3185 |
| 4 | Karsiyang(Matigara) | Tomato | 0.2688 |
| 5 | Asansol | Onion | 0.1391 |
| 6 | Barasat | Onion | 0.2211 |
| 7 | Medinipur(West) | Onion | 0.1542 |
| 8 | Bankura Sadar | Onion | 0.1744 |

(0.3365 ≈ a 40% rise or 29% fall within a week.)

### Availability, overall

| model_version | forecasts | available | available % | target unreported | scored |
|---|---|---|---|---|---|
| naive.v1 | 58800 | 58800 | 100.0 | 3553 | 55247 |
| seasonal_naive.v1 | 58800 | 47337 | 80.5 | 2490 | 44847 |

### Availability by horizon

| horizon_days | model_version | forecasts | available | available % | target unreported | scored |
|---|---|---|---|---|---|---|
| 1 | naive.v1 | 8400 | 8400 | 100.0 | 506 | 7894 |
| 1 | seasonal_naive.v1 | 8400 | 7890 | 93.9 | 414 | 7476 |
| 2 | naive.v1 | 8400 | 8400 | 100.0 | 506 | 7894 |
| 2 | seasonal_naive.v1 | 8400 | 7890 | 93.9 | 414 | 7476 |
| 3 | naive.v1 | 8400 | 8400 | 100.0 | 506 | 7894 |
| 3 | seasonal_naive.v1 | 8400 | 7889 | 93.9 | 414 | 7475 |
| 4 | naive.v1 | 8400 | 8400 | 100.0 | 506 | 7894 |
| 4 | seasonal_naive.v1 | 8400 | 7889 | 93.9 | 414 | 7475 |
| 5 | naive.v1 | 8400 | 8400 | 100.0 | 508 | 7892 |
| 5 | seasonal_naive.v1 | 8400 | 7889 | 93.9 | 416 | 7473 |
| 6 | naive.v1 | 8400 | 8400 | 100.0 | 510 | 7890 |
| 6 | seasonal_naive.v1 | 8400 | 7890 | 93.9 | 418 | 7472 |
| 7 | naive.v1 | 8400 | 8400 | 100.0 | 511 | 7889 |
| 7 | seasonal_naive.v1 | 8400 | 0 | 0.0 | 0 | 0 |

### Each model on its own scored rows

| model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|
| naive.v1 | 55247 | 246.5 | 8.18 | 0.703 |
| seasonal_naive.v1 | 44847 | 311.9 | 10.28 | 0.882 |

These are different samples (naive includes horizon 7 and the days seasonal naive couldn't forecast). Compare
the models with the like-for-like tables.

### Like-for-like: rows both models scored

| model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|
| naive.v1 | 44847 | 232.0 | 7.71 | 0.659 |
| seasonal_naive.v1 | 44847 | 311.9 | 10.28 | 0.882 |

### Like-for-like by horizon

| horizon_days | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| 1 | naive.v1 | 7476 | 143.4 | 4.86 | 0.412 |
| 1 | seasonal_naive.v1 | 7476 | 312.3 | 10.3 | 0.884 |
| 2 | naive.v1 | 7476 | 184.0 | 6.16 | 0.524 |
| 2 | seasonal_naive.v1 | 7476 | 312.2 | 10.29 | 0.883 |
| 3 | naive.v1 | 7475 | 218.6 | 7.29 | 0.621 |
| 3 | seasonal_naive.v1 | 7475 | 312.1 | 10.28 | 0.883 |
| 4 | naive.v1 | 7475 | 250.9 | 8.33 | 0.712 |
| 4 | seasonal_naive.v1 | 7475 | 311.8 | 10.27 | 0.882 |
| 5 | naive.v1 | 7473 | 283.7 | 9.38 | 0.804 |
| 5 | seasonal_naive.v1 | 7473 | 311.6 | 10.26 | 0.881 |
| 6 | naive.v1 | 7472 | 311.5 | 10.26 | 0.88 |
| 6 | seasonal_naive.v1 | 7472 | 311.5 | 10.26 | 0.88 |

### By horizon (each model on its own rows)

| horizon_days | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| 1 | naive.v1 | 7894 | 142.2 | 4.81 | 0.411 |
| 1 | seasonal_naive.v1 | 7476 | 312.3 | 10.3 | 0.884 |
| 2 | naive.v1 | 7894 | 182.8 | 6.12 | 0.523 |
| 2 | seasonal_naive.v1 | 7476 | 312.2 | 10.29 | 0.883 |
| 3 | naive.v1 | 7894 | 217.4 | 7.24 | 0.62 |
| 3 | seasonal_naive.v1 | 7475 | 312.1 | 10.28 | 0.883 |
| 4 | naive.v1 | 7894 | 249.8 | 8.29 | 0.712 |
| 4 | seasonal_naive.v1 | 7475 | 311.8 | 10.27 | 0.882 |
| 5 | naive.v1 | 7892 | 282.8 | 9.36 | 0.806 |
| 5 | seasonal_naive.v1 | 7473 | 311.6 | 10.26 | 0.881 |
| 6 | naive.v1 | 7890 | 311.7 | 10.26 | 0.886 |
| 6 | seasonal_naive.v1 | 7472 | 311.5 | 10.26 | 0.88 |
| 7 | naive.v1 | 7889 | 339.2 | 11.17 | 0.965 |

### By crop

| crop | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| Onion | naive.v1 | 26934 | 133.2 | 5.1 | 0.696 |
| Onion | seasonal_naive.v1 | 21461 | 167.7 | 6.36 | 0.873 |
| Tomato | naive.v1 | 28313 | 354.4 | 11.1 | 0.71 |
| Tomato | seasonal_naive.v1 | 23386 | 444.2 | 13.87 | 0.89 |

### By market

Barasat has two series (tomato and onion); they are pooled here and split in the next table.

| market | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| Asansol | naive.v1 | 7060 | 113.8 | 4.33 | 0.637 |
| Asansol | seasonal_naive.v1 | 5849 | 142.0 | 5.33 | 0.794 |
| Bankura Sadar | naive.v1 | 6265 | 126.9 | 4.77 | 0.678 |
| Bankura Sadar | seasonal_naive.v1 | 4778 | 159.2 | 5.96 | 0.852 |
| Barasat | naive.v1 | 14197 | 232.7 | 8.21 | 0.711 |
| Barasat | seasonal_naive.v1 | 11774 | 302.6 | 10.63 | 0.922 |
| Habra | naive.v1 | 7105 | 425.8 | 15.71 | 0.815 |
| Habra | seasonal_naive.v1 | 5886 | 520.7 | 19.07 | 0.995 |
| Karsiyang(Matigara) | naive.v1 | 7036 | 306.0 | 7.63 | 0.597 |
| Karsiyang(Matigara) | seasonal_naive.v1 | 5784 | 398.3 | 9.89 | 0.776 |
| Medinipur(West) | naive.v1 | 6528 | 136.3 | 5.42 | 0.746 |
| Medinipur(West) | seasonal_naive.v1 | 4974 | 166.5 | 6.54 | 0.91 |
| Sealdah Koley Market | naive.v1 | 7056 | 375.8 | 10.51 | 0.73 |
| Sealdah Koley Market | seasonal_naive.v1 | 5802 | 454.4 | 12.76 | 0.881 |

### By series

| market | crop | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|---|
| Asansol | Onion | naive.v1 | 7060 | 113.8 | 4.33 | 0.637 |
| Asansol | Onion | seasonal_naive.v1 | 5849 | 142.0 | 5.33 | 0.794 |
| Bankura Sadar | Onion | naive.v1 | 6265 | 126.9 | 4.77 | 0.678 |
| Bankura Sadar | Onion | seasonal_naive.v1 | 4778 | 159.2 | 5.96 | 0.852 |
| Barasat | Onion | naive.v1 | 7081 | 155.3 | 5.88 | 0.725 |
| Barasat | Onion | seasonal_naive.v1 | 5860 | 201.3 | 7.56 | 0.938 |
| Barasat | Tomato | naive.v1 | 7116 | 309.7 | 10.52 | 0.697 |
| Barasat | Tomato | seasonal_naive.v1 | 5914 | 403.0 | 13.68 | 0.907 |
| Habra | Tomato | naive.v1 | 7105 | 425.8 | 15.71 | 0.815 |
| Habra | Tomato | seasonal_naive.v1 | 5886 | 520.7 | 19.07 | 0.995 |
| Karsiyang(Matigara) | Tomato | naive.v1 | 7036 | 306.0 | 7.63 | 0.597 |
| Karsiyang(Matigara) | Tomato | seasonal_naive.v1 | 5784 | 398.3 | 9.89 | 0.776 |
| Medinipur(West) | Onion | naive.v1 | 6528 | 136.3 | 5.42 | 0.746 |
| Medinipur(West) | Onion | seasonal_naive.v1 | 4974 | 166.5 | 6.54 | 0.91 |
| Sealdah Koley Market | Tomato | naive.v1 | 7056 | 375.8 | 10.51 | 0.73 |
| Sealdah Koley Market | Tomato | seasonal_naive.v1 | 5802 | 454.4 | 12.76 | 0.881 |

### Normal vs. volatile target dates

| regime | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| normal | naive.v1 | 47414 | 199.4 | 6.44 | 0.55 |
| normal | seasonal_naive.v1 | 40632 | 238.8 | 7.59 | 0.653 |
| unlabelled | naive.v1 | 2926 | 229.2 | 7.53 | 0.717 |
| volatile | naive.v1 | 4907 | 712.4 | 25.33 | 2.177 |
| volatile | seasonal_naive.v1 | 4215 | 1016.5 | 36.13 | 3.089 |

Seasonal naive has no unlabelled rows: a scored seasonal-naive forecast always has a reported t − 7, which is
exactly what the label needs (see the caveat in [setup](#setup)).

---

## LightGBM (M08)

Run on 2026-11-23 with:

```bash
uv run python -m mandi.evaluate.backtest_lgbm
```

Inputs: prices as known at 2026-11-23 21:39 IST; latest arrival date 2026-11-22. Same 150 folds, origins,
targets, information rule, volatile labels and MASE scales as the baselines above. 77,042 feature rows
(origins 2023-01-02 to 2026-11-15; the 2023 origins are training-only). Per-forecast rows:
`data/backtest/lgbm_forecasts.csv` (not committed; re-created by the command). The run was repeated once
(to add the "unchanged vs. changed" table); every other table was identical.

**The model** (`src/mandi/models/lgbm.py`, features in `src/mandi/features/build.py`): one global LightGBM
for all 8 series. It predicts log(price(target) / p_last), where p_last is the last price known at the
origin, and forecasts p_last · exp(prediction), so a prediction of 0 is exactly naive. Features: lags 1, 2,
3, 7, 14, 28 days (NaN when unreported), days since the last price, mean/std/min/max and count of reported
days over 7/14/28-day calendar windows, the min–max spread, target day of week and month, series/crop/market
(categorical), and log(p_last). Every feature uses only prices dated before the origin. Fixed parameters, no
tuning: 300 trees, learning rate 0.05, 15 leaves, ≥ 50 rows per leaf, 80% of features per tree, L1 objective
(median = P50). Each fold trains on the rows whose **target** date is before its cutoff.

**Fixed before any LightGBM result was seen:** the one choice made from results (pooled model with horizon as
a feature vs. one model per horizon) uses only the **development** origins (2024); the headline comparison
uses the **held-out** origins (2025-01-01 to 2026-11-15), which played no part in any choice. The shock
phases, completeness bins and the single Diebold–Mariano test were also defined in advance
(`docs/M08-lightgbm.md`).

### Findings

1. **LightGBM does not beat naive.** Held-out, on rows both scored (n = 36,401): MASE 0.646 vs. 0.611, MAE
   ₹223.7 vs. ₹213.3, sMAPE 8.17% vs. 7.82%. It does beat seasonal naive (MASE 0.602 vs. 0.754 on the h = 1–6
   rows all three models scored). Naive remains the baseline to beat.
2. **It loses almost everywhere, and more at longer horizons.** Worse at every horizon (MASE 0.375 vs. 0.373
   at h = 1, 0.892 vs. 0.824 at h = 7), for both crops and in every market, and in 54 of 56 series × horizon
   cells. The largest gaps are onion at h = 5–7 (Barasat onion h = 7: 0.983 vs. 0.856). The smallest are
   Habra tomato, the most volatile series.
3. **Why: prices often don't change.** In 38.7% of held-out rows (14,079; 52% at h = 1, 29% at h = 7) the
   target's modal price is *exactly* the last known price. Naive is perfect on those rows; LightGBM, which
   predicts small moves, averages ₹40.3 there. On the rows where the price did change, LightGBM is better than
   naive **at every horizon** (overall MAE ₹339.3 vs. ₹347.9, MASE 0.973 vs. 0.997). This split is by outcome
   (it can't be known at the origin), so it's a diagnosis, not a usable rule. It shows the model has some
   signal, and that the overall loss comes from the cost of predicting movement on the many days without any.
4. **Shocks: slightly better than naive, but both fail.** On volatile target dates LightGBM's MASE is 1.855
   vs. 1.911. By phase: at the **start** of a shock 2.243 vs. 2.317 (MAE ₹736 vs. ₹766), **during** 1.401 vs.
   1.436, but in **recovery** 0.578 vs. 0.539 and on **calm** days 0.497 vs. 0.453. Neither model anticipates
   a shock; LightGBM's small advantage there is outweighed by its losses on the far more common calm days.
5. **Completeness and staleness (volume stand-ins; there is no volume field).** 82% of held-out rows have
   ≥ 90% of the previous 28 days reported; only 14 rows have < 70%, too few to conclude anything. LightGBM loses
   in both large bins. 95% of rows have yesterday reported. When the last price is 3–6 days old (n = 359) the
   gap widens (MASE 0.863 vs. 0.760). There were no held-out rows with a last price 7+ days old.
6. **The layout choice barely mattered.** Per horizon was chosen on the development origins by MASE 0.928
   vs. 0.931 (pooled). On held-out, pooled was marginally better (0.641 vs. 0.646). Both lose to naive in both
   periods. Per horizon was clearly better at h = 1 in development (0.503 vs. 0.590).
7. **Feature importance (gain, last fold).** The largest shares are the price level `log_last_price` (13.5%),
   `target_month` (8.8%), 28-day volatility `std_28` (7.7%) and `lag_28` (6.3%). The level and the month
   together describe "is the price high for this time of year", which matches how mandi prices behave:
   seasonal harvests bring new arrivals and push unusually high prices back down. Volatility sets how large a
   move is plausible. Day of week has almost none (0.3%), consistent with the baselines finding no weekly
   pattern. `lag_1` and `days_since_last` have none: yesterday is reported in 95% of rows, so they barely vary.
   Gain shows what the trees split on, not causal effects, and it is from one fold's models.
8. **Diebold–Mariano (the one test): held-out, h = 1, LightGBM vs. naive.** Over 684 days, LightGBM's absolute
   error was higher by ₹1.2 per day on average; statistic 7.59, two-sided p < 0.0001 (Newey–West, 7 lags). At
   h = 1 the difference is tiny in rupees but consistent: LightGBM is reliably slightly **worse**, not better.
   One test only; other horizons were not tested.

### Layout choice (development origins, before 2025-01-01)

Chosen: **lgbm_per_horizon** (renamed lgbm.v1).

| model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|
| lgbm.v1 | 18846 | 327.1 | 9.3 | 0.928 |
| lgbm_pooled | 18846 | 328.8 | 9.34 | 0.931 |
| naive.v1 | 18846 | 310.7 | 8.87 | 0.881 |

By horizon:

| horizon_days | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| 1 | lgbm.v1 | 2693 | 176.1 | 5.13 | 0.503 |
| 1 | lgbm_pooled | 2693 | 207.1 | 5.97 | 0.59 |
| 1 | naive.v1 | 2693 | 170.3 | 4.97 | 0.486 |
| 2 | lgbm.v1 | 2693 | 235.1 | 6.77 | 0.671 |
| 2 | lgbm_pooled | 2693 | 249.5 | 7.17 | 0.71 |
| 2 | naive.v1 | 2693 | 222.4 | 6.44 | 0.633 |
| 3 | lgbm.v1 | 2693 | 286.0 | 8.19 | 0.815 |
| 3 | lgbm_pooled | 2693 | 290.6 | 8.3 | 0.827 |
| 3 | naive.v1 | 2693 | 268.2 | 7.72 | 0.765 |
| 4 | lgbm.v1 | 2692 | 337.5 | 9.62 | 0.959 |
| 4 | lgbm_pooled | 2692 | 335.0 | 9.53 | 0.949 |
| 4 | naive.v1 | 2692 | 315.2 | 9.02 | 0.894 |
| 5 | lgbm.v1 | 2691 | 381.6 | 10.81 | 1.078 |
| 5 | lgbm_pooled | 2691 | 373.3 | 10.57 | 1.056 |
| 5 | naive.v1 | 2691 | 360.1 | 10.24 | 1.02 |
| 6 | lgbm.v1 | 2691 | 416.6 | 11.8 | 1.179 |
| 6 | lgbm_pooled | 2691 | 408.3 | 11.51 | 1.15 |
| 6 | naive.v1 | 2691 | 401.1 | 11.33 | 1.133 |
| 7 | lgbm.v1 | 2693 | 456.7 | 12.8 | 1.293 |
| 7 | lgbm_pooled | 2693 | 437.8 | 12.34 | 1.234 |
| 7 | naive.v1 | 2693 | 437.5 | 12.36 | 1.237 |

### Both layouts and naive, by period (like-for-like)

| period | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| development | lgbm.v1 | 18846 | 327.1 | 9.3 | 0.928 |
| development | lgbm_pooled | 18846 | 328.8 | 9.34 | 0.931 |
| development | naive.v1 | 18846 | 310.7 | 8.87 | 0.881 |
| held-out | lgbm.v1 | 36401 | 223.7 | 8.17 | 0.646 |
| held-out | lgbm_pooled | 36401 | 222.3 | 8.16 | 0.641 |
| held-out | naive.v1 | 36401 | 213.3 | 7.82 | 0.611 |

### Held-out (2025-01-01 onwards): availability

| model_version | forecasts | available | available % | target unreported | scored |
|---|---|---|---|---|---|
| lgbm.v1 | 38304 | 38304 | 100.0 | 1903 | 36401 |
| naive.v1 | 38304 | 38304 | 100.0 | 1903 | 36401 |
| seasonal_naive.v1 | 38304 | 31177 | 81.4 | 1365 | 29812 |

### Held-out, like-for-like: lgbm.v1 vs. naive

| model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|
| lgbm.v1 | 36401 | 223.7 | 8.17 | 0.646 |
| naive.v1 | 36401 | 213.3 | 7.82 | 0.611 |

### Held-out, all three models on rows all scored (horizons 1-6)

| model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|
| lgbm.v1 | 29812 | 210.1 | 7.67 | 0.602 |
| naive.v1 | 29812 | 201.2 | 7.37 | 0.573 |
| seasonal_naive.v1 | 29812 | 266.2 | 9.73 | 0.754 |

### Held-out, lgbm.v1 vs. naive, by horizon

| horizon_days | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| 1 | lgbm.v1 | 5201 | 128.8 | 4.77 | 0.375 |
| 1 | naive.v1 | 5201 | 127.6 | 4.73 | 0.373 |
| 2 | lgbm.v1 | 5201 | 166.3 | 6.1 | 0.479 |
| 2 | naive.v1 | 5201 | 162.2 | 5.95 | 0.467 |
| 3 | lgbm.v1 | 5201 | 198.7 | 7.27 | 0.568 |
| 3 | naive.v1 | 5201 | 191.1 | 6.99 | 0.545 |
| 4 | lgbm.v1 | 5202 | 227.0 | 8.29 | 0.654 |
| 4 | naive.v1 | 5202 | 216.0 | 7.91 | 0.619 |
| 5 | lgbm.v1 | 5201 | 255.6 | 9.34 | 0.737 |
| 5 | naive.v1 | 5201 | 242.8 | 8.9 | 0.695 |
| 6 | lgbm.v1 | 5199 | 282.4 | 10.24 | 0.816 |
| 6 | naive.v1 | 5199 | 265.4 | 9.71 | 0.758 |
| 7 | lgbm.v1 | 5196 | 307.0 | 11.17 | 0.892 |
| 7 | naive.v1 | 5196 | 288.2 | 10.55 | 0.824 |

### Held-out, lgbm.v1 vs. naive, by crop

| crop | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| Onion | lgbm.v1 | 17756 | 121.0 | 5.36 | 0.641 |
| Onion | naive.v1 | 17756 | 111.3 | 4.93 | 0.593 |
| Tomato | lgbm.v1 | 18645 | 321.4 | 10.84 | 0.651 |
| Tomato | naive.v1 | 18645 | 310.5 | 10.58 | 0.629 |

### Held-out, lgbm.v1 vs. naive, by market

| market | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| Asansol | lgbm.v1 | 4554 | 96.7 | 4.38 | 0.555 |
| Asansol | naive.v1 | 4554 | 88.7 | 4.02 | 0.51 |
| Bankura Sadar | lgbm.v1 | 4199 | 110.4 | 4.73 | 0.618 |
| Bankura Sadar | naive.v1 | 4199 | 99.1 | 4.24 | 0.562 |
| Barasat | lgbm.v1 | 9241 | 222.3 | 8.7 | 0.678 |
| Barasat | naive.v1 | 9241 | 207.8 | 8.22 | 0.63 |
| Habra | lgbm.v1 | 4662 | 364.7 | 14.27 | 0.673 |
| Habra | naive.v1 | 4662 | 361.8 | 14.36 | 0.668 |
| Karsiyang(Matigara) | lgbm.v1 | 4697 | 306.1 | 8.22 | 0.629 |
| Karsiyang(Matigara) | naive.v1 | 4697 | 294.9 | 7.99 | 0.607 |
| Medinipur(West) | lgbm.v1 | 4400 | 134.0 | 6.16 | 0.718 |
| Medinipur(West) | naive.v1 | 4400 | 128.6 | 5.86 | 0.692 |
| Sealdah Koley Market | lgbm.v1 | 4648 | 313.3 | 9.67 | 0.618 |
| Sealdah Koley Market | naive.v1 | 4648 | 298.5 | 9.11 | 0.59 |

### Held-out, lgbm.v1 vs. naive, by series

| market | crop | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|---|
| Asansol | Onion | lgbm.v1 | 4554 | 96.7 | 4.38 | 0.555 |
| Asansol | Onion | naive.v1 | 4554 | 88.7 | 4.02 | 0.51 |
| Bankura Sadar | Onion | lgbm.v1 | 4199 | 110.4 | 4.73 | 0.618 |
| Bankura Sadar | Onion | naive.v1 | 4199 | 99.1 | 4.24 | 0.562 |
| Barasat | Onion | lgbm.v1 | 4603 | 142.5 | 6.16 | 0.673 |
| Barasat | Onion | naive.v1 | 4603 | 128.3 | 5.56 | 0.609 |
| Barasat | Tomato | lgbm.v1 | 4638 | 301.5 | 11.22 | 0.684 |
| Barasat | Tomato | naive.v1 | 4638 | 286.6 | 10.87 | 0.65 |
| Habra | Tomato | lgbm.v1 | 4662 | 364.7 | 14.27 | 0.673 |
| Habra | Tomato | naive.v1 | 4662 | 361.8 | 14.36 | 0.668 |
| Karsiyang(Matigara) | Tomato | lgbm.v1 | 4697 | 306.1 | 8.22 | 0.629 |
| Karsiyang(Matigara) | Tomato | naive.v1 | 4697 | 294.9 | 7.99 | 0.607 |
| Medinipur(West) | Onion | lgbm.v1 | 4400 | 134.0 | 6.16 | 0.718 |
| Medinipur(West) | Onion | naive.v1 | 4400 | 128.6 | 5.86 | 0.692 |
| Sealdah Koley Market | Tomato | lgbm.v1 | 4648 | 313.3 | 9.67 | 0.618 |
| Sealdah Koley Market | Tomato | naive.v1 | 4648 | 298.5 | 9.11 | 0.59 |

### Held-out, lgbm.v1 vs. naive, by normal vs. volatile target dates

| regime | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| normal | lgbm.v1 | 31717 | 189.0 | 6.66 | 0.526 |
| normal | naive.v1 | 31717 | 175.6 | 6.2 | 0.483 |
| unlabelled | lgbm.v1 | 1623 | 214.0 | 8.04 | 0.71 |
| unlabelled | naive.v1 | 1623 | 203.3 | 7.62 | 0.66 |
| volatile | lgbm.v1 | 3061 | 588.3 | 23.9 | 1.855 |
| volatile | naive.v1 | 3061 | 609.7 | 24.74 | 1.911 |

### Error analysis: series x horizon cells where lgbm.v1 loses to naive (MASE)

lgbm.v1 loses in 54 of 56 cells.

| market | crop | horizon_days | n | naive MASE | lgbm MASE | difference |
|---|---|---|---|---|---|---|
| Barasat | Onion | 7 | 657 | 0.856 | 0.983 | 0.127 |
| Barasat | Onion | 6 | 657 | 0.782 | 0.899 | 0.117 |
| Bankura Sadar | Onion | 7 | 600 | 0.799 | 0.909 | 0.11 |
| Asansol | Onion | 7 | 649 | 0.694 | 0.793 | 0.099 |
| Bankura Sadar | Onion | 6 | 601 | 0.71 | 0.803 | 0.093 |
| Bankura Sadar | Onion | 5 | 601 | 0.649 | 0.731 | 0.082 |
| Barasat | Onion | 5 | 657 | 0.713 | 0.794 | 0.081 |
| Asansol | Onion | 6 | 650 | 0.627 | 0.703 | 0.076 |
| Medinipur(West) | Onion | 7 | 627 | 0.832 | 0.908 | 0.076 |
| Barasat | Tomato | 7 | 662 | 0.909 | 0.971 | 0.062 |
| Barasat | Onion | 4 | 658 | 0.633 | 0.693 | 0.06 |
| Barasat | Tomato | 6 | 662 | 0.839 | 0.896 | 0.057 |
| Bankura Sadar | Onion | 4 | 600 | 0.57 | 0.626 | 0.056 |
| Medinipur(West) | Onion | 6 | 628 | 0.793 | 0.839 | 0.046 |
| Asansol | Onion | 4 | 651 | 0.499 | 0.544 | 0.045 |
| Asansol | Onion | 5 | 651 | 0.573 | 0.618 | 0.045 |
| Sealdah Koley Market | Tomato | 6 | 664 | 0.738 | 0.781 | 0.043 |
| Barasat | Tomato | 5 | 662 | 0.752 | 0.794 | 0.042 |
| Karsiyang(Matigara) | Tomato | 7 | 671 | 0.826 | 0.867 | 0.041 |
| Karsiyang(Matigara) | Tomato | 6 | 671 | 0.757 | 0.798 | 0.041 |
| Sealdah Koley Market | Tomato | 5 | 664 | 0.678 | 0.717 | 0.039 |
| Barasat | Onion | 3 | 658 | 0.523 | 0.561 | 0.038 |
| Sealdah Koley Market | Tomato | 4 | 664 | 0.603 | 0.638 | 0.035 |
| Asansol | Onion | 3 | 651 | 0.445 | 0.478 | 0.033 |
| Sealdah Koley Market | Tomato | 7 | 664 | 0.796 | 0.829 | 0.033 |
| Barasat | Tomato | 4 | 663 | 0.656 | 0.689 | 0.033 |
| Bankura Sadar | Onion | 3 | 599 | 0.475 | 0.508 | 0.033 |
| Karsiyang(Matigara) | Tomato | 5 | 671 | 0.693 | 0.722 | 0.029 |
| Sealdah Koley Market | Tomato | 3 | 664 | 0.532 | 0.56 | 0.028 |
| Karsiyang(Matigara) | Tomato | 4 | 671 | 0.614 | 0.641 | 0.027 |
| Medinipur(West) | Onion | 5 | 629 | 0.753 | 0.776 | 0.023 |
| Barasat | Tomato | 3 | 663 | 0.563 | 0.586 | 0.023 |
| Medinipur(West) | Onion | 4 | 629 | 0.698 | 0.719 | 0.021 |
| Barasat | Onion | 2 | 658 | 0.426 | 0.444 | 0.018 |
| Asansol | Onion | 2 | 651 | 0.401 | 0.416 | 0.015 |
| Bankura Sadar | Onion | 2 | 599 | 0.406 | 0.421 | 0.015 |
| Sealdah Koley Market | Tomato | 2 | 664 | 0.446 | 0.461 | 0.015 |
| Barasat | Tomato | 2 | 663 | 0.467 | 0.48 | 0.013 |
| Habra | Tomato | 3 | 666 | 0.617 | 0.629 | 0.012 |
| Karsiyang(Matigara) | Tomato | 3 | 671 | 0.548 | 0.56 | 0.012 |
| Habra | Tomato | 4 | 666 | 0.672 | 0.681 | 0.009 |
| Medinipur(West) | Onion | 3 | 629 | 0.653 | 0.662 | 0.009 |
| Habra | Tomato | 2 | 666 | 0.534 | 0.542 | 0.008 |
| Medinipur(West) | Onion | 2 | 629 | 0.596 | 0.603 | 0.007 |
| Karsiyang(Matigara) | Tomato | 2 | 671 | 0.455 | 0.461 | 0.006 |
| Barasat | Onion | 1 | 658 | 0.333 | 0.337 | 0.004 |
| Sealdah Koley Market | Tomato | 1 | 664 | 0.335 | 0.339 | 0.004 |
| Barasat | Tomato | 1 | 663 | 0.366 | 0.37 | 0.004 |
| Bankura Sadar | Onion | 1 | 599 | 0.325 | 0.329 | 0.004 |
| Habra | Tomato | 5 | 666 | 0.745 | 0.749 | 0.004 |
| Habra | Tomato | 1 | 666 | 0.418 | 0.421 | 0.003 |
| Karsiyang(Matigara) | Tomato | 1 | 671 | 0.354 | 0.357 | 0.003 |
| Habra | Tomato | 7 | 666 | 0.878 | 0.881 | 0.003 |
| Asansol | Onion | 1 | 651 | 0.329 | 0.331 | 0.002 |

### Error analysis: shock phase

| shock_phase | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| calm | lgbm.v1 | 20439 | 175.9 | 6.07 | 0.497 |
| calm | naive.v1 | 20439 | 161.7 | 5.6 | 0.453 |
| during | lgbm.v1 | 1412 | 415.8 | 18.58 | 1.401 |
| during | naive.v1 | 1412 | 426.8 | 18.93 | 1.436 |
| recovery | lgbm.v1 | 11278 | 212.6 | 7.72 | 0.578 |
| recovery | naive.v1 | 11278 | 200.7 | 7.28 | 0.539 |
| start | lgbm.v1 | 1649 | 736.0 | 28.46 | 2.243 |
| start | naive.v1 | 1649 | 766.3 | 29.72 | 2.317 |
| unlabelled | lgbm.v1 | 1623 | 214.0 | 8.04 | 0.71 |
| unlabelled | naive.v1 | 1623 | 203.3 | 7.62 | 0.66 |

### Error analysis: share of the 28 days before the origin reported

| completeness_28d | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| a: < 70% | lgbm.v1 | 14 | 156.7 | 12.36 | 0.944 |
| a: < 70% | naive.v1 | 14 | 160.0 | 12.75 | 0.964 |
| b: 70-90% | lgbm.v1 | 6531 | 159.1 | 6.1 | 0.619 |
| b: 70-90% | naive.v1 | 6531 | 150.7 | 5.79 | 0.574 |
| c: >= 90% | lgbm.v1 | 29856 | 237.8 | 8.62 | 0.652 |
| c: >= 90% | naive.v1 | 29856 | 227.0 | 8.26 | 0.619 |

### Error analysis: days since the last observation at the origin

| days_since_last_bin | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| a: 1 | lgbm.v1 | 34654 | 223.9 | 8.16 | 0.641 |
| a: 1 | naive.v1 | 34654 | 213.5 | 7.81 | 0.607 |
| b: 2 | lgbm.v1 | 1388 | 210.1 | 7.89 | 0.72 |
| b: 2 | naive.v1 | 1388 | 202.7 | 7.62 | 0.684 |
| c: 3-6 | lgbm.v1 | 359 | 254.8 | 10.07 | 0.863 |
| c: 3-6 | naive.v1 | 359 | 238.4 | 9.34 | 0.76 |

### Error analysis: target price equal to the last known price, or changed (by outcome)

| target_vs_last_price | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|
| changed | lgbm.v1 | 22322 | 339.3 | 12.33 | 0.973 |
| changed | naive.v1 | 22322 | 347.9 | 12.75 | 0.997 |
| unchanged | lgbm.v1 | 14079 | 40.3 | 1.57 | 0.128 |
| unchanged | naive.v1 | 14079 | 0.0 | 0.0 | 0.0 |

By horizon:

| target_vs_last_price | horizon_days | model_version | n | MAE | sMAPE % | MASE |
|---|---|---|---|---|---|---|
| changed | 1 | lgbm.v1 | 2474 | 267.5 | 9.91 | 0.779 |
| changed | 1 | naive.v1 | 2474 | 268.3 | 9.95 | 0.783 |
| changed | 2 | lgbm.v1 | 2825 | 295.8 | 10.84 | 0.85 |
| changed | 2 | naive.v1 | 2825 | 298.7 | 10.96 | 0.859 |
| changed | 3 | lgbm.v1 | 3051 | 321.0 | 11.69 | 0.915 |
| changed | 3 | naive.v1 | 3051 | 325.8 | 11.91 | 0.929 |
| changed | 4 | lgbm.v1 | 3265 | 337.2 | 12.24 | 0.964 |
| changed | 4 | naive.v1 | 3265 | 344.1 | 12.61 | 0.986 |
| changed | 5 | lgbm.v1 | 3458 | 353.4 | 12.83 | 1.011 |
| changed | 5 | naive.v1 | 3458 | 365.2 | 13.39 | 1.045 |
| changed | 6 | lgbm.v1 | 3562 | 374.1 | 13.48 | 1.07 |
| changed | 6 | naive.v1 | 3562 | 387.4 | 14.17 | 1.106 |
| changed | 7 | lgbm.v1 | 3687 | 391.1 | 14.14 | 1.125 |
| changed | 7 | naive.v1 | 3687 | 406.1 | 14.87 | 1.162 |
| unchanged | 1 | lgbm.v1 | 2727 | 3.0 | 0.11 | 0.009 |
| unchanged | 1 | naive.v1 | 2727 | 0.0 | 0.0 | 0.0 |
| unchanged | 2 | lgbm.v1 | 2376 | 12.3 | 0.46 | 0.037 |
| unchanged | 2 | naive.v1 | 2376 | 0.0 | 0.0 | 0.0 |
| unchanged | 3 | lgbm.v1 | 2150 | 25.0 | 1.01 | 0.077 |
| unchanged | 3 | naive.v1 | 2150 | 0.0 | 0.0 | 0.0 |
| unchanged | 4 | lgbm.v1 | 1937 | 41.1 | 1.63 | 0.131 |
| unchanged | 4 | naive.v1 | 1937 | 0.0 | 0.0 | 0.0 |
| unchanged | 5 | lgbm.v1 | 1743 | 61.6 | 2.42 | 0.196 |
| unchanged | 5 | naive.v1 | 1743 | 0.0 | 0.0 | 0.0 |
| unchanged | 6 | lgbm.v1 | 1637 | 82.7 | 3.19 | 0.265 |
| unchanged | 6 | naive.v1 | 1637 | 0.0 | 0.0 | 0.0 |
| unchanged | 7 | lgbm.v1 | 1509 | 101.6 | 3.93 | 0.325 |
| unchanged | 7 | naive.v1 | 1509 | 0.0 | 0.0 | 0.0 |

### Feature importance (share of total gain, lgbm.v1, last fold)

| feature | gain % |
|---|---|
| log_last_price | 13.5 |
| target_month | 8.8 |
| std_28 | 7.7 |
| lag_28 | 6.3 |
| spread_mean_7 | 5.4 |
| series_id | 5.1 |
| std_14 | 4.6 |
| count_28 | 4.6 |
| max_28 | 4.5 |
| spread_last | 3.7 |
| mean_28 | 3.6 |
| min_28 | 3.6 |
| max_14 | 3.4 |
| lag_14 | 3.0 |
| std_7 | 2.6 |
| min_14 | 2.6 |
| mean_14 | 2.4 |
| max_7 | 2.2 |
| lag_7 | 2.1 |
| min_7 | 1.9 |
| mean_7 | 1.5 |
| lag_2 | 1.4 |
| count_14 | 1.3 |
| lag_3 | 1.3 |
| crop | 1.1 |
| market | 0.8 |
| count_7 | 0.7 |
| target_day_of_week | 0.3 |
| days_since_last | 0.0 |
| lag_1 | 0.0 |

### Diebold-Mariano test (held-out, horizon 1, lgbm.v1 vs. naive)

Days: 684. Mean daily |error| difference (lgbm - naive): 1.2 rupees. Newey-West lags: 7. Statistic: 7.59. Two-sided p-value: 0.0000.
