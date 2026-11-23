# Backtest report

Rolling-origin backtest results for every model, all run through the same harness
(`src/mandi/evaluate/backtest.py`). This page covers **backtest** results only. Live results (forecasts made
each morning and scored as actuals arrive) are a separate record and are never mixed into these tables.

Sections: [setup](#setup) · [limitations](#limitations) · [baselines](#baselines-m07) · LightGBM (M08, not yet
built).

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
