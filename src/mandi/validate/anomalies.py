"""Statistical anomaly flags for modal prices (blueprint 5.4): flag and keep, never remove.

Method (a trailing Hampel filter on log prices), for each series and each reported day t:
1. Window = the series' reported modal prices in the 14 calendar days before t (t itself excluded). At
   least 5 are needed; otherwise t is not judged (no flag).
2. deviation = log(price_t) - median(log prices in the window). Logs make it a relative move, so a
   tomato at 4,000 and an onion at 1,500 are judged on the same scale.
3. scale = 1.4826 * MAD of the window. MAD = median absolute deviation from the median; times 1.4826 it
   estimates the standard deviation for normal data, but one wild value can't inflate it.
4. Flag when |deviation| > 3.5 * max(scale, 0.10).

Why the floor of 0.10: these prices are often flat for days and then jump, so MAD is 0 in about half of
all 14-day windows (checked on the 8 series, 2026-11-23); without a floor every small step would be
flagged (~30% of days). With it, the smallest move that can be flagged is 0.35 in log terms (about
+42% / -30% vs. the window median). Settings were fixed from flag counts on the history, never from
model results.

Why trailing rather than centred: a flag uses only data available on day t, so the daily job can set it
the same day and later days never change an earlier flag (unless the source revises a price). The cost: a
lasting level shift is flagged for several days until the window catches up.

Shared move: a flagged value is "shared" if another tracked series of the same crop is flagged in the
same direction within +-1 day (owner decision: the other tracked series of the crop are the neighbours).
A shared move is more likely a real market shock than a bad entry.
"""

import numpy as np
import pandas as pd

WINDOW = "14D"
MIN_OBSERVATIONS = 5
THRESHOLD = 3.5
MIN_SCALE = 0.10
MAD_TO_STD = 1.4826
NEIGHBOUR_DAYS = 1


def median_absolute_deviation(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values))))


def hampel_flags(one_series: pd.DataFrame) -> pd.DataFrame:
    """Add window_median_price, deviation and is_anomaly to the reported days of one series.

    Input columns: arrival_date (datetime64, one row per date), modal_price (float).
    """
    result = one_series.sort_values("arrival_date").reset_index(drop=True)
    log_price = pd.Series(np.log(result["modal_price"].to_numpy()), index=result["arrival_date"])

    # closed="left": the window is [t - 14 days, t), so day t never judges itself.
    window = log_price.rolling(WINDOW, closed="left", min_periods=MIN_OBSERVATIONS)
    window_median = window.median()
    window_mad = window.apply(median_absolute_deviation, raw=True)

    deviation = log_price - window_median
    limit = THRESHOLD * np.maximum(MAD_TO_STD * window_mad, MIN_SCALE)
    # A day with too little history has NaN deviation; NaN > limit is False, so it isn't flagged.
    is_anomaly = deviation.abs() > limit

    result["window_median_price"] = np.exp(window_median.to_numpy())
    result["deviation"] = deviation.to_numpy()
    result["is_anomaly"] = is_anomaly.to_numpy()
    return result


def anomaly_reason(
    deviation: float, window_median_price: float, commodity: str, shared_with: list[str]
) -> str:
    """e.g. 'shared move: +52% vs 14-day median 2800; also flagged: Habra Tomato'."""
    change = np.exp(deviation) - 1
    text = f"{change:+.0%} vs 14-day median {window_median_price:.0f}"
    if shared_with:
        return f"shared move: {text}; also flagged: {', '.join(shared_with)}"
    return f"isolated move: {text}; no other tracked {commodity} market flagged within 1 day"


def flag_anomalies(prices: pd.DataFrame) -> pd.DataFrame:
    """Flag unusual modal prices. One row per reported series-day in and out.

    Input columns: series_id, series_name, commodity, arrival_date (datetime64), modal_price (float).
    Adds window_median_price, deviation, is_anomaly and anomaly_reason (None when not flagged).
    """
    parts = []
    for _series_id, one_series in prices.groupby("series_id"):
        parts.append(hampel_flags(one_series))
    flagged = pd.concat(parts, ignore_index=True)

    anomalies = flagged[flagged["is_anomaly"]]
    reasons = []
    for row in flagged.itertuples(index=False):
        if not row.is_anomaly:
            reasons.append(None)
            continue
        same_move = anomalies[
            (anomalies["commodity"] == row.commodity)
            & (anomalies["series_id"] != row.series_id)
            & ((anomalies["arrival_date"] - row.arrival_date).abs() <= pd.Timedelta(days=NEIGHBOUR_DAYS))
            & (np.sign(anomalies["deviation"]) == np.sign(row.deviation))
        ]
        shared_with = sorted(set(same_move["series_name"]))
        reasons.append(anomaly_reason(row.deviation, row.window_median_price, row.commodity, shared_with))
    flagged["anomaly_reason"] = reasons
    return flagged
