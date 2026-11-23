"""Anomaly flags (validate/anomalies.py) and day status rows (validate/day_status.py); no database needed."""

import pandas as pd

from mandi.validate.anomalies import flag_anomalies
from mandi.validate.day_status import build_day_status

START = pd.Timestamp("2026-01-01")


def series_prices(series_id: int, name: str, commodity: str, prices: dict[int, float]) -> pd.DataFrame:
    """Rows for one series; `prices` maps day number (0 = START) to modal price. Missing days = unreported."""
    return pd.DataFrame(
        {
            "series_id": series_id,
            "series_name": name,
            "commodity": commodity,
            "arrival_date": [START + pd.Timedelta(days=day) for day in prices],
            "modal_price": list(prices.values()),
        }
    )


def flat_with(changes: dict[int, float], days: int = 20, level: float = 2000.0) -> dict[int, float]:
    prices = {day: level for day in range(days)}
    prices.update(changes)
    return prices


def flags_by_day(flagged: pd.DataFrame, series_id: int = 1) -> dict[int, bool]:
    one = flagged[flagged["series_id"] == series_id]
    return {(row.arrival_date - START).days: row.is_anomaly for row in one.itertuples()}


def test_a_spike_is_flagged_and_kept():
    prices = series_prices(1, "Barasat Tomato", "Tomato", flat_with({15: 4000.0}))

    flagged = flag_anomalies(prices)

    assert len(flagged) == len(prices)  # nothing removed
    assert [day for day, flag in flags_by_day(flagged).items() if flag] == [15]
    spike = flagged[flagged["is_anomaly"]].iloc[0]
    assert spike["modal_price"] == 4000.0
    assert spike["anomaly_reason"] == (
        "isolated move: +100% vs 14-day median 2000; no other tracked Tomato market flagged within 1 day"
    )


def test_a_small_step_after_flat_prices_is_not_flagged():
    # Flat prices give MAD = 0; without the minimum scale any change at all would be flagged.
    prices = series_prices(1, "Barasat Onion", "Onion", flat_with({15: 2400.0}))  # +20%

    assert not flag_anomalies(prices)["is_anomaly"].any()


def test_a_drop_is_flagged_too():
    prices = series_prices(1, "Barasat Onion", "Onion", flat_with({15: 1200.0}))  # -40%

    assert flags_by_day(flag_anomalies(prices))[15]


def test_too_little_history_is_not_judged():
    # Only 4 earlier observations in the 14-day window (minimum is 5).
    prices = series_prices(
        1, "Barasat Tomato", "Tomato", {0: 2000.0, 1: 2000.0, 2: 2000.0, 3: 2000.0, 4: 9000.0}
    )

    assert not flag_anomalies(prices)["is_anomaly"].any()


def test_the_window_is_calendar_days_not_observations():
    # Six observations, but the first five are more than 14 calendar days before day 30.
    prices = series_prices(
        1, "Barasat Tomato", "Tomato", {0: 2000.0, 1: 2000.0, 2: 2000.0, 3: 2000.0, 4: 2000.0, 30: 9000.0}
    )

    assert not flag_anomalies(prices)["is_anomaly"].any()


def test_a_flag_uses_only_earlier_days():
    # Adding later prices (even a huge jump right after) must not change the flag on day 15: no look-ahead.
    before = flag_anomalies(series_prices(1, "Barasat Tomato", "Tomato", flat_with({15: 4000.0}, days=16)))
    after = flag_anomalies(series_prices(1, "Barasat Tomato", "Tomato", flat_with({15: 4000.0, 16: 20000.0})))

    assert flags_by_day(before) == {day: flag for day, flag in flags_by_day(after).items() if day <= 15}


def test_same_direction_move_in_another_tracked_series_of_the_crop_is_shared():
    prices = pd.concat(
        [
            series_prices(1, "Barasat Tomato", "Tomato", flat_with({15: 4000.0})),
            series_prices(2, "Habra Tomato", "Tomato", flat_with({16: 3800.0})),  # one day later: counts
            series_prices(3, "Sealdah Koley Market Tomato", "Tomato", flat_with({15: 900.0})),  # opposite way
            series_prices(4, "Barasat Onion", "Onion", flat_with({15: 4000.0})),  # other crop
        ]
    )

    flagged = flag_anomalies(prices).set_index("series_id")

    assert flagged.loc[1].query("is_anomaly")["anomaly_reason"].item() == (
        "shared move: +100% vs 14-day median 2000; also flagged: Habra Tomato"
    )
    assert flagged.loc[3].query("is_anomaly")["anomaly_reason"].item().startswith("isolated move: -55%")
    assert flagged.loc[4].query("is_anomaly")["anomaly_reason"].item().startswith("isolated move")


def test_moves_two_days_apart_are_not_shared():
    prices = pd.concat(
        [
            series_prices(1, "Barasat Tomato", "Tomato", flat_with({15: 4000.0})),
            series_prices(2, "Habra Tomato", "Tomato", flat_with({17: 4000.0})),
        ]
    )

    reasons = flag_anomalies(prices).query("is_anomaly")["anomaly_reason"]

    assert all(reason.startswith("isolated move") for reason in reasons)


def test_missing_days_are_unreported_and_never_dropped_or_filled():
    barasat = series_prices(1, "Barasat Tomato", "Tomato", {0: 2000.0, 1: 2100.0, 3: 2200.0})
    # Starts later and reports the last date: the end date for every series.
    habra = series_prices(2, "Habra Tomato", "Tomato", {2: 1800.0, 5: 1900.0})

    status = build_day_status(pd.concat([barasat, habra]))

    rows = {(row.series_id, (row.arrival_date - START).days): row.status for row in status.itertuples()}
    assert rows == {
        (1, 0): "reported",
        (1, 1): "reported",
        (1, 2): "unreported",
        (1, 3): "reported",
        (1, 4): "unreported",
        (1, 5): "unreported",
        # A series' rows start at its first reported date.
        (2, 2): "reported",
        (2, 3): "unreported",
        (2, 4): "unreported",
        (2, 5): "reported",
    }
    assert "no_transaction" not in set(status["status"])  # the source has no NIL marker (M01)
    assert not status["is_anomaly"].any()


def test_day_status_carries_the_anomaly_flag():
    status = build_day_status(series_prices(1, "Barasat Tomato", "Tomato", flat_with({15: 4000.0})))

    spike = status[status["is_anomaly"]]
    assert list(spike["arrival_date"]) == [START + pd.Timedelta(days=15)]
    assert list(spike["status"]) == ["reported"]
    assert status.loc[~status["is_anomaly"], "anomaly_reason"].isna().all()
