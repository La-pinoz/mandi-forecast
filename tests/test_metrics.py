"""MAE, sMAPE, MASE, the MASE scale and the Diebold-Mariano test (pure functions, no database)."""

import math
from datetime import date

import pandas as pd
import pytest

from mandi.evaluate.metrics import diebold_mariano, mase, mean_absolute_error, seasonal_naive_scale, smape


def test_mean_absolute_error():
    # |100 - 110| = 10, |200 - 180| = 20 -> 15
    assert mean_absolute_error(pd.Series([100.0, 200.0]), pd.Series([110.0, 180.0])) == 15.0


def test_smape_in_percent():
    # Row 1: 200 * 50 / 150 = 66.67; row 2: exact forecast = 0 -> mean 33.33
    value = smape(pd.Series([100.0, 80.0]), pd.Series([50.0, 80.0]))
    assert value == pytest.approx(100 / 3)


def test_mase_divides_each_row_by_its_own_scale():
    # Row 1 (scale 10): 10 / 10 = 1.0; row 2 (scale 40): 20 / 40 = 0.5 -> 0.75
    value = mase(pd.Series([110.0, 200.0]), pd.Series([100.0, 180.0]), pd.Series([10.0, 40.0]))
    assert value == pytest.approx(0.75)


@pytest.mark.parametrize("bad_scale", [float("nan"), 0.0])
def test_mase_refuses_a_missing_or_zero_scale(bad_scale):
    with pytest.raises(ValueError, match="positive MASE scale"):
        mase(pd.Series([110.0, 200.0]), pd.Series([100.0, 180.0]), pd.Series([10.0, bad_scale]))


# January 2024. The 3rd is unreported. Its neighbours (2nd: 110, 4th: 130) are reported but must never
# stand in for it.
SCALE_PRICES = {
    date(2024, 1, 1): 100.0,
    date(2024, 1, 2): 110.0,
    date(2024, 1, 4): 130.0,
    date(2024, 1, 5): 140.0,
    date(2024, 1, 8): 170.0,  # pair with 1st: |170 - 100| = 70
    date(2024, 1, 9): 200.0,  # pair with 2nd: |200 - 110| = 90
    date(2024, 1, 10): 300.0,  # 3rd unreported: no pair (a substitute would give 190 or 170)
    date(2024, 1, 13): 500.0,  # 6th unreported: no pair
}


def test_mase_scale_skips_pairs_with_an_unreported_day():
    # Only the pairs (8th, 1st) and (9th, 2nd) have both days reported: (70 + 90) / 2 = 80
    assert seasonal_naive_scale(SCALE_PRICES, cutoff_date=date(2024, 2, 1)) == 80.0


def test_mase_scale_uses_only_days_before_the_cutoff():
    # Cutoff 9th: the 9th itself is outside training, so only (8th, 1st) is left.
    assert seasonal_naive_scale(SCALE_PRICES, cutoff_date=date(2024, 1, 9)) == 70.0


def test_mase_scale_is_none_without_a_complete_pair():
    assert seasonal_naive_scale(SCALE_PRICES, cutoff_date=date(2024, 1, 8)) is None
    assert seasonal_naive_scale({}, cutoff_date=date(2024, 1, 8)) is None


def test_diebold_mariano_by_hand_without_and_with_newey_west_lags():
    # d = 1, 2, 3, 4: mean 2.5, autocovariance at lag 0 = 1.25, at lag 1 = 0.3125 (sums divided by 4).
    differences = pd.Series([1.0, 2.0, 3.0, 4.0])

    statistic, p_value = diebold_mariano(differences, max_lag=0)
    assert statistic == pytest.approx(2.5 / math.sqrt(1.25 / 4))  # 4.472
    assert p_value == pytest.approx(math.erfc(statistic / math.sqrt(2)))

    # Lag 1 with weight 1 - 1/2: variance (1.25 + 2 * 0.5 * 0.3125) / 4 = 0.390625, statistic 2.5 / 0.625.
    statistic, _ = diebold_mariano(differences, max_lag=1)
    assert statistic == pytest.approx(4.0)


def test_diebold_mariano_refuses_constant_differences():
    with pytest.raises(ValueError, match="no variance"):
        diebold_mariano(pd.Series([2.0, 2.0, 2.0]), max_lag=1)
