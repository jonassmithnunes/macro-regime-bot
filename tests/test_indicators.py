import pandas as pd
import pytest

from macro_regime_bot.indicators import (
    InsufficientDataError,
    annualized_growth,
    trailing_average,
)


def monthly_series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-31", periods=len(values), freq="ME"))


def test_three_month_annualized_growth() -> None:
    series = monthly_series([100, 100, 100, 100, 101])
    value, as_of = annualized_growth(series, periods=3, periods_per_year=12)
    assert value == pytest.approx((1.01**4 - 1) * 100)
    assert as_of == series.index[-1]


def test_previous_window_is_non_overlapping() -> None:
    series = monthly_series([100, 100, 100, 100, 101, 102, 103])
    current, _ = annualized_growth(series, 3, 12)
    baseline, _ = annualized_growth(series, 3, 12, offset=3)
    assert current == pytest.approx((103 / 100) ** 4 * 100 - 100)
    assert baseline == 0


def test_trailing_average_baseline_is_thirteen_weeks_back() -> None:
    series = pd.Series(range(30), index=pd.date_range("2024-01-05", periods=30, freq="W-FRI"))
    current, _ = trailing_average(series)
    baseline, _ = trailing_average(series, offset=13)
    assert current == pytest.approx(27.5)
    assert baseline == pytest.approx(14.5)


def test_missing_observations_are_ignored() -> None:
    series = monthly_series([100, float("nan"), 100, 100, 100, 101])
    value, _ = annualized_growth(series, 3, 12)
    assert value == pytest.approx((1.01**4 - 1) * 100)


def test_insufficient_data_has_clear_error() -> None:
    with pytest.raises(InsufficientDataError, match="Need 7 observations"):
        annualized_growth(monthly_series([100, 101]), 3, 12, offset=3)
