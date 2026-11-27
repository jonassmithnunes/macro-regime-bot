import pandas as pd
import pytest

from validation_50yr import (
    CONFIRMATION_STREAKS,
    TREND_SMAS,
    build_stitched_gold,
    run_robustness_matrix,
)


def test_stitched_gold_uses_yahoo_returns_without_level_jump() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=5)
    fred_gold = pd.Series([100.0, 105.0, 110.0], index=calendar[:3])
    yahoo_gold = pd.Series([200.0, 201.0, 202.0], index=calendar[2:])

    stitched = build_stitched_gold(fred_gold, yahoo_gold, calendar)

    assert stitched.iloc[2] == 110.0
    assert stitched.iloc[3] == pytest.approx(110.0 * 201.0 / 200.0)
    assert stitched.iloc[4] == pytest.approx(110.0 * 202.0 / 200.0)
    assert stitched.iloc[3] / stitched.iloc[2] == pytest.approx(201.0 / 200.0)


def test_perturbation_matrix_has_all_cells_and_stable_plateau() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=300)
    frame = pd.DataFrame(
        {
            "gold": pd.Series(range(1, 301), index=calendar, dtype=float),
            "dxy": pd.Series(range(300, 0, -1), index=calendar, dtype=float),
            "dgs2": pd.Series(range(300, 0, -1), index=calendar, dtype=float) / 100,
            "real_yield": (
                pd.Series(range(300, 0, -1), index=calendar, dtype=float) / 100
            ),
        }
    )

    result = run_robustness_matrix(frame)

    assert result.win_rates.shape == (len(TREND_SMAS), len(CONFIRMATION_STREAKS))
    assert result.median_returns.shape == result.win_rates.shape
    assert result.event_counts.shape == result.win_rates.shape
    assert (result.event_counts == 1).all().all()
    assert (result.win_rates == 1.0).all().all()
    assert (result.median_returns > 0).all().all()
    assert result.stability_score == 1.0
