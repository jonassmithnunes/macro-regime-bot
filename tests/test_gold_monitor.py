import pandas as pd
import pytest

from gold_monitor import (
    GoldMonitor,
    GoldSignal,
    dollar_progress,
    front_end_progress,
    gold_trend_progress,
    real_yield_progress,
    trailing_streak,
)


SIGNAL_NAMES = ("real_yields", "dollar", "fed_pivot", "gold_technical")


def signals_with_active(count: int) -> dict[str, GoldSignal]:
    return {
        name: GoldSignal(
            active=index < count,
            current=100.0,
            sma=99.0,
            change_10d=1.0,
        )
        for index, name in enumerate(SIGNAL_NAMES)
    }


@pytest.mark.parametrize(
    ("score", "readiness"),
    [
        (0, "WAIT / FALLING KNIFE"),
        (1, "WAIT / FALLING KNIFE"),
        (2, "EARLY WATCH / STABILIZING"),
        (3, "SURGE TRIGGER ACTIVE"),
        (4, "SURGE TRIGGER ACTIVE"),
    ],
)
def test_gold_scoring_from_zero_to_four(score: int, readiness: str) -> None:
    assessment = GoldMonitor.build_assessment(
        signals_with_active(score),
        gold_silver_20d=2.5,
        as_of=pd.Timestamp("2026-01-30"),
    )
    assert assessment.score == score
    assert assessment.readiness == readiness
    assert assessment.gold_silver_20d == 2.5


def test_gold_scoring_requires_exactly_four_catalysts() -> None:
    with pytest.raises(ValueError, match="Exactly four"):
        GoldMonitor.build_assessment({}, 0.0, pd.Timestamp("2026-01-30"))


@pytest.mark.parametrize("score", [-1, 5])
def test_gold_score_rejects_out_of_range_values(score: int) -> None:
    with pytest.raises(ValueError, match="between zero and four"):
        GoldMonitor.classify_score(score)


def test_real_yield_progress_covers_active_pending_and_inactive_states() -> None:
    assert real_yield_progress(1.90, 2.0, -0.10, 3) == 1.0
    assert real_yield_progress(1.97, 2.0, -0.10, 1) == pytest.approx(0.94)
    assert real_yield_progress(1.97, 2.0, -0.10, 2) == pytest.approx(0.98)
    assert real_yield_progress(2.37, 2.0, 0.10, 0) == 0.0
    assert real_yield_progress(1.97, 2.0, 0.10, 0) == pytest.approx(0.89)


def test_dollar_progress_covers_active_pending_and_inactive_states() -> None:
    target = 30.0 * 0.9975
    assert dollar_progress(target, 30.0, 3) == 1.0
    assert dollar_progress(target, 30.0, 1) == pytest.approx(0.94)
    assert dollar_progress(target, 30.0, 2) == pytest.approx(0.98)
    assert dollar_progress(target * 1.03, 30.0, 0) == 0.0
    assert dollar_progress(target, 30.0, 0) == pytest.approx(0.89)


def test_yield_and_gold_progress_cover_linear_and_special_states() -> None:
    values = [
        front_end_progress(4.0, 4.5),
        front_end_progress(5.0, 4.5),
        front_end_progress(4.5, 4.5),
        gold_trend_progress(110.0, 100.0, 1.0),
        gold_trend_progress(110.0, 100.0, 0.0),
        gold_trend_progress(85.0, 100.0, -1.0),
        gold_trend_progress(100.0, 100.0, 0.0),
    ]
    assert values[0] == 1.0
    assert values[1] == 0.0
    assert values[2] == pytest.approx(0.95)
    assert values[3] == 1.0
    assert values[4] == pytest.approx(0.92)
    assert values[5] == 0.0
    assert values[6] == pytest.approx(0.90)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_trailing_streak_and_progress_midpoints() -> None:
    condition = pd.Series([True, False, True, True, True, True])
    assert trailing_streak(condition) == 3
    assert trailing_streak(condition.iloc[:4]) == 2
    assert real_yield_progress(2.17, 2.0, 0.0, 0) == pytest.approx(0.445)
    assert dollar_progress(31.0, 30.0, 0) == pytest.approx(0.0)
    assert gold_trend_progress(92.5, 100.0, 0.0) == pytest.approx(0.45)
