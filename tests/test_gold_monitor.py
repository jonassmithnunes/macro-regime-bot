import pandas as pd
import pytest

from gold_monitor import GoldMonitor, GoldSignal


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
