from datetime import datetime

import pandas as pd
import pytest

from macro_regime_bot.regime import Comparison, Regime, analyze_regime, classify_regime


DATE = pd.Timestamp("2025-01-01")
OLD_DATE = pd.Timestamp("2024-01-01")


def comparison(current: float, baseline: float) -> Comparison:
    return Comparison(current, baseline, DATE, OLD_DATE)


def assessment(growth: tuple[float, float, float], inflation: float):
    signals = {
        "real_gdp": comparison(growth[0], 0),
        "industrial_production": comparison(growth[1], 0),
        "coincident_activity": comparison(growth[2], 0),
    }
    return classify_regime(
        signals,
        comparison(inflation, 0),
        comparison(inflation, 0),
        "WEI",
    )


@pytest.mark.parametrize(
    ("growth", "inflation", "expected"),
    [
        ((1, 1, -1), -1, Regime.GOLDILOCKS),
        ((1, 1, -1), 1, Regime.REFLATION),
        ((-1, -1, 1), 1, Regime.STAGFLATION),
        ((-1, -1, 1), -1, Regime.RECESSION),
    ],
)
def test_four_quadrants(growth, inflation, expected) -> None:
    assert assessment(growth, inflation).regime is expected


def test_growth_requires_two_accelerating_signals() -> None:
    result = assessment((1, -1, -1), -1)
    assert result.growth_votes == 1
    assert not result.growth_expanding
    assert result.regime is Regime.RECESSION


def monthly(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2023-01-31", periods=len(values), freq="ME"))


def full_data() -> dict[str, pd.Series]:
    return {
        # Both current and previous real-GDP quarterly rates, accelerating.
        "GDPC1": pd.Series(
            [100, 101, 103], index=pd.date_range("2024-03-31", periods=3, freq="QE")
        ),
        # Flat 12-month baseline windows followed by growth in the current year.
        "INDPRO": monthly([100] * 13 + [100 + i for i in range(1, 13)]),
        "CPIAUCSL": monthly([100] * 13 + [100 + i for i in range(1, 13)]),
        "CPILFESL": monthly([100] * 13 + [100 + 0.5 * i for i in range(1, 13)]),
        # Latest four-week average exceeds its value thirteen weeks earlier.
        "WEI": pd.Series(
            list(range(30)), index=pd.date_range("2024-01-05", periods=30, freq="W-FRI")
        ),
    }


def test_analyze_regime_with_wei() -> None:
    result = analyze_regime(full_data())
    assert result.regime is Regime.REFLATION
    assert result.growth_votes == 3
    assert result.inflation_rising
    assert result.headline_inflation.current == pytest.approx(12)
    assert result.headline_inflation.baseline == 0


def test_analyze_regime_with_usphci() -> None:
    data = full_data()
    data["USPHCI"] = data["INDPRO"]
    result = analyze_regime(data, coincident_series="USPHCI")
    assert result.regime is Regime.REFLATION
    assert result.coincident_series == "USPHCI"


def test_analyze_regime_rejects_unknown_proxy() -> None:
    with pytest.raises(ValueError, match="WEI or USPHCI"):
        analyze_regime(full_data(), coincident_series="UNKNOWN")


def test_comparison_serializes_dates_and_direction() -> None:
    result = comparison(2.5, 1.5).to_dict()
    assert result["direction"] == "RISING"
    assert result["change"] == 1
    assert result["current_as_of"] == DATE.date().isoformat()
    assert datetime.fromisoformat(result["baseline_as_of"])
