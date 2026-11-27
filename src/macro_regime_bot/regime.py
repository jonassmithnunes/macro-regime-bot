"""Two-factor macro regime model and its four-quadrant mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import pandas as pd

from .indicators import annualized_growth, trailing_average


class Regime(str, Enum):
    GOLDILOCKS = "GOLDILOCKS"
    REFLATION = "REFLATION"
    STAGFLATION = "STAGFLATION"
    RECESSION = "RECESSION"


@dataclass(frozen=True)
class Comparison:
    """A current indicator reading and its non-overlapping baseline."""

    current: float
    baseline: float
    current_as_of: pd.Timestamp
    baseline_as_of: pd.Timestamp

    @property
    def change(self) -> float:
        return self.current - self.baseline

    @property
    def rising(self) -> bool:
        return self.current > self.baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": round(self.current, 6),
            "baseline": round(self.baseline, 6),
            "change": round(self.change, 6),
            "direction": "RISING" if self.rising else "FALLING",
            "current_as_of": self.current_as_of.date().isoformat(),
            "baseline_as_of": self.baseline_as_of.date().isoformat(),
        }


@dataclass(frozen=True)
class RegimeAssessment:
    regime: Regime
    growth_expanding: bool
    growth_votes: int
    growth_signals: dict[str, Comparison]
    inflation_rising: bool
    headline_inflation: Comparison
    core_inflation: Comparison
    coincident_series: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "growth": {
                "direction": "EXPANDING" if self.growth_expanding else "SLOWING",
                "accelerating_votes": self.growth_votes,
                "required_votes": 2,
                "signals": {
                    name: comparison.to_dict()
                    for name, comparison in self.growth_signals.items()
                },
            },
            "inflation": {
                "direction": "RISING" if self.inflation_rising else "FALLING",
                "headline_cpi_12m_annualized_pct": self.headline_inflation.to_dict(),
                "core_cpi_12m_annualized_pct": self.core_inflation.to_dict(),
            },
            "coincident_series": self.coincident_series,
        }


def _growth_rate_comparison(
    series: pd.Series, periods: int, periods_per_year: int
) -> Comparison:
    current, current_date = annualized_growth(series, periods, periods_per_year)
    baseline, baseline_date = annualized_growth(
        series, periods, periods_per_year, offset=periods
    )
    return Comparison(current, baseline, current_date, baseline_date)


def _average_comparison(series: pd.Series) -> Comparison:
    current, current_date = trailing_average(series)
    baseline, baseline_date = trailing_average(series, offset=13)
    return Comparison(current, baseline, current_date, baseline_date)


def classify_regime(
    growth_signals: Mapping[str, Comparison],
    headline_inflation: Comparison,
    core_inflation: Comparison,
    coincident_series: str,
) -> RegimeAssessment:
    """Classify growth by majority vote and inflation by headline CPI momentum."""
    required_signals = {"real_gdp", "industrial_production", "coincident_activity"}
    missing = required_signals.difference(growth_signals)
    if missing:
        raise ValueError(f"Missing growth signals: {', '.join(sorted(missing))}")

    growth_votes = sum(growth_signals[name].rising for name in required_signals)
    growth_expanding = growth_votes >= 2
    inflation_rising = headline_inflation.rising

    if growth_expanding and not inflation_rising:
        regime = Regime.GOLDILOCKS
    elif growth_expanding and inflation_rising:
        regime = Regime.REFLATION
    elif not growth_expanding and inflation_rising:
        regime = Regime.STAGFLATION
    else:
        regime = Regime.RECESSION

    return RegimeAssessment(
        regime=regime,
        growth_expanding=growth_expanding,
        growth_votes=growth_votes,
        growth_signals=dict(growth_signals),
        inflation_rising=inflation_rising,
        headline_inflation=headline_inflation,
        core_inflation=core_inflation,
        coincident_series=coincident_series,
    )


def analyze_regime(
    data: Mapping[str, pd.Series], coincident_series: str = "WEI"
) -> RegimeAssessment:
    """Build indicator comparisons from raw FRED observations and classify them."""
    if coincident_series not in {"WEI", "USPHCI"}:
        raise ValueError("coincident_series must be WEI or USPHCI")

    required = {"GDPC1", "INDPRO", "CPIAUCSL", "CPILFESL", coincident_series}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing FRED series: {', '.join(sorted(missing))}")

    coincident = (
        _average_comparison(data[coincident_series])
        if coincident_series == "WEI"
        else _growth_rate_comparison(data[coincident_series], 12, 12)
    )
    growth_signals = {
        "real_gdp": _growth_rate_comparison(data["GDPC1"], 1, 4),
        "industrial_production": _growth_rate_comparison(data["INDPRO"], 12, 12),
        "coincident_activity": coincident,
    }
    headline = _growth_rate_comparison(data["CPIAUCSL"], 12, 12)
    core = _growth_rate_comparison(data["CPILFESL"], 12, 12)
    return classify_regime(growth_signals, headline, core, coincident_series)
