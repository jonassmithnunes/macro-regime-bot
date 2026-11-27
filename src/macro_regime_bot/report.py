"""Assemble a JSON-serializable macro regime report."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .config import COINCIDENT_SERIES, FRED_SERIES
from .indicators import clean_series
from .regime import RegimeAssessment


def _latest(series: pd.Series) -> dict[str, Any]:
    clean = clean_series(series)
    if clean.empty:
        return {"value": None, "as_of": None}
    return {
        "value": round(float(clean.iloc[-1]), 6),
        "as_of": pd.Timestamp(clean.index[-1]).date().isoformat(),
    }


def build_report(
    data: Mapping[str, pd.Series],
    assessment: RegimeAssessment,
    markets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for category, series_metadata in FRED_SERIES.items():
        observations[category] = {
            series_id: {"name": name, **_latest(data[series_id])}
            for series_id, name in series_metadata.items()
        }

    coincident_id = assessment.coincident_series
    observations["coincident_activity"] = {
        coincident_id: {
            "name": COINCIDENT_SERIES[coincident_id],
            **_latest(data[coincident_id]),
        }
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **assessment.to_dict(),
        "observations": observations,
        "markets": dict(markets or {}),
        "methodology": {
            "growth": (
                "Expanding when at least two of real GDP quarter-over-quarter "
                "annualized momentum, industrial production 12-month annualized "
                "momentum, and the coincident activity proxy are accelerating."
            ),
            "inflation": (
                "Rising when the latest headline CPI 12-month annualized rate "
                "exceeds the preceding, non-overlapping 12-month annualized rate."
            ),
            "market_prices": "Latest available daily Yahoo Finance closes; not tick-by-tick quotes.",
        },
    }
