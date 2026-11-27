"""Pure time-series calculations used by the regime classifier."""

from __future__ import annotations

import pandas as pd


class InsufficientDataError(ValueError):
    """Raised when an indicator does not have enough valid observations."""


def clean_series(series: pd.Series) -> pd.Series:
    """Convert observations to numeric, remove missing values, and sort by date."""
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    return clean[~clean.index.duplicated(keep="last")]


def annualized_growth(
    series: pd.Series,
    periods: int,
    periods_per_year: int,
    offset: int = 0,
) -> tuple[float, pd.Timestamp]:
    """Return compounded growth and its end date, optionally shifted backward."""
    clean = clean_series(series)
    end = len(clean) - 1 - offset
    start = end - periods
    if periods <= 0 or offset < 0 or start < 0 or end >= len(clean):
        raise InsufficientDataError(
            f"Need {periods + offset + 1} observations; got {len(clean)}"
        )

    start_value = float(clean.iloc[start])
    end_value = float(clean.iloc[end])
    if start_value <= 0 or end_value <= 0:
        raise ValueError("Annualized growth requires positive index values")

    value = ((end_value / start_value) ** (periods_per_year / periods) - 1) * 100
    return value, pd.Timestamp(clean.index[end])


def trailing_average(
    series: pd.Series,
    window: int = 4,
    offset: int = 0,
) -> tuple[float, pd.Timestamp]:
    """Return a trailing average and its end date, optionally shifted backward."""
    clean = clean_series(series)
    end = len(clean) - offset
    start = end - window
    if window <= 0 or offset < 0 or start < 0 or end > len(clean):
        raise InsufficientDataError(
            f"Need at least {window + offset} observations; got {len(clean)}"
        )

    average = float(clean.iloc[start:end].mean())
    return average, pd.Timestamp(clean.index[end - 1])
