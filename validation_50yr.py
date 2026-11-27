#!/usr/bin/env python3
"""Fifty-year blind-era and parameter-robustness validation for strict Gold signals."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Mapping, Tuple

import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

from backtest import (
    ReturnStats,
    align_fred,
    apply_cooldown,
    forward_returns,
    summarize_returns,
)
from regime_predictor import ForwardRegimePredictor, PredictorError, _socket_timeout


FETCH_START = "1975-01-01"
ANALYSIS_START = "1976-06-01"
TIPS_START = pd.Timestamp("2003-01-02")
HORIZONS = (21, 63, 126)
TREND_SMAS = (40, 45, 50, 55, 60)
CONFIRMATION_STREAKS = (2, 3, 4)
YAHOO_TICKERS = ("DX-Y.NYB", "GC=F")
GOLD_SERIES_ID = "GOLDAMGBD228NLBM"
FRED_TICKERS = ("DGS2", "DGS10", "CPIAUCSL", "DFII10")
GOLD_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/unbalancedparentheses/forex-centuries/"
    "main/data/sources/lbma/lbma_gold_daily.csv"
)
ERA_WINDOWS = (
    ("Blind OOS 1976-2006", pd.Timestamp("1976-06-01"), pd.Timestamp("2006-12-31")),
    ("Modern ETF Era 2007+", pd.Timestamp("2007-01-01"), None),
    ("Full 50-Year 1976+", pd.Timestamp(ANALYSIS_START), None),
)


@dataclass(frozen=True)
class RobustnessMatrix:
    """Six-month statistics for every neighboring parameter combination."""

    win_rates: pd.DataFrame
    median_returns: pd.DataFrame
    event_counts: pd.DataFrame
    stability_score: float


def clean_daily(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Clean and forward-fill a daily series without backfilling missing history."""
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    clean.index = pd.to_datetime(clean.index)
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)
    clean.index = clean.index.normalize()
    clean = clean[~clean.index.duplicated(keep="last")]
    union = calendar.union(clean.index).sort_values()
    return clean.reindex(union).ffill().reindex(calendar)


def build_stitched_gold(
    fred_gold: pd.Series,
    yahoo_gold: pd.Series,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Append Yahoo returns to the last FRED anchor without a price-level jump."""
    historical = clean_daily(fred_gold, calendar)
    modern = clean_daily(yahoo_gold, calendar)
    modern_valid = modern.dropna()
    if modern_valid.empty:
        return historical
    historical_valid = historical.dropna()
    if historical_valid.empty:
        raise ValueError("FRED Gold history is required before the Yahoo stitch")

    stitch_date = modern_valid.index[0]
    anchors = historical.loc[:stitch_date].dropna()
    if anchors.empty:
        raise ValueError("No FRED Gold anchor exists at the Yahoo stitch date")
    anchor = float(anchors.iloc[-1])
    stitched = historical.copy()
    stitched.loc[stitch_date] = anchor

    future_dates = calendar[calendar > stitch_date]
    modern_returns = modern.pct_change(fill_method=None).reindex(future_dates)
    if modern_returns.isna().any():
        raise ValueError("Yahoo Gold returns contain gaps after the stitch date")
    stitched.loc[future_dates] = anchor * (1.0 + modern_returns).cumprod()
    return stitched


def build_historical_frame(
    fred_data: Mapping[str, pd.Series],
    yahoo_data: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build stitched Gold, DXY, DGS2, and publication-lagged real yields."""
    gold = build_stitched_gold(
        fred_data["GOLDAMGBD228NLBM"],
        yahoo_data["GC=F"],
        calendar,
    )
    dxy = clean_daily(yahoo_data["DX-Y.NYB"], calendar)
    dgs2 = align_fred(fred_data["DGS2"], calendar, 1)
    dgs10 = align_fred(fred_data["DGS10"], calendar, 1)
    tips = align_fred(fred_data["DFII10"], calendar, 1)

    cpi = pd.to_numeric(fred_data["CPIAUCSL"], errors="coerce").dropna().sort_index()
    cpi_yoy = cpi.pct_change(12, fill_method=None) * 100.0
    lagged_cpi_yoy = align_fred(cpi_yoy, calendar, 20)
    estimated_real_yield = dgs10 - lagged_cpi_yoy
    real_yield = estimated_real_yield.copy()
    real_yield.loc[calendar >= TIPS_START] = tips.loc[calendar >= TIPS_START]

    return pd.DataFrame(
        {
            "gold": gold,
            "dxy": dxy,
            "dgs2": dgs2,
            "real_yield": real_yield,
        },
        index=calendar,
    )


def strict_active_series(
    frame: pd.DataFrame,
    trend_sma: int = 50,
    confirmation_streak: int = 3,
) -> pd.Series:
    """Apply the frozen strict rules with only two explicitly varied parameters."""
    gold = frame["gold"]
    dxy = frame["dxy"]
    dgs2 = frame["dgs2"]
    real_yield = frame["real_yield"]

    gold_sma = gold.rolling(trend_sma).mean()
    dxy_sma = dxy.rolling(20).mean()
    dgs2_sma = dgs2.rolling(50).mean()
    real_sma = real_yield.rolling(50).mean()
    gold_roc = gold.pct_change(10, fill_method=None)
    real_change = real_yield.diff(10)

    real_buffer = real_yield <= real_sma - 0.03
    dollar_buffer = dxy <= dxy_sma * 0.9975
    real_confirmed = (
        real_buffer.rolling(confirmation_streak).sum() == confirmation_streak
    )
    dollar_confirmed = (
        dollar_buffer.rolling(confirmation_streak).sum() == confirmation_streak
    )
    components = pd.DataFrame(
        {
            "real_yields": real_confirmed & (real_change < 0),
            "dollar": dollar_confirmed,
            "fed_pivot": dgs2 < dgs2_sma,
            "gold_technical": (gold > gold_sma) & (gold_roc > 0),
        },
        index=frame.index,
    )
    valid = pd.concat(
        (
            gold_sma,
            gold_roc,
            dxy_sma,
            dgs2_sma,
            real_sma,
            real_change,
            real_confirmed,
            dollar_confirmed,
        ),
        axis=1,
    ).notna().all(axis=1)
    score = components.sum(axis=1)
    return ((score >= 3) & components["gold_technical"] & valid).rename("active")


def strict_event_dates(
    frame: pd.DataFrame,
    trend_sma: int = 50,
    confirmation_streak: int = 3,
) -> pd.DatetimeIndex:
    """Generate strict active rising edges with a 21-business-day cooldown."""
    active = strict_active_series(frame, trend_sma, confirmation_streak)
    crossings = active & ~active.shift(1, fill_value=False)
    crossings.loc[crossings.index < pd.Timestamp(ANALYSIS_START)] = False
    return apply_cooldown(crossings)


def run_robustness_matrix(frame: pd.DataFrame) -> RobustnessMatrix:
    """Evaluate all 15 trend/streak combinations at the 126-day horizon."""
    win_rates = pd.DataFrame(
        index=TREND_SMAS,
        columns=CONFIRMATION_STREAKS,
        dtype=float,
    )
    medians = pd.DataFrame(
        index=TREND_SMAS,
        columns=CONFIRMATION_STREAKS,
        dtype=float,
    )
    counts = pd.DataFrame(
        index=TREND_SMAS,
        columns=CONFIRMATION_STREAKS,
        dtype=int,
    )
    gold = frame["gold"]
    returns = forward_returns(gold, 126)

    for trend_sma in TREND_SMAS:
        for streak in CONFIRMATION_STREAKS:
            events = strict_event_dates(frame, trend_sma, streak)
            stats = summarize_returns(returns.reindex(events).dropna(), gold, 126)
            win_rates.loc[trend_sma, streak] = stats.win_rate
            medians.loc[trend_sma, streak] = stats.median
            counts.loc[trend_sma, streak] = stats.observations

    stability = float((win_rates >= 0.65).mean().mean())
    return RobustnessMatrix(win_rates, medians, counts, stability)


def fetch_gold_history(client: Fred, timeout: float) -> pd.Series:
    """Fetch deleted FRED Gold history or its archived LBMA source fallback."""
    try:
        return client.get_series(GOLD_SERIES_ID, observation_start=FETCH_START)
    except Exception:
        try:
            response = requests.get(GOLD_ARCHIVE_URL, timeout=timeout)
            response.raise_for_status()
            archive = pd.read_csv(StringIO(response.text))
        except (requests.RequestException, ValueError) as exc:
            raise PredictorError("Unable to fetch historical LBMA Gold prices") from exc
    archive["date"] = pd.to_datetime(archive["date"])
    return archive.set_index("date")["gold_pm_usd"].rename(GOLD_SERIES_ID)


class FiftyYearValidation:
    """Download and validate the stitched long-history Gold dataset."""

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key.strip():
            raise PredictorError("FRED_API_KEY is missing")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.frame = pd.DataFrame()

    def fetch_data(self) -> None:
        """Fetch full FRED/Yahoo histories and construct the aligned dataset."""
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        try:
            raw = yf.download(
                tickers=list(YAHOO_TICKERS),
                start=FETCH_START,
                end=end.date().isoformat(),
                auto_adjust=True,
                group_by="column",
                progress=False,
                threads=True,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise PredictorError(f"Long-history Yahoo download failed: {exc}") from exc
        closes = ForwardRegimePredictor._extract_closes(raw, YAHOO_TICKERS)
        calendar = pd.bdate_range(FETCH_START, pd.Timestamp.today().normalize())
        yahoo_data = {
            ticker: clean_daily(closes[ticker], calendar) for ticker in YAHOO_TICKERS
        }

        fred_data: Dict[str, pd.Series] = {}
        client = Fred(api_key=self.api_key)
        try:
            with _socket_timeout(self.timeout):
                fred_data[GOLD_SERIES_ID] = fetch_gold_history(
                    client,
                    self.timeout,
                )
                for ticker in FRED_TICKERS:
                    fred_data[ticker] = client.get_series(
                        ticker,
                        observation_start=FETCH_START,
                    )
        except Exception as exc:
            raise PredictorError(f"Long-history FRED download failed: {exc}") from exc
        self.frame = build_historical_frame(fred_data, yahoo_data, calendar)

    def run_era_validation(
        self,
    ) -> Tuple[Mapping[str, int], Mapping[str, Mapping[int, ReturnStats]]]:
        """Evaluate frozen baseline events by blind, modern, and combined eras."""
        if self.frame.empty:
            raise PredictorError("fetch_data() must complete first")
        events = strict_event_dates(self.frame)
        event_counts: Dict[str, int] = {}
        era_stats: Dict[str, Mapping[int, ReturnStats]] = {}
        gold = self.frame["gold"]
        for era, start, end in ERA_WINDOWS:
            era_events = events[events >= start]
            if end is not None:
                era_events = era_events[era_events <= end]
            event_counts[era] = len(era_events)
            horizon_stats: Dict[int, ReturnStats] = {}
            for horizon in HORIZONS:
                returns = forward_returns(gold, horizon)
                horizon_stats[horizon] = summarize_returns(
                    returns.reindex(era_events).dropna(),
                    gold,
                    horizon,
                )
            era_stats[era] = horizon_stats
        return event_counts, era_stats


def _pct(value: float) -> str:
    """Convert decimal results to display percentages."""
    return "N/A" if pd.isna(value) else f"{value * 100:.2f}%"


def generate_report(
    event_counts: Mapping[str, int],
    era_stats: Mapping[str, Mapping[int, ReturnStats]],
    robustness: RobustnessMatrix,
) -> str:
    """Render blind-era results and the 5x3 anti-curve-fitting matrix."""
    lines = [
        "=" * 100,
        f"STRICT GOLD SURGE — 50-YEAR ANTI-OVERFITTING VALIDATION ({ANALYSIS_START}+)",
        "Return-stitched Gold (archived LBMA fallback); DXY proxy; "
        "CPI lag 20d; daily FRED yields lag 1d.",
        "=" * 100,
    ]
    for horizon in HORIZONS:
        lines.extend(
            (
                "",
                f"BLIND ERA COMPARISON — {horizon} BUSINESS DAYS FORWARD",
                "Era                         Total Events   Valid N   Win Rate  "
                "    Mean    Median   Avg Max DD",
                "-" * 96,
            )
        )
        for era, _, _ in ERA_WINDOWS:
            stats = era_stats[era][horizon]
            lines.append(
                f"{era:<25} {event_counts[era]:>12}  {stats.observations:>8}  "
                f"{_pct(stats.win_rate):>10}  {_pct(stats.mean):>8}  "
                f"{_pct(stats.median):>9}  {_pct(stats.average_max_drawdown):>10}"
            )

    lines.extend(
        (
            "",
            "=" * 100,
            "PARAMETER PERTURBATION — 126D WIN RATE / MEDIAN RETURN / VALID N",
            "Rows: Gold trend SMA | Columns: macro confirmation streak",
            "=" * 100,
        )
    )
    header = "SMA / Streak" + "".join(
        f"{streak:>25}d" for streak in CONFIRMATION_STREAKS
    )
    lines.append(header)
    for trend_sma in TREND_SMAS:
        cells = []
        for streak in CONFIRMATION_STREAKS:
            win = _pct(robustness.win_rates.loc[trend_sma, streak])
            median = _pct(robustness.median_returns.loc[trend_sma, streak])
            count = int(robustness.event_counts.loc[trend_sma, streak])
            cells.append(f"{win} / {median} / N={count}")
        rendered_cells = "  ".join(f"{cell:>24}" for cell in cells)
        lines.append(f"{trend_sma:>10}d  {rendered_cells}")
    lines.extend(
        (
            "-" * 100,
            f"Stability Score (grid cells with >=65% 126d win rate): "
            f"{_pct(robustness.stability_score)}",
            "Returns overlap and are not annualized; eras are assigned by "
            "signal entry date.",
        )
    )
    return "\n".join(lines)


def main() -> int:
    """Run blind-era and robustness validations without changing live signals."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: set FRED_API_KEY before running validation_50yr.py",
            file=sys.stderr,
        )
        return 1
    try:
        validation = FiftyYearValidation(api_key)
        validation.fetch_data()
        counts, era_stats = validation.run_era_validation()
        robustness = run_robustness_matrix(validation.frame)
        print(generate_report(counts, era_stats, robustness))
    except (PredictorError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
