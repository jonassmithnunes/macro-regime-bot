#!/usr/bin/env python3
"""Gold macro catalyst and technical inflection monitor."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Mapping

import pandas as pd
from fredapi import Fred

from regime_predictor import (
    PredictorError,
    _business_daily,
    _ratio,
    _socket_timeout,
    download_market_data,
)


GOLD_MARKET_TICKERS = ("UUP", "GLD", "SLV")
GOLD_FRED_TICKERS = ("DFII10", "DGS2")


@dataclass(frozen=True)
class GoldSignal:
    """One Gold catalyst with its current, SMA, and ten-day values."""

    active: bool
    current: float
    sma: float
    change_10d: float


@dataclass(frozen=True)
class GoldAssessment:
    """Gold catalyst score, readiness classification, and leadership metric."""

    signals: Mapping[str, GoldSignal]
    score: int
    readiness: str
    gold_silver_20d: float
    as_of: pd.Timestamp


class GoldMonitor:
    """Evaluate four macro and technical conditions for a Gold inflection."""

    def __init__(self, api_key: str, years: int = 3, timeout: float = 30.0) -> None:
        if not api_key.strip():
            raise PredictorError("FRED_API_KEY is missing")
        self.api_key = api_key.strip()
        self.years = years
        self.timeout = timeout
        self.markets: Dict[str, pd.Series] = {}
        self.fred: Dict[str, pd.Series] = {}

    def fetch_data(self) -> None:
        """Fetch GLD, SLV, UUP, real yields, and two-year yields."""
        self.markets = download_market_data(
            GOLD_MARKET_TICKERS,
            years=self.years,
            timeout=self.timeout,
        )
        start = pd.Timestamp.today().normalize() - pd.DateOffset(years=self.years)
        client = Fred(api_key=self.api_key)
        try:
            with _socket_timeout(self.timeout):
                for ticker in GOLD_FRED_TICKERS:
                    raw = client.get_series(
                        ticker,
                        observation_start=start.date().isoformat(),
                    )
                    self.fred[ticker] = _business_daily(
                        raw,
                        end=pd.Timestamp.today(),
                    ).rename(ticker)
        except Exception as exc:
            raise PredictorError(f"Gold FRED download failed: {exc}") from exc

    @staticmethod
    def _signal(
        series: pd.Series,
        sma_window: int,
        percent_change: bool,
    ) -> GoldSignal:
        """Calculate a current/SMA pair and ten-observation change."""
        clean = series.dropna().sort_index()
        required = max(sma_window, 11)
        if len(clean) < required:
            raise PredictorError("Insufficient history for Gold signal calculation")
        current = float(clean.iloc[-1])
        sma = float(clean.rolling(sma_window).mean().iloc[-1])
        past = float(clean.iloc[-11])
        if percent_change:
            if past == 0:
                raise PredictorError("Gold signal has a zero price baseline")
            change = (current / past - 1.0) * 100.0
        else:
            change = current - past
        return GoldSignal(False, current, sma, change)

    @staticmethod
    def classify_score(score: int) -> str:
        """Map a zero-to-four catalyst score to the required readiness label."""
        if score < 0 or score > 4:
            raise ValueError("Gold score must be between zero and four")
        if score <= 1:
            return "WAIT / FALLING KNIFE"
        if score == 2:
            return "EARLY WATCH / STABILIZING"
        return "SURGE TRIGGER ACTIVE"

    @classmethod
    def build_assessment(
        cls,
        signals: Mapping[str, GoldSignal],
        gold_silver_20d: float,
        as_of: pd.Timestamp,
    ) -> GoldAssessment:
        """Count active catalysts and construct an immutable assessment."""
        if len(signals) != 4:
            raise ValueError("Exactly four Gold signals are required")
        score = sum(signal.active for signal in signals.values())
        return GoldAssessment(
            signals=dict(signals),
            score=score,
            readiness=cls.classify_score(score),
            gold_silver_20d=gold_silver_20d,
            as_of=as_of,
        )

    def evaluate(self) -> GoldAssessment:
        """Apply all four Gold trigger rules to the downloaded observations."""
        if not self.markets or not self.fred:
            raise PredictorError("fetch_data() must complete before evaluate()")

        real_yield = self._signal(self.fred["DFII10"], 50, False)
        dollar = self._signal(self.markets["UUP"], 20, True)
        front_end = self._signal(self.fred["DGS2"], 50, False)
        gold = self._signal(self.markets["GLD"], 50, True)
        signals = {
            "real_yields": GoldSignal(
                real_yield.current < real_yield.sma and real_yield.change_10d < 0,
                real_yield.current,
                real_yield.sma,
                real_yield.change_10d,
            ),
            "dollar": GoldSignal(
                dollar.current < dollar.sma,
                dollar.current,
                dollar.sma,
                dollar.change_10d,
            ),
            "fed_pivot": GoldSignal(
                front_end.current < front_end.sma,
                front_end.current,
                front_end.sma,
                front_end.change_10d,
            ),
            "gold_technical": GoldSignal(
                gold.current > gold.sma and gold.change_10d > 0,
                gold.current,
                gold.sma,
                gold.change_10d,
            ),
        }

        ratio = _ratio(self.markets["GLD"], self.markets["SLV"])
        if len(ratio) < 21 or ratio.iloc[-21] == 0:
            raise PredictorError("Insufficient GLD/SLV ratio history")
        gold_silver_20d = float((ratio.iloc[-1] / ratio.iloc[-21] - 1.0) * 100.0)
        return self.build_assessment(
            signals,
            gold_silver_20d,
            self.markets["GLD"].index[-1],
        )

    @staticmethod
    def generate_report(assessment: GoldAssessment) -> str:
        """Render the four Gold triggers and precious-metals leadership metric."""
        real_yield = assessment.signals["real_yields"]
        dollar = assessment.signals["dollar"]
        front_end = assessment.signals["fed_pivot"]
        gold = assessment.signals["gold_technical"]

        def flag(signal: GoldSignal) -> str:
            return "ACTIVE" if signal.active else "INACTIVE"

        lines = [
            "GOLD BOUNCE & INFLECTION MONITOR",
            f"Gold Readiness: {assessment.readiness} "
            f"({assessment.score}/4 catalysts active)",
            f"  - Real Yields Rolling Over (DFII10): {flag(real_yield)} | "
            f"Current: {real_yield.current:.2f}% | 50D SMA: {real_yield.sma:.2f}% "
            f"| 10D: {real_yield.change_10d:+.2f} pp",
            f"  - US Dollar Breakdown (UUP): {flag(dollar)} | "
            f"Current: ${dollar.current:.2f} | 20D SMA: ${dollar.sma:.2f}",
            f"  - Fed Pivot Pricing (DGS2): {flag(front_end)} | "
            f"Current: {front_end.current:.2f}% | 50D SMA: {front_end.sma:.2f}%",
            f"  - Gold Trend Reclaim (GLD): {flag(gold)} | "
            f"Current: ${gold.current:.2f} | 50D SMA: ${gold.sma:.2f} | "
            f"10D ROC: {gold.change_10d:+.2f}%",
            f"  - GLD/SLV Leadership Ratio: 20D {assessment.gold_silver_20d:+.2f}%",
            f"  - Market data through: {assessment.as_of.date().isoformat()}",
        ]
        return "\n".join(lines)


def main() -> int:
    """Run the standalone live Gold monitor."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print("Error: set FRED_API_KEY before running gold_monitor.py", file=sys.stderr)
        return 1
    try:
        monitor = GoldMonitor(api_key)
        monitor.fetch_data()
        print(monitor.generate_report(monitor.evaluate()))
    except (PredictorError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
