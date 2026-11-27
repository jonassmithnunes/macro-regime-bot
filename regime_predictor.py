#!/usr/bin/env python3
"""Forecast the six-month-forward US macro regime from leading indicators."""

from __future__ import annotations

import os
import socket
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred


MARKET_TICKERS = (
    "HG=F",
    "GC=F",
    "HYG",
    "IEF",
    "CL=F",
    "SPY",
    "BTC-USD",
    "DX-Y.NYB",
)
FRED_TICKERS = ("PERMIT", "IC4WSA", "T5YIE", "T10Y2Y")
LOOKBACK_3M = 63
SMA_WINDOW = 50
FRED_6M_WINDOW = 126
BREAKEVEN_THRESHOLD = 2.25
CURVE_SHARP_MOVE_PP = 0.25
CURVE_DEPRESSED_LEVEL = 0.25


class PredictorError(RuntimeError):
    """Raised when source data cannot support a reliable forecast."""


@dataclass(frozen=True)
class Metric:
    """Latest value, three-month change, reference level, and observation date."""

    current: float
    change_3m: float
    reference: float
    as_of: pd.Timestamp


@dataclass(frozen=True)
class Signal:
    """A binary leading indicator and its supporting metric."""

    active: bool
    metric: Metric


@dataclass(frozen=True)
class SignalSnapshot:
    """All growth, inflation, liquidity, and tail-risk measurements."""

    growth: Mapping[str, Signal]
    inflation: Mapping[str, Signal]
    bitcoin: Metric
    dollar: Metric
    yield_curve: Metric


@dataclass(frozen=True)
class Prediction:
    """Forward regime classification and confidence diagnostics."""

    regime: str
    growth_outlook: str
    inflation_outlook: str
    growth_score: int
    inflation_score: int
    confidence: int
    liquidity_status: str
    bull_steepening: bool
    stagflation_warning: bool
    signals: SignalSnapshot


@contextmanager
def _socket_timeout(seconds: float) -> Iterator[None]:
    """Apply a timeout to fredapi's urllib calls and restore prior state."""
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _clean_series(series: pd.Series) -> pd.Series:
    """Return sorted, numeric, non-null observations with unique dates."""
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    clean.index = pd.to_datetime(clean.index)
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)
    clean.index = clean.index.normalize()
    return clean[~clean.index.duplicated(keep="last")]


def _business_daily(
    series: pd.Series,
    end: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Forward-fill observations onto business days without backfilling."""
    clean = _clean_series(series)
    if clean.empty:
        raise PredictorError(f"{series.name or 'Series'} has no valid data")
    last_date = clean.index[-1] if end is None else pd.Timestamp(end).normalize()
    last_date = max(last_date, clean.index[-1])
    daily_index = pd.date_range(clean.index[0], last_date, freq="D")
    daily = clean.reindex(daily_index).ffill()
    return daily.asfreq("B").ffill().dropna()


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Align two market series using only current and past observations."""
    pair = pd.concat(
        [_clean_series(numerator), _clean_series(denominator)],
        axis=1,
    ).sort_index()
    pair.columns = ["numerator", "denominator"]
    pair = pair.ffill().dropna()
    pair = pair[pair["denominator"] != 0]
    return pair["numerator"] / pair["denominator"]


def _metric(
    series: pd.Series,
    reference: Optional[float] = None,
    absolute_change: bool = False,
) -> Metric:
    """Calculate a 63-observation change and latest reference level."""
    clean = _clean_series(series)
    if len(clean) < LOOKBACK_3M + 1:
        raise PredictorError(
            f"{series.name or 'Series'} needs at least 64 valid observations"
        )

    current = float(clean.iloc[-1])
    past = float(clean.iloc[-LOOKBACK_3M - 1])
    if reference is None:
        reference = float(clean.rolling(SMA_WINDOW).mean().iloc[-1])
    if absolute_change:
        change = current - past
    else:
        if past == 0:
            raise PredictorError(f"{series.name or 'Series'} has a zero baseline")
        change = (current / past - 1.0) * 100.0

    values = (current, past, change, reference)
    if not all(np.isfinite(value) for value in values):
        raise PredictorError(f"{series.name or 'Series'} has invalid calculations")
    return Metric(current, change, float(reference), clean.index[-1])


def _pct(value: float) -> str:
    """Format a percentage change with an explicit sign."""
    return f"{value:+.2f}%"


def _pp(value: float) -> str:
    """Format a percentage-point change with an explicit sign."""
    return f"{value:+.2f} pp"


class ForwardRegimePredictor:
    """Download leading data, evaluate signals, and classify forward regimes."""

    def __init__(
        self,
        api_key: str,
        years: int = 3,
        timeout: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise PredictorError("FRED_API_KEY is missing")
        if years < 2:
            raise ValueError("At least two years of history are required")
        self.api_key = api_key.strip()
        self.years = years
        self.timeout = timeout
        self.markets: Dict[str, pd.Series] = {}
        self.fred: Dict[str, pd.Series] = {}

    def fetch_data(self) -> None:
        """Fetch adjusted daily market closes and daily-aligned FRED series."""
        start = pd.Timestamp.today().normalize() - pd.DateOffset(years=self.years)
        self.markets = download_market_data(
            MARKET_TICKERS,
            years=self.years,
            timeout=self.timeout,
        )

        client = Fred(api_key=self.api_key)
        try:
            with _socket_timeout(self.timeout):
                for ticker in FRED_TICKERS:
                    series = client.get_series(
                        ticker,
                        observation_start=start.date().isoformat(),
                    )
                    aligned = _business_daily(
                        series,
                        end=pd.Timestamp.today(),
                    ).rename(ticker)
                    if len(aligned) < FRED_6M_WINDOW:
                        raise PredictorError(f"Insufficient FRED history for {ticker}")
                    self.fred[ticker] = aligned
        except PredictorError:
            raise
        except Exception as exc:
            raise PredictorError(f"FRED download failed: {exc}") from exc

    @staticmethod
    def _extract_closes(
        raw: pd.DataFrame,
        tickers: Sequence[str] = MARKET_TICKERS,
    ) -> pd.DataFrame:
        """Handle both yfinance MultiIndex column layouts and missing tickers."""
        if raw is None or raw.empty:
            raise PredictorError("Yahoo Finance returned an empty dataset")
        if not isinstance(raw.columns, pd.MultiIndex):
            if len(tickers) == 1 and "Close" in raw:
                return raw[["Close"]].rename(columns={"Close": tickers[0]})
            raise PredictorError("Unexpected Yahoo Finance column structure")

        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                closes = raw.xs("Close", axis=1, level=level)
                if isinstance(closes, pd.Series):
                    closes = closes.to_frame(tickers[0])
                return closes
        raise PredictorError("Yahoo Finance response has no Close column")

    def compute_signals(self) -> SignalSnapshot:
        """Compute the four growth and three inflation leading signals."""
        if not self.markets or not self.fred:
            raise PredictorError("fetch_data() must complete before compute_signals()")

        copper_gold = _ratio(self.markets["HG=F"], self.markets["GC=F"])
        credit = _ratio(self.markets["HYG"], self.markets["IEF"])
        gold_equities = _ratio(self.markets["GC=F"], self.markets["SPY"])

        copper_metric = _metric(copper_gold)
        credit_metric = _metric(credit)
        oil_metric = _metric(self.markets["CL=F"])
        gold_equity_metric = _metric(gold_equities)

        permits = self.fred["PERMIT"]
        permit_average = float(permits.rolling(LOOKBACK_3M).mean().iloc[-1])
        permit_metric = _metric(permits, reference=permit_average)

        claims = self.fred["IC4WSA"]
        claims_minimum = float(
            claims.rolling(FRED_6M_WINDOW).min().iloc[-1]
        )
        claims_metric = _metric(claims, reference=claims_minimum)
        breakeven_metric = _metric(
            self.fred["T5YIE"],
            reference=BREAKEVEN_THRESHOLD,
            absolute_change=True,
        )

        growth = {
            "copper_gold": Signal(
                copper_metric.change_3m > 0
                and copper_metric.current > copper_metric.reference,
                copper_metric,
            ),
            "credit": Signal(
                credit_metric.change_3m > 0
                and credit_metric.current > credit_metric.reference,
                credit_metric,
            ),
            "permits": Signal(
                permit_metric.current > permit_metric.reference,
                permit_metric,
            ),
            "claims": Signal(
                claims_metric.current <= claims_metric.reference * 1.10,
                claims_metric,
            ),
        }
        inflation = {
            "breakeven": Signal(
                breakeven_metric.change_3m > 0
                and breakeven_metric.current > BREAKEVEN_THRESHOLD,
                breakeven_metric,
            ),
            "oil": Signal(
                oil_metric.change_3m > 0
                and oil_metric.current > oil_metric.reference,
                oil_metric,
            ),
            "gold_equities": Signal(
                gold_equity_metric.change_3m > 0
                and gold_equity_metric.current > gold_equity_metric.reference,
                gold_equity_metric,
            ),
        }
        curve = self.fred["T10Y2Y"]
        curve_start = float(curve.iloc[-LOOKBACK_3M - 1])
        curve_metric = _metric(
            curve,
            reference=curve_start,
            absolute_change=True,
        )
        return SignalSnapshot(
            growth=growth,
            inflation=inflation,
            bitcoin=_metric(self.markets["BTC-USD"]),
            dollar=_metric(self.markets["DX-Y.NYB"]),
            yield_curve=curve_metric,
        )

    @staticmethod
    def classify_regime(signals: SignalSnapshot) -> Prediction:
        """Apply quadrant, tie-break, confidence, and tail-risk rules."""
        growth_score = sum(signal.active for signal in signals.growth.values())
        inflation_score = sum(
            signal.active for signal in signals.inflation.values()
        )

        if growth_score >= 3:
            growth_outlook = "RISING"
        elif growth_score <= 1:
            growth_outlook = "FALLING"
        else:
            copper_change = signals.growth["copper_gold"].metric.change_3m
            growth_outlook = "RISING" if copper_change > 0 else "FALLING"
        inflation_outlook = "RISING" if inflation_score >= 2 else "FALLING"

        regimes = {
            ("RISING", "FALLING"): "GOLDILOCKS",
            ("RISING", "RISING"): "REFLATION",
            ("FALLING", "RISING"): "STAGFLATION",
            ("FALLING", "FALLING"): "DEFLATION / RECESSION",
        }
        regime = regimes[(growth_outlook, inflation_outlook)]

        btc_change = signals.bitcoin.change_3m
        dollar_change = signals.dollar.change_3m
        if btc_change > 0 and dollar_change < 0:
            liquidity_status = "BULLISH"
        elif btc_change < 0 and dollar_change > 0:
            liquidity_status = "BEARISH"
        else:
            liquidity_status = "MIXED"

        growth_agreement = (
            growth_score if growth_outlook == "RISING" else 4 - growth_score
        ) / 4.0
        inflation_agreement = (
            inflation_score if inflation_outlook == "RISING" else 3 - inflation_score
        ) / 3.0
        confirms_growth = (
            (growth_outlook == "RISING" and liquidity_status == "BULLISH")
            or (growth_outlook == "FALLING" and liquidity_status == "BEARISH")
        )
        conflicts_growth = (
            (growth_outlook == "RISING" and liquidity_status == "BEARISH")
            or (growth_outlook == "FALLING" and liquidity_status == "BULLISH")
        )
        liquidity_agreement = 1.0 if confirms_growth else 0.0
        if not confirms_growth and not conflicts_growth:
            liquidity_agreement = 0.5
        confidence = round(
            100
            * (
                0.40 * growth_agreement
                + 0.40 * inflation_agreement
                + 0.20 * liquidity_agreement
            )
        )

        curve = signals.yield_curve
        bull_steepening = (
            curve.change_3m >= CURVE_SHARP_MOVE_PP
            and curve.reference <= CURVE_DEPRESSED_LEVEL
            and signals.growth["credit"].metric.change_3m < 0
        )
        stagflation_warning = (
            growth_outlook == "FALLING"
            and inflation_outlook == "RISING"
            and signals.inflation["gold_equities"].active
        )
        return Prediction(
            regime=regime,
            growth_outlook=growth_outlook,
            inflation_outlook=inflation_outlook,
            growth_score=growth_score,
            inflation_score=inflation_score,
            confidence=confidence,
            liquidity_status=liquidity_status,
            bull_steepening=bull_steepening,
            stagflation_warning=stagflation_warning,
            signals=signals,
        )

    @staticmethod
    def generate_report(prediction: Prediction) -> str:
        """Render the forward forecast as a structured terminal report."""
        signals = prediction.signals
        copper = signals.growth["copper_gold"]
        credit = signals.growth["credit"]
        permits = signals.growth["permits"]
        claims = signals.growth["claims"]
        breakeven = signals.inflation["breakeven"]
        oil = signals.inflation["oil"]
        gold_equities = signals.inflation["gold_equities"]
        curve = signals.yield_curve

        def flag(signal: Signal, rising: bool = False) -> str:
            if rising:
                return "RISING" if signal.active else "NOT RISING"
            return "BULLISH" if signal.active else "BEARISH"

        lines = [
            "=" * 78,
            f"6-Month Forward Regime: {prediction.regime} "
            f"(Confidence: {prediction.confidence}%)",
            "=" * 78,
            "",
            f"Forward Growth Outlook: {prediction.growth_outlook} "
            f"({prediction.growth_score}/4 leading signals bullish)",
            f"  - Copper/Gold (HG=F/GC=F): {flag(copper)} | "
            f"Current: {copper.metric.current:.6f} | "
            f"3M: {_pct(copper.metric.change_3m)} | "
            f"50D SMA: {copper.metric.reference:.6f}",
            f"  - High-Yield Credit (HYG/IEF): {flag(credit)} | "
            f"Current: {credit.metric.current:.6f} | "
            f"3M: {_pct(credit.metric.change_3m)} | "
            f"50D SMA: {credit.metric.reference:.6f}",
            f"  - Building Permits (PERMIT): {flag(permits)} | "
            f"Current: {permits.metric.current:,.2f} | "
            f"3M: {_pct(permits.metric.change_3m)} | "
            f"3M Avg: {permits.metric.reference:,.2f}",
            f"  - Jobless Claims (IC4WSA): {flag(claims)} | "
            f"Current: {claims.metric.current:,.2f} | "
            f"3M: {_pct(claims.metric.change_3m)} | "
            f"6M Min: {claims.metric.reference:,.2f} | "
            f"Within 10% of Min: "
            f"{'YES' if claims.active else 'NO'}",
            "",
            f"Forward Inflation Outlook: {prediction.inflation_outlook} "
            f"({prediction.inflation_score}/3 leading signals rising)",
            f"  - 5Y Breakeven (T5YIE): {flag(breakeven, rising=True)} | "
            f"Current: {breakeven.metric.current:.2f}% | "
            f"3M: {_pp(breakeven.metric.change_3m)} | "
            f"Above 2.25%: "
            f"{'YES' if breakeven.metric.current > 2.25 else 'NO'}",
            f"  - Crude Oil (CL=F): {flag(oil, rising=True)} | "
            f"Current: ${oil.metric.current:,.2f} | "
            f"3M: {_pct(oil.metric.change_3m)} | "
            f"50D SMA: ${oil.metric.reference:,.2f}",
            f"  - Gold/Equities (GC=F/SPY): {flag(gold_equities, rising=True)} | "
            f"Current: {gold_equities.metric.current:.6f} | "
            f"3M: {_pct(gold_equities.metric.change_3m)} | "
            f"50D SMA: {gold_equities.metric.reference:.6f}",
            "",
            "Liquidity & Tail-Risk Monitor:",
            f"  - Global Liquidity & Risk Appetite: {prediction.liquidity_status}",
            f"  - Bitcoin (BTC-USD): Current: ${signals.bitcoin.current:,.2f} | "
            f"3M: {_pct(signals.bitcoin.change_3m)}",
            f"  - US Dollar (DX-Y.NYB): Current: {signals.dollar.current:.2f} | "
            f"3M: {_pct(signals.dollar.change_3m)}",
            f"  - Yield Curve (T10Y2Y): Current: {curve.current:.2f}% | "
            f"3M: {_pp(curve.change_3m)} | "
            f"3M Start: {curve.reference:.2f}% | "
            f"HYG/IEF 3M: {_pct(credit.metric.change_3m)}",
            f"  - Bull-Steepening Warning: "
            f"{'ACTIVE' if prediction.bull_steepening else 'INACTIVE'}",
            f"  - STAGFLATION WARNING: "
            f"{'ACTIVE' if prediction.stagflation_warning else 'INACTIVE'}",
            "",
            f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        ]
        return "\n".join(lines)


def download_market_data(
    tickers: Sequence[str],
    years: int = 3,
    timeout: float = 30.0,
) -> Dict[str, pd.Series]:
    """Download adjusted closes with shared validation and column handling."""
    symbols = tuple(dict.fromkeys(tickers))
    if not symbols:
        return {}
    if years < 2:
        raise ValueError("At least two years of history are required")

    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    try:
        raw = yf.download(
            tickers=list(symbols),
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True,
            timeout=timeout,
        )
    except Exception as exc:
        raise PredictorError(f"Yahoo Finance download failed: {exc}") from exc

    closes = ForwardRegimePredictor._extract_closes(raw, symbols)
    result: Dict[str, pd.Series] = {}
    for ticker in symbols:
        if ticker not in closes:
            raise PredictorError(f"Yahoo Finance returned no data for {ticker}")
        series = _clean_series(closes[ticker]).rename(ticker)
        if len(series) < LOOKBACK_3M + 1:
            raise PredictorError(f"Insufficient Yahoo Finance history for {ticker}")
        result[ticker] = series
    return result


def market_metric(series: pd.Series) -> Metric:
    """Expose the predictor's 63-day return and 50-day SMA calculation."""
    return _metric(series)


def main() -> int:
    """Run the live forecast and return a shell-friendly exit status."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: set FRED_API_KEY before running regime_predictor.py",
            file=sys.stderr,
        )
        return 1

    predictor = ForwardRegimePredictor(api_key=api_key)
    try:
        predictor.fetch_data()
        signals = predictor.compute_signals()
        prediction = predictor.classify_regime(signals)
    except (PredictorError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(predictor.generate_report(prediction))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
