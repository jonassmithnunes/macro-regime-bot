"""Concise Telegram Markdown formatters for on-demand macro commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from asset_playbook import AssetPlaybook, OUTPERFORM
from gold_monitor import GoldAssessment
from macro_regime_bot.regime import Comparison, RegimeAssessment
from notifier import EDUCATIONAL_DISCLAIMER
from regime_predictor import Metric, Prediction, Signal, market_metric


MACRO_MARKET_TICKERS = (
    "SPY",
    "TLT",
    "GLD",
    "GC=F",
    "SLV",
    "CPER",
    "HG=F",
    "CL=F",
    "BTC-USD",
    "UUP",
)
MACRO_FRED_TICKERS = (
    "DFII10",
    "DGS2",
    "T10Y2Y",
    "T5YIE",
    "CPIAUCSL",
    "CPILFESL",
    "GDPC1",
)
COMMAND_DESCRIPTIONS = (
    ("run", "Unified regime, confirming asset winners, and Gold summary"),
    ("regime", "Coincident growth and inflation indicator breakdown"),
    ("forecast", "Six-month forward signals, liquidity, and tail risks"),
    ("assets", "Predicted-regime allocation playbook and live SMA checks"),
    ("gold", "Gold prices, momentum, and four-factor bounce monitor"),
    ("goldprice", "Alias for the gold command"),
    ("macro", "Raw tracked market prices and FRED observations"),
    ("help", "List available on-demand commands"),
    ("scheduled", "Run stateful monitoring and alert only on inflections"),
)


@dataclass(frozen=True)
class PriceSnapshot:
    """Latest adjusted close with one-day and three-month returns."""

    current: float
    change_1d: float
    change_3m: float
    as_of: pd.Timestamp

    @classmethod
    def from_series(cls, series: pd.Series) -> "PriceSnapshot":
        """Calculate price changes from a validated Yahoo Finance series."""
        clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
        if len(clean) < 64 or clean.iloc[-2] == 0:
            raise ValueError(f"Insufficient price history for {series.name or 'asset'}")
        metric = market_metric(clean)
        change_1d = float((clean.iloc[-1] / clean.iloc[-2] - 1.0) * 100.0)
        return cls(metric.current, change_1d, metric.change_3m, metric.as_of)


@dataclass(frozen=True)
class RawObservation:
    """One raw FRED value and its observation date."""

    value: float
    as_of: str


@dataclass(frozen=True)
class ReportContext:
    """Complete live context shared by dashboard and on-demand run reports."""

    current: RegimeAssessment
    forward: Prediction
    momentum: Mapping[str, Metric]
    gold: GoldAssessment


def _message(title: str, lines: Sequence[str]) -> str:
    """Wrap a report with a Markdown title and mandatory research disclaimer."""
    return "\n".join((f"*{title}*", "", *lines, "", EDUCATIONAL_DISCLAIMER))


def _sma_status(metric: Metric) -> str:
    """Return the compact on-demand SMA status tag."""
    return "[ABOVE 50d SMA]" if metric.current > metric.reference else "[BELOW 50d SMA]"


def _comparison(comparison: Comparison, digits: int = 2, percent: bool = True) -> str:
    """Format current and baseline values consistently across regime signals."""
    suffix = "%" if percent else ""
    return (
        f"{comparison.current:.{digits}f}{suffix} vs "
        f"{comparison.baseline:.{digits}f}{suffix} baseline "
        f"({comparison.change:+.{digits}f}; {comparison.to_dict()['direction']})"
    )


def format_run_report(context: ReportContext) -> str:
    """Summarize regimes, confirming expected winners, and Gold readiness."""
    current = AssetPlaybook.normalize_regime(context.current.regime.value)
    forward = AssetPlaybook.normalize_regime(context.forward.regime)
    playbook = AssetPlaybook.for_regime(forward)
    winners = {
        asset.proxy for asset in playbook.core_assets if asset.bias == OUTPERFORM
    }
    winners.update(play.live_proxy for play in playbook.plays)
    confirming = [
        ticker
        for ticker in sorted(winners)
        if context.momentum[ticker].current > context.momentum[ticker].reference
    ]
    lines = [
        f"Current Regime: *{current}*",
        f"6M Forward Regime: *{forward}* "
        f"({context.forward.confidence}% confidence)",
        f"Transition: {current} -> {forward}",
        f"Gold Bounce Score: *{context.gold.score}/4* — {context.gold.readiness}",
        "",
        "*Confirming Asset Winners*",
    ]
    if confirming:
        lines.extend(
            f"• {ticker}: {context.momentum[ticker].change_3m:+.2f}% 3M "
            f"{_sma_status(context.momentum[ticker])}"
            for ticker in confirming
        )
    else:
        lines.append("• None are currently above their 50-day SMA")
    return _message("UNIFIED MACRO DASHBOARD", lines)


def format_regime_report(current: RegimeAssessment) -> str:
    """Detail the five coincident growth and inflation comparisons."""
    growth = current.growth_signals
    proxy = current.coincident_series
    lines = [
        f"Regime: *{AssetPlaybook.normalize_regime(current.regime.value)}*",
        f"Growth: {current.to_dict()['growth']['direction']} "
        f"({current.growth_votes}/3 accelerating)",
        f"• Real GDP: {_comparison(growth['real_gdp'])}",
        f"• Industrial Production: {_comparison(growth['industrial_production'])}",
        f"• {proxy}: {_comparison(growth['coincident_activity'], 4, False)}",
        "",
        f"Inflation: {'RISING' if current.inflation_rising else 'FALLING'}",
        f"• Headline CPI (12M): {_comparison(current.headline_inflation)}",
        f"• Core CPI (12M): {_comparison(current.core_inflation)}",
    ]
    return _message("COINCIDENT CURRENT REGIME", lines)


def _signal_line(
    name: str,
    signal: Signal,
    inflation: bool = False,
    change_unit: str = "%",
) -> str:
    """Format one forward leading signal with its value and 3M change."""
    if inflation:
        status = "RISING" if signal.active else "NOT RISING"
    else:
        status = "BULLISH" if signal.active else "BEARISH"
    return (
        f"• {name}: {status} | {signal.metric.current:.4f} | "
        f"3M {signal.metric.change_3m:+.2f}{change_unit}"
    )


def format_forecast_report(forward: Prediction) -> str:
    """Detail all forward signals, liquidity, and tail-risk monitors."""
    growth_names = {
        "copper_gold": "Copper/Gold",
        "credit": "HYG/IEF Credit",
        "permits": "Building Permits",
        "claims": "Jobless Claims",
    }
    inflation_names = {
        "breakeven": "5Y Breakeven",
        "oil": "Crude Oil",
        "gold_equities": "Gold/Equities",
    }
    lines = [
        f"Regime: *{forward.regime}* ({forward.confidence}% confidence)",
        f"Growth Outlook: {forward.growth_outlook} ({forward.growth_score}/4 bullish)",
    ]
    lines.extend(
        _signal_line(name, forward.signals.growth[key])
        for key, name in growth_names.items()
    )
    lines.append(
        f"Inflation Outlook: {forward.inflation_outlook} "
        f"({forward.inflation_score}/3 rising)"
    )
    lines.extend(
        _signal_line(
            name,
            forward.signals.inflation[key],
            inflation=True,
            change_unit=" pp" if key == "breakeven" else "%",
        )
        for key, name in inflation_names.items()
    )
    lines.extend(
        (
            "",
            f"Liquidity: *{forward.liquidity_status}*",
            f"• BTC-USD: ${forward.signals.bitcoin.current:,.2f} | "
            f"3M {forward.signals.bitcoin.change_3m:+.2f}%",
            f"• DXY: {forward.signals.dollar.current:.2f} | "
            f"3M {forward.signals.dollar.change_3m:+.2f}%",
            f"• T10Y2Y: {forward.signals.yield_curve.current:.2f}% | "
            f"3M {forward.signals.yield_curve.change_3m:+.2f} pp",
            f"Bull-Steepening Warning: "
            f"{'ACTIVE' if forward.bull_steepening else 'INACTIVE'}",
            f"STAGFLATION WARNING: "
            f"{'ACTIVE' if forward.stagflation_warning else 'INACTIVE'}",
        )
    )
    return _message("6-MONTH FORWARD REGIME", lines)


def format_assets_report(forward: Prediction, momentum: Mapping[str, Metric]) -> str:
    """List core biases and specialized plays with live momentum confirmation."""
    playbook = AssetPlaybook.for_regime(forward.regime)
    lines = [f"Predicted Regime: *{forward.regime}*", "", "*Core Assets*"]
    lines.extend(
        f"• {asset.name} ({asset.proxy}): {asset.bias} | "
        f"3M {momentum[asset.proxy].change_3m:+.2f}% "
        f"{_sma_status(momentum[asset.proxy])}"
        for asset in playbook.core_assets
    )
    lines.extend(("", "*Top Specialized Plays*"))
    lines.extend(
        f"• {play.name} ({play.live_proxy}): "
        f"3M {momentum[play.live_proxy].change_3m:+.2f}% "
        f"{_sma_status(momentum[play.live_proxy])}"
        for play in playbook.plays
    )
    return _message("ASSET ALLOCATION PLAYBOOK", lines)


def format_gold_report(
    assessment: GoldAssessment,
    gld: PriceSnapshot,
    futures: PriceSnapshot,
) -> str:
    """Report GLD/futures momentum and every Gold bounce catalyst."""
    signals = assessment.signals
    real_yield = signals["real_yields"]
    dollar = signals["dollar"]
    front_end = signals["fed_pivot"]
    technical = signals["gold_technical"]
    lines = [
        f"• GLD: ${gld.current:,.2f} | 1D {gld.change_1d:+.2f}% | "
        f"3M {gld.change_3m:+.2f}%",
        f"• Gold Futures (GC=F): ${futures.current:,.2f} | "
        f"1D {futures.change_1d:+.2f}% | 3M {futures.change_3m:+.2f}%",
        "",
        f"Gold Readiness: *{assessment.readiness}* ({assessment.score}/4)",
        f"• DFII10 Real Yield: {real_yield.current:.2f}% vs "
        f"50D {real_yield.sma:.2f}% | 10D {real_yield.change_10d:+.2f} pp | "
        f"{'ACTIVE' if real_yield.active else 'INACTIVE'}",
        f"• UUP Dollar: ${dollar.current:.2f} vs 20D ${dollar.sma:.2f} | "
        f"{'ACTIVE' if dollar.active else 'INACTIVE'}",
        f"• DGS2 2Y Yield: {front_end.current:.2f}% vs "
        f"50D {front_end.sma:.2f}% | "
        f"{'ACTIVE' if front_end.active else 'INACTIVE'}",
        f"• GLD Trend: ${technical.current:.2f} vs 50D ${technical.sma:.2f} | "
        f"10D {technical.change_10d:+.2f}% | "
        f"{'ACTIVE' if technical.active else 'INACTIVE'}",
        f"• GLD/SLV Ratio 20D Trend: {assessment.gold_silver_20d:+.2f}%",
    ]
    return _message("GOLD PRICE & BOUNCE MONITOR", lines)


def format_macro_report(
    prices: Mapping[str, PriceSnapshot],
    fred: Mapping[str, RawObservation],
) -> str:
    """Render the complete compact raw market and FRED datapoint sheet."""
    lines = ["*Market Prices*"]
    lines.extend(
        f"• {ticker}: {prices[ticker].current:,.2f} "
        f"({prices[ticker].as_of.date().isoformat()})"
        for ticker in MACRO_MARKET_TICKERS
    )
    lines.extend(("", "*FRED Observations*"))
    lines.extend(
        f"• {ticker}: {fred[ticker].value:,.2f} ({fred[ticker].as_of})"
        for ticker in MACRO_FRED_TICKERS
    )
    return _message("MACRO RAW DATAPOINT SHEET", lines)


def format_help_report() -> str:
    """List every dropdown/CLI command and its purpose."""
    lines = [
        f"• `{command}` — {description}"
        for command, description in COMMAND_DESCRIPTIONS
    ]
    return _message("MACRO BOT COMMANDS", lines)


__all__ = [
    "COMMAND_DESCRIPTIONS",
    "MACRO_FRED_TICKERS",
    "MACRO_MARKET_TICKERS",
    "PriceSnapshot",
    "RawObservation",
    "ReportContext",
    "format_assets_report",
    "format_forecast_report",
    "format_gold_report",
    "format_help_report",
    "format_macro_report",
    "format_regime_report",
    "format_run_report",
]
