#!/usr/bin/env python3
"""Unified coincident regime, forward forecast, and asset allocation dashboard."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

from asset_playbook import AssetPlaybook, RegimePlaybook
from gold_monitor import GoldAssessment, GoldMonitor
from macro_regime_bot.config import series_ids
from macro_regime_bot.fred import FredClient, FredError
from macro_regime_bot.regime import RegimeAssessment, analyze_regime
from regime_predictor import (
    ForwardRegimePredictor,
    Metric,
    Prediction,
    PredictorError,
    download_market_data,
    market_metric,
)


SEPARATOR = "=" * 88


def run_regime_engines(
    api_key: str,
    coincident_series: str = "WEI",
) -> Tuple[RegimeAssessment, Prediction, ForwardRegimePredictor]:
    """Run the existing coincident and forward engines without fetch duplication."""
    coincident_data = FredClient(api_key).fetch_many(series_ids(coincident_series))
    current = analyze_regime(coincident_data, coincident_series=coincident_series)

    forward_engine = ForwardRegimePredictor(api_key=api_key)
    forward_engine.fetch_data()
    forward_signals = forward_engine.compute_signals()
    forward = forward_engine.classify_regime(forward_signals)
    return current, forward, forward_engine


def collect_asset_momentum(
    forward: Prediction,
    engine: ForwardRegimePredictor,
) -> Dict[str, Metric]:
    """Reuse predictor prices and fetch only missing core/play ETF proxies."""
    required = AssetPlaybook.required_tickers(forward.regime)
    market_data = dict(engine.markets)
    missing = tuple(ticker for ticker in required if ticker not in market_data)
    market_data.update(
        download_market_data(
            missing,
            years=engine.years,
            timeout=engine.timeout,
        )
    )
    return {ticker: market_metric(market_data[ticker]) for ticker in required}


def _momentum_status(metric: Metric) -> str:
    """Format live three-month momentum and the required SMA status tag."""
    if metric.current > metric.reference:
        tag = "[CONFIRMING: Above 50d SMA]"
    else:
        tag = "[EARLY / NOT YET MOVING: Below 50d SMA]"
    return f"Live 3M: {metric.change_3m:+.2f}% | {tag}"


def generate_dashboard(
    current: RegimeAssessment,
    forward: Prediction,
    momentum: Mapping[str, Metric],
    gold: GoldAssessment,
) -> str:
    """Render regime transitions, historical biases, and live confirmations."""
    current_name = AssetPlaybook.normalize_regime(current.regime.value)
    forward_name = AssetPlaybook.normalize_regime(forward.regime)
    playbook = AssetPlaybook.for_regime(forward_name)
    transition_note = AssetPlaybook.transition_note(current_name, forward_name)

    lines = [
        SEPARATOR,
        "REGIME SUMMARY & TRANSITION PATH",
        SEPARATOR,
        f"Current Coincident Regime: {current_name}",
        f"6-Month Forward Regime: {forward_name} "
        f"(Confidence: {forward.confidence}%)",
        f"Transition Path: {current_name} -> {forward_name}",
        f"Forward Growth: {forward.growth_outlook} | "
        f"Forward Inflation: {forward.inflation_outlook}",
    ]
    if transition_note:
        lines.extend(("", f"Transition Note: {transition_note}"))

    latest_market_date = max(metric.as_of for metric in momentum.values())
    lines.extend(
        (
            "",
            SEPARATOR,
            "CORE ASSET CLASS OUTLOOK",
            SEPARATOR,
            f"Live market data through: {latest_market_date.date().isoformat()}",
        )
    )
    lines.extend(_format_core_assets(playbook, momentum))

    lines.extend(("", SEPARATOR, "HIGH-CONVICTION REGIME PLAYS", SEPARATOR))
    for number, play in enumerate(playbook.plays, start=1):
        proxies = " / ".join(play.proxies)
        lines.extend(
            (
                f"{number}. {play.name} ({proxies})",
                f"   Rationale: {play.rationale}",
                f"   {play.live_proxy}: "
                f"{_momentum_status(momentum[play.live_proxy])}",
            )
        )

    lines.extend(
        (
            "",
            SEPARATOR,
            GoldMonitor.generate_report(gold),
        )
    )

    lines.extend(
        (
            "",
            SEPARATOR,
            "EDUCATIONAL DISCLAIMER",
            SEPARATOR,
            "Historical macro correlations are for informational research only, "
            "not financial advice.",
        )
    )
    return "\n".join(lines)


def _format_core_assets(
    playbook: RegimePlaybook,
    momentum: Mapping[str, Metric],
) -> Sequence[str]:
    """Format all seven core assets in the historical macro matrix order."""
    return [
        f"{asset.name} ({asset.proxy}): {asset.bias} | "
        f"{_momentum_status(momentum[asset.proxy])}"
        for asset in playbook.core_assets
    ]


def build_parser() -> argparse.ArgumentParser:
    """Create the dashboard command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run coincident/forward regimes and the asset playbook."
    )
    parser.add_argument(
        "--coincident-series",
        choices=("WEI", "USPHCI"),
        default=os.getenv("COINCIDENT_SERIES", "WEI").upper(),
        help="Coincident activity proxy for the current regime (default: WEI)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the complete live dashboard and return a shell exit status."""
    args = build_parser().parse_args(argv)
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print("Error: set FRED_API_KEY before running dashboard.py", file=sys.stderr)
        return 1

    try:
        current, forward, engine = run_regime_engines(
            api_key,
            coincident_series=args.coincident_series,
        )
        momentum = collect_asset_momentum(forward, engine)
        gold_engine = GoldMonitor(api_key)
        gold_engine.fetch_data()
        gold = gold_engine.evaluate()
        print(generate_dashboard(current, forward, momentum, gold))
    except (FredError, PredictorError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
