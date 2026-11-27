#!/usr/bin/env python3
"""Stateful high-impact inflection detection and private alert dispatch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from asset_playbook import AssetPlaybook, OUTPERFORM
from command_reports import (
    MACRO_FRED_TICKERS,
    MACRO_MARKET_TICKERS,
    PriceSnapshot,
    RawObservation,
    ReportContext,
    format_assets_report,
    format_forecast_report,
    format_gold_report,
    format_help_report,
    format_macro_report,
    format_regime_report,
    format_run_report,
)
from dashboard import (
    collect_asset_momentum,
    generate_dashboard,
    run_regime_engines,
)
from gold_monitor import GoldAssessment, GoldMonitor
from macro_regime_bot.config import series_ids
from macro_regime_bot.fred import FredClient, FredError
from macro_regime_bot.regime import RegimeAssessment, analyze_regime
from notifier import (
    NotificationError,
    TelegramNotifier,
    format_inflection_message,
    format_summary_message,
    format_test_message,
)
from regime_predictor import (
    ForwardRegimePredictor,
    Metric,
    Prediction,
    PredictorError,
    download_market_data,
)


STATE_PATH = Path("data/last_state.json")
COMMAND_CHOICES = (
    "scheduled",
    "run",
    "regime",
    "forecast",
    "assets",
    "gold",
    "goldprice",
    "macro",
    "help",
)


@dataclass(frozen=True)
class MonitorSnapshot:
    """Serializable signal state plus its full terminal dashboard rendering."""

    state: Mapping[str, Any]
    dashboard: str


def build_state(
    current: RegimeAssessment,
    forward: Prediction,
    momentum: Mapping[str, Metric],
    gold: GoldAssessment,
) -> Dict[str, Any]:
    """Capture only alert-relevant values, avoiding timestamp-only commits."""
    playbook = AssetPlaybook.for_regime(forward.regime)
    winners = {
        asset.proxy for asset in playbook.core_assets if asset.bias == OUTPERFORM
    }
    winners.update(play.live_proxy for play in playbook.plays)
    return {
        "current_regime": AssetPlaybook.normalize_regime(current.regime.value),
        "forward_regime": AssetPlaybook.normalize_regime(forward.regime),
        "forward_confidence": forward.confidence,
        "gold_score": gold.score,
        "gold_readiness": gold.readiness,
        "asset_above_sma": {
            ticker: momentum[ticker].current > momentum[ticker].reference
            for ticker in sorted(winners)
        },
        "stagflation_warning": forward.stagflation_warning,
        "bull_steepening": forward.bull_steepening,
        "liquidity_status": forward.liquidity_status,
    }


def detect_events(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> List[str]:
    """Return only newly triggered regime, Gold, asset, and tail-risk events."""
    if not previous:
        return []
    events: List[str] = []

    for key, label in (
        ("current_regime", "Current Regime"),
        ("forward_regime", "6-Month Forward Regime"),
    ):
        if key in previous and previous[key] != current[key]:
            events.append(f"{label} shifted: {previous[key]} -> {current[key]}")

    if "forward_confidence" in previous:
        old_confidence = int(previous["forward_confidence"])
        new_confidence = int(current["forward_confidence"])
        if old_confidence < 80 <= new_confidence:
            events.append(
                f"Forward Confidence crossed above 80%: "
                f"{old_confidence}% -> {new_confidence}%"
            )
        if old_confidence >= 60 > new_confidence:
            events.append(
                f"Forward Confidence dropped below 60%: "
                f"{old_confidence}% -> {new_confidence}%"
            )

    if "gold_score" in previous:
        old_gold = int(previous["gold_score"])
        new_gold = int(current["gold_score"])
        if old_gold < 3 <= new_gold:
            events.append(f"Gold Surge Trigger activated: {old_gold}/4 -> {new_gold}/4")
        elif new_gold - old_gold >= 2:
            events.append(f"Gold Score jumped two points: {old_gold}/4 -> {new_gold}/4")

    old_assets = previous.get("asset_above_sma", {})
    for ticker, above_sma in current.get("asset_above_sma", {}).items():
        if ticker not in old_assets or bool(old_assets[ticker]) == bool(above_sma):
            continue
        if above_sma:
            events.append(f"{ticker} breakout: crossed above its 50-day SMA")
        else:
            events.append(f"{ticker} breakdown: crossed below its 50-day SMA")

    for key, label in (
        ("stagflation_warning", "STAGFLATION WARNING"),
        ("bull_steepening", "Bull-Steepening Warning"),
    ):
        if key in previous and bool(previous[key]) != bool(current[key]):
            events.append(f"{label} changed: {previous[key]} -> {current[key]}")

    old_liquidity = previous.get("liquidity_status")
    new_liquidity = current.get("liquidity_status")
    liquid_states = {"BULLISH", "BEARISH"}
    if (
        old_liquidity in liquid_states
        and new_liquidity in liquid_states
        and old_liquidity != new_liquidity
    ):
        events.append(f"Global Liquidity flipped: {old_liquidity} -> {new_liquidity}")
    return events


def load_state(path: Path = STATE_PATH) -> Dict[str, Any]:
    """Load prior state without silently overwriting corrupt alert memory."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"State file must contain a JSON object: {path}")
    return state


def save_state(state: Mapping[str, Any], path: Path = STATE_PATH) -> None:
    """Atomically persist state so interrupted cloud runs cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def create_report_context(
    api_key: str,
    coincident_series: str = "WEI",
) -> ReportContext:
    """Fetch the complete context used by scheduled and unified run reports."""
    current, forward, engine = run_regime_engines(api_key, coincident_series)
    momentum = collect_asset_momentum(forward, engine)
    gold_engine = GoldMonitor(api_key)
    gold_engine.fetch_data()
    gold = gold_engine.evaluate()
    return ReportContext(current, forward, momentum, gold)


def create_snapshot(
    api_key: str,
    coincident_series: str = "WEI",
) -> MonitorSnapshot:
    """Run both regimes, asset momentum, Gold catalysts, and dashboard output."""
    context = create_report_context(api_key, coincident_series)
    state = build_state(
        context.current,
        context.forward,
        context.momentum,
        context.gold,
    )
    dashboard = generate_dashboard(
        context.current,
        context.forward,
        context.momentum,
        context.gold,
    )
    return MonitorSnapshot(state=state, dashboard=dashboard)


def _forward_prediction(api_key: str) -> tuple[Prediction, ForwardRegimePredictor]:
    """Run only the forward engine for forecast and asset commands."""
    engine = ForwardRegimePredictor(api_key=api_key)
    engine.fetch_data()
    prediction = engine.classify_regime(engine.compute_signals())
    return prediction, engine


def build_command_response(
    command: str,
    api_key: str = "",
    coincident_series: str = "WEI",
) -> str:
    """Fetch command-specific live data and return its Telegram Markdown report."""
    if command == "help":
        return format_help_report()
    if command == "scheduled":
        raise ValueError("Use the stateful pipeline for the scheduled command")
    if not api_key.strip():
        raise PredictorError("FRED_API_KEY is missing")

    if command == "run":
        return format_run_report(create_report_context(api_key, coincident_series))
    if command == "regime":
        data = FredClient(api_key).fetch_many(series_ids(coincident_series))
        return format_regime_report(
            analyze_regime(data, coincident_series=coincident_series)
        )
    if command == "forecast":
        forward, _ = _forward_prediction(api_key)
        return format_forecast_report(forward)
    if command == "assets":
        forward, engine = _forward_prediction(api_key)
        momentum = collect_asset_momentum(forward, engine)
        return format_assets_report(forward, momentum)
    if command in {"gold", "goldprice"}:
        engine = GoldMonitor(api_key)
        engine.fetch_data()
        assessment = engine.evaluate()
        futures_data = download_market_data(
            ("GC=F",),
            years=engine.years,
            timeout=engine.timeout,
        )
        return format_gold_report(
            assessment,
            PriceSnapshot.from_series(engine.markets["GLD"]),
            PriceSnapshot.from_series(futures_data["GC=F"]),
        )
    if command == "macro":
        market_data = download_market_data(MACRO_MARKET_TICKERS)
        prices = {
            ticker: PriceSnapshot.from_series(series)
            for ticker, series in market_data.items()
        }
        fred_data = FredClient(api_key).fetch_many(list(MACRO_FRED_TICKERS))
        observations = {}
        for ticker, series in fred_data.items():
            clean = series.dropna()
            if clean.empty:
                raise PredictorError(f"No current FRED observation for {ticker}")
            observations[ticker] = RawObservation(
                float(clean.iloc[-1]),
                clean.index[-1].date().isoformat(),
            )
        return format_macro_report(prices, observations)
    raise ValueError(f"Unsupported command: {command}")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI controls for normal, test, and forced-summary operation."""
    parser = argparse.ArgumentParser(
        description="Send private alerts only for high-impact macro inflections."
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--test-alert",
        action="store_true",
        help="Send a private Telegram delivery test immediately",
    )
    actions.add_argument(
        "--force-summary",
        action="store_true",
        help="Send the full live dashboard regardless of state changes",
    )
    parser.add_argument(
        "--command",
        choices=COMMAND_CHOICES,
        default="scheduled",
        help="On-demand Telegram report or stateful scheduled monitoring",
    )
    parser.add_argument(
        "--coincident-series",
        choices=("WEI", "USPHCI"),
        default=os.getenv("COINCIDENT_SERIES", "WEI").upper(),
        help="Coincident activity proxy for the current regime",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run stateful monitoring and suppress notifications when nothing changes."""
    parser = build_parser()
    args = parser.parse_args(argv)
    notifier = TelegramNotifier()
    if args.test_alert:
        try:
            notifier.send(format_test_message())
        except NotificationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print("Private test alert sent.")
        return 0

    if args.force_summary and args.command != "scheduled":
        parser.error("--force-summary requires the default scheduled command")
    if args.command == "help":
        try:
            notifier.send(build_command_response("help"))
        except NotificationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print("Command help sent privately.")
        return 0

    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: set FRED_API_KEY before running alert_monitor.py",
            file=sys.stderr,
        )
        return 1

    try:
        if args.command != "scheduled":
            response = build_command_response(
                args.command,
                api_key,
                args.coincident_series,
            )
            notifier.send(response)
            print(f"{args.command} report sent privately.")
            return 0

        snapshot = create_snapshot(api_key, args.coincident_series)
        previous = load_state()
        events = detect_events(previous, snapshot.state)
        if args.force_summary:
            notifier.send(format_summary_message(snapshot.dashboard))
            print("Full dashboard snapshot sent privately.")
        elif events:
            notifier.send(format_inflection_message(events))
            print(f"Private alert sent for {len(events)} inflection event(s).")
        else:
            print("No high-impact inflections detected; no alert sent.")
        save_state(snapshot.state)
    except (
        FredError,
        NotificationError,
        PredictorError,
        ValueError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
