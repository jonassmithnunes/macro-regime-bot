from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from alert_monitor import build_command_response, build_parser, detect_events
from asset_playbook import AssetPlaybook
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
from gold_monitor import GoldMonitor, GoldSignal
from macro_regime_bot.regime import Comparison, Regime, RegimeAssessment
from regime_predictor import Metric, Prediction, Signal, SignalSnapshot
from notifier import (
    EDUCATIONAL_DISCLAIMER,
    NotificationError,
    TelegramNotifier,
    format_inflection_message,
)


def base_state() -> dict:
    return {
        "current_regime": "REFLATION",
        "forward_regime": "DEFLATION / RECESSION",
        "forward_confidence": 70,
        "gold_score": 1,
        "gold_readiness": "WAIT / FALLING KNIFE",
        "asset_above_sma": {"TLT": False, "BIL": True, "GLD": False},
        "stagflation_warning": False,
        "bull_steepening": False,
        "liquidity_status": "BEARISH",
    }


def test_expected_winner_breakout_above_sma_is_detected() -> None:
    previous = base_state()
    current = base_state()
    current["asset_above_sma"] = dict(previous["asset_above_sma"], TLT=True)
    events = detect_events(previous, current)
    assert events == ["TLT breakout: crossed above its 50-day SMA"]


def test_expected_winner_breakdown_below_sma_is_detected() -> None:
    previous = base_state()
    current = base_state()
    current["asset_above_sma"] = dict(previous["asset_above_sma"], BIL=False)
    events = detect_events(previous, current)
    assert events == ["BIL breakdown: crossed below its 50-day SMA"]


def test_new_asset_without_prior_state_is_not_a_crossover() -> None:
    previous = base_state()
    current = base_state()
    current["asset_above_sma"] = dict(previous["asset_above_sma"], SGOV=True)
    assert detect_events(previous, current) == []


@pytest.mark.parametrize(
    ("old", "new", "expected_text"),
    [
        (79, 80, "crossed above 80%"),
        (80, 90, ""),
        (60, 59, "dropped below 60%"),
        (59, 55, ""),
    ],
)
def test_forward_confidence_threshold_crossings(old, new, expected_text) -> None:
    previous = base_state()
    previous["forward_confidence"] = old
    current = base_state()
    current["forward_confidence"] = new
    events = detect_events(previous, current)
    if expected_text:
        assert any(expected_text in event for event in events)
    else:
        assert events == []


def test_gold_surge_and_two_point_jump_are_detected() -> None:
    previous = base_state()
    current = base_state()
    current["gold_score"] = 3
    assert "Gold Surge Trigger" in detect_events(previous, current)[0]
    current["gold_score"] = 3
    previous["gold_score"] = 1
    current["gold_readiness"] = "SURGE TRIGGER ACTIVE"
    assert detect_events(previous, current)


def test_liquidity_mixed_transition_does_not_count_as_flip() -> None:
    previous = base_state()
    current = base_state()
    current["liquidity_status"] = "MIXED"
    assert detect_events(previous, current) == []
    current["liquidity_status"] = "BULLISH"
    assert "Global Liquidity flipped" in detect_events(previous, current)[0]


def successful_response() -> Mock:
    response = Mock(status_code=200)
    response.json.return_value = {"ok": True}
    return response


def test_private_telegram_payload_formatting(monkeypatch, tmp_path) -> None:
    post = Mock(return_value=successful_response())
    monkeypatch.setattr("notifier.requests.post", post)
    monkeypatch.setattr(
        TelegramNotifier,
        "_macos_banner",
        staticmethod(lambda message: None),
    )
    log_path = tmp_path / "alerts.log"
    notifier = TelegramNotifier(
        token="test-secret-token",
        chat_id="123456789",
        log_path=log_path,
    )

    assert notifier.send(format_inflection_message(["TLT breakout"])) == 1
    kwargs = post.call_args.kwargs
    assert kwargs["timeout"] == 10
    assert kwargs["json"] == {
        "chat_id": "123456789",
        "text": post.call_args.kwargs["json"]["text"],
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    assert "TLT breakout" in kwargs["json"]["text"]
    assert EDUCATIONAL_DISCLAIMER in kwargs["json"]["text"]
    assert "test-secret-token" not in log_path.read_text()


def test_chat_id_is_required_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    notifier = TelegramNotifier(token="token", log_path=tmp_path / "alerts.log")
    assert notifier.chat_id == ""
    with pytest.raises(NotificationError, match="TELEGRAM_CHAT_ID"):
        notifier._post("Test alert")


def test_telegram_network_error_does_not_leak_token(monkeypatch, tmp_path) -> None:
    def fail(*args, **kwargs):
        raise requests.ConnectionError("request containing test-secret-token failed")

    monkeypatch.setattr("notifier.requests.post", fail)
    monkeypatch.setattr(
        TelegramNotifier,
        "_macos_banner",
        staticmethod(lambda message: None),
    )
    notifier = TelegramNotifier(
        token="test-secret-token",
        log_path=tmp_path / "alerts.log",
    )
    with pytest.raises(NotificationError) as error:
        notifier.send("Test alert")
    assert "test-secret-token" not in str(error.value)


def sample_comparison(current: float, baseline: float) -> Comparison:
    return Comparison(
        current,
        baseline,
        pd.Timestamp("2026-10-01"),
        pd.Timestamp("2025-10-01"),
    )


def sample_current() -> RegimeAssessment:
    return RegimeAssessment(
        regime=Regime.REFLATION,
        growth_expanding=True,
        growth_votes=3,
        growth_signals={
            "real_gdp": sample_comparison(4.28, 2.22),
            "industrial_production": sample_comparison(1.66, 1.25),
            "coincident_activity": sample_comparison(3.245, 2.7325),
        },
        inflation_rising=True,
        headline_inflation=sample_comparison(4.08, 3.02),
        core_inflation=sample_comparison(2.87, 3.02),
        coincident_series="WEI",
    )


def sample_metric(current: float = 101.0, reference: float = 100.0) -> Metric:
    return Metric(current, 5.25, reference, pd.Timestamp("2026-11-25"))


def sample_forward() -> Prediction:
    growth = {
        "copper_gold": Signal(False, sample_metric()),
        "credit": Signal(False, sample_metric()),
        "permits": Signal(False, sample_metric()),
        "claims": Signal(True, sample_metric()),
    }
    inflation = {
        "breakeven": Signal(False, sample_metric()),
        "oil": Signal(False, sample_metric()),
        "gold_equities": Signal(False, sample_metric()),
    }
    signals = SignalSnapshot(
        growth=growth,
        inflation=inflation,
        bitcoin=sample_metric(),
        dollar=sample_metric(),
        yield_curve=sample_metric(),
    )
    return Prediction(
        regime="DEFLATION / RECESSION",
        growth_outlook="FALLING",
        inflation_outlook="FALLING",
        growth_score=1,
        inflation_score=0,
        confidence=90,
        liquidity_status="BEARISH",
        bull_steepening=False,
        stagflation_warning=False,
        signals=signals,
    )


def sample_gold():
    names = ("real_yields", "dollar", "fed_pivot", "gold_technical")
    signals = {
        name: GoldSignal(index == 2, 100.0, 101.0, -0.25)
        for index, name in enumerate(names)
    }
    return GoldMonitor.build_assessment(
        signals,
        gold_silver_20d=15.93,
        as_of=pd.Timestamp("2026-11-25"),
    )


def sample_momentum() -> dict[str, Metric]:
    confirming = {"TLT", "BIL", "XLU", "UUP"}
    return {
        ticker: sample_metric(
            current=101.0 if ticker in confirming else 90.0,
            reference=100.0,
        )
        for ticker in AssetPlaybook.required_tickers("DEFLATION / RECESSION")
    }


def test_run_command_formatter() -> None:
    report = format_run_report(
        ReportContext(
            sample_current(),
            sample_forward(),
            sample_momentum(),
            sample_gold(),
        )
    )
    assert "Current Regime: *REFLATION*" in report
    assert "90% confidence" in report
    assert "Confirming Asset Winners" in report
    assert "TLT" in report
    assert "Gold Bounce Score: *1/4*" in report
    assert EDUCATIONAL_DISCLAIMER in report


def test_regime_command_formatter_has_exact_comparisons() -> None:
    report = format_regime_report(sample_current())
    assert "Real GDP: 4.28% vs 2.22% baseline" in report
    assert "Industrial Production: 1.66% vs 1.25% baseline" in report
    assert "WEI: 3.2450 vs 2.7325 baseline" in report
    assert "Headline CPI (12M): 4.08% vs 3.02% baseline" in report
    assert "Core CPI (12M): 2.87% vs 3.02% baseline" in report


def test_forecast_command_formatter_has_all_signals_and_risks() -> None:
    report = format_forecast_report(sample_forward())
    for text in (
        "Copper/Gold",
        "HYG/IEF Credit",
        "Building Permits",
        "Jobless Claims",
        "5Y Breakeven",
        "Crude Oil",
        "Gold/Equities",
        "BTC-USD",
        "DXY",
        "T10Y2Y",
        "Bull-Steepening Warning: INACTIVE",
        "STAGFLATION WARNING: INACTIVE",
    ):
        assert text in report


def test_assets_command_formatter_has_core_and_specialized_assets() -> None:
    report = format_assets_report(sample_forward(), sample_momentum())
    for text in (
        "Stocks (SPY)",
        "Long-Term Bonds (TLT)",
        "Cash / T-Bills (BIL)",
        "Commodities (DBC)",
        "Gold (GLD)",
        "Silver (SLV)",
        "Copper (CPER)",
        "Long-Duration US Treasuries (TLT)",
        "Defensive Equity Sectors (XLU)",
        "US Dollar & Investment-Grade Credit (UUP)",
        "[ABOVE 50d SMA]",
        "[BELOW 50d SMA]",
    ):
        assert text in report


def test_gold_command_formatter_has_prices_and_four_factors() -> None:
    gld = PriceSnapshot(327.45, -0.50, -22.52, pd.Timestamp("2026-11-25"))
    futures = PriceSnapshot(4100.0, 0.75, -20.0, pd.Timestamp("2026-11-25"))
    report = format_gold_report(sample_gold(), gld, futures)
    for text in (
        "GLD: $327.45 | 1D -0.50% | 3M -22.52%",
        "Gold Futures (GC=F): $4,100.00 | 1D +0.75% | 3M -20.00%",
        "DFII10 Real Yield",
        "UUP Dollar",
        "DGS2 2Y Yield",
        "GLD Trend",
        "GLD/SLV Ratio 20D Trend: +15.93%",
    ):
        assert text in report


def test_macro_command_formatter_contains_every_tracked_datapoint() -> None:
    prices = {
        ticker: PriceSnapshot(100.0, 1.0, 3.0, pd.Timestamp("2026-11-25"))
        for ticker in MACRO_MARKET_TICKERS
    }
    observations = {
        ticker: RawObservation(2.5, "2026-11-25") for ticker in MACRO_FRED_TICKERS
    }
    report = format_macro_report(prices, observations)
    for ticker in (*MACRO_MARKET_TICKERS, *MACRO_FRED_TICKERS):
        assert ticker in report


def test_help_command_formatter_and_parser_cover_all_commands() -> None:
    report = format_help_report()
    commands = (
        "run",
        "regime",
        "forecast",
        "assets",
        "gold",
        "goldprice",
        "macro",
        "help",
    )
    for command in commands:
        assert f"`{command}`" in report
    assert build_command_response("help") == report
    args = build_parser().parse_args(["--command", "goldprice"])
    assert args.command == "goldprice"
    assert build_parser().parse_args([]).command == "scheduled"
