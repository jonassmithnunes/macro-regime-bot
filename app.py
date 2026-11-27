"""Streamlit showcase for the macro regime and quantitative Gold research engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from asset_playbook import AssetPlaybook
from backtest import compute_strict_gold_filter
from dashboard import collect_asset_momentum, run_regime_engines
from gold_monitor import GoldMonitor


ROOT = Path(__file__).parent
REPOSITORY_URL = "https://github.com/jonassmithnunes/macro-regime-bot"


def fred_api_key() -> str:
    """Read FRED credentials from Streamlit secrets or the process environment."""
    try:
        secret = str(st.secrets.get("FRED_API_KEY", "")).strip()
    except Exception:
        secret = ""
    return secret or os.getenv("FRED_API_KEY", "").strip()


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_research(api_key: str) -> Dict[str, Any]:
    """Load live regimes, asset momentum, and strict Gold catalyst status."""
    current, forward, engine = run_regime_engines(api_key)
    momentum = collect_asset_momentum(forward, engine)
    gold_engine = GoldMonitor(api_key)
    gold_engine.fetch_data()
    assessment = gold_engine.evaluate()

    real = gold_engine.fred["DFII10"]
    front = gold_engine.fred["DGS2"]
    dollar = gold_engine.markets["UUP"]
    gold = gold_engine.markets["GLD"]
    real_sma = real.rolling(50).mean()
    front_sma = front.rolling(50).mean()
    dollar_sma = dollar.rolling(20).mean()
    gold_sma = gold.rolling(50).mean()
    real_change = real.diff(10)
    gold_roc = gold.pct_change(10, fill_method=None)
    components = pd.DataFrame(
        {
            "real_yields": (real < real_sma) & (real_change < 0),
            "dollar": dollar < dollar_sma,
            "fed_pivot": front < front_sma,
            "gold_technical": (gold > gold_sma) & (gold_roc > 0),
        }
    )
    real_buffer = real <= real_sma - 0.03
    dollar_buffer = dollar <= dollar_sma * 0.9975
    strict_score, strict_active = compute_strict_gold_filter(
        components,
        real_buffer,
        dollar_buffer,
    )
    real_confirmed = real_buffer.rolling(3).sum() == 3
    dollar_confirmed = dollar_buffer.rolling(3).sum() == 3
    latest = components.index[-1]

    def macro_status(component: str, confirmed: pd.Series) -> str:
        if bool(components.loc[latest, component]) and bool(confirmed.loc[latest]):
            return "ACTIVE"
        if bool(components.loc[latest, component]):
            return "PENDING"
        return "INACTIVE"

    factors = [
        {
            "name": "DFII10 Real Yield",
            "value": f"{real.iloc[-1]:.2f}% vs 50D {real_sma.iloc[-1]:.2f}%",
            "status": macro_status("real_yields", real_confirmed),
        },
        {
            "name": "UUP Dollar",
            "value": f"${dollar.iloc[-1]:.2f} vs 20D ${dollar_sma.iloc[-1]:.2f}",
            "status": macro_status("dollar", dollar_confirmed),
        },
        {
            "name": "DGS2 2Y Yield",
            "value": f"{front.iloc[-1]:.2f}% vs 50D {front_sma.iloc[-1]:.2f}%",
            "status": "ACTIVE" if components.loc[latest, "fed_pivot"] else "INACTIVE",
        },
        {
            "name": "GLD Technical Trend",
            "value": f"${gold.iloc[-1]:.2f} vs 50D ${gold_sma.iloc[-1]:.2f}",
            "status": (
                "ACTIVE"
                if components.loc[latest, "gold_technical"]
                else "PENDING"
                if gold_roc.iloc[-1] > 0
                else "INACTIVE"
            ),
        },
    ]
    return {
        "current": current,
        "forward": forward,
        "momentum": momentum,
        "gold_score": int(strict_score.iloc[-1]),
        "gold_active": bool(strict_active.iloc[-1]),
        "gold_readiness": GoldMonitor.classify_score(int(strict_score.iloc[-1])),
        "gold_silver_20d": assessment.gold_silver_20d,
        "factors": factors,
    }


@st.cache_data(show_spinner=False)
def load_research_artifacts() -> Dict[str, Any]:
    """Load non-sensitive empirical curves and frozen result summaries."""
    equity = pd.read_csv(
        ROOT / "research" / "portfolio_equity.csv",
        parse_dates=["Date"],
    )
    with (ROOT / "research" / "portfolio_summary.json").open() as handle:
        portfolio = json.load(handle)
    with (ROOT / "research" / "validation_summary.json").open() as handle:
        validation = json.load(handle)
    return {"equity": equity, "portfolio": portfolio, "validation": validation}


def status_badge(status: str) -> str:
    """Return a consistently colored Streamlit Markdown status badge."""
    colors = {"ACTIVE": "green", "PENDING": "orange", "INACTIVE": "red"}
    return f":{colors[status]}[**{status}**]"


def render_live_tab() -> None:
    """Render optional live FRED/Yahoo data without requiring Telegram secrets."""
    api_key = fred_api_key()
    if not api_key:
        st.warning("Add `FRED_API_KEY` to Streamlit secrets to enable live data.")
        return
    with st.spinner("Loading live macroeconomic and market signals…"):
        try:
            live = load_live_research(api_key)
        except Exception:
            st.error(
                "Live data could not be loaded. Check `FRED_API_KEY` and "
                "provider availability."
            )
            return

    current = AssetPlaybook.normalize_regime(live["current"].regime.value)
    forward = live["forward"]
    columns = st.columns(3)
    columns[0].metric("Current Coincident Regime", current)
    columns[1].metric(
        "6-Month Forward Regime",
        forward.regime,
        f"{forward.confidence}% confidence",
    )
    columns[2].metric("Transition Path", f"{current} → {forward.regime}")

    st.subheader("Strict 4-Factor Gold Bounce Monitor")
    st.metric(
        "Gold Readiness",
        live["gold_readiness"],
        f"{live['gold_score']}/4 strict factors active",
    )
    factor_columns = st.columns(4)
    for column, factor in zip(factor_columns, live["factors"]):
        with column:
            st.markdown(f"**{factor['name']}**")
            st.markdown(status_badge(factor["status"]))
            st.caption(factor["value"])
    st.caption(f"GLD / SLV 20-day leadership change: {live['gold_silver_20d']:+.2f}%")

    st.subheader("Asset Allocation Playbook")
    playbook = AssetPlaybook.for_regime(forward.regime)
    rows = []
    for asset in playbook.core_assets:
        metric = live["momentum"][asset.proxy]
        rows.append(
            {
                "Asset": f"{asset.name} ({asset.proxy})",
                "Historical Bias": asset.bias,
                "3M Return": f"{metric.change_3m:+.2f}%",
                "50D Confirmation": (
                    "ABOVE 50d SMA"
                    if metric.current > metric.reference
                    else "BELOW 50d SMA"
                ),
            }
        )
    for play in playbook.plays:
        metric = live["momentum"][play.live_proxy]
        rows.append(
            {
                "Asset": f"{play.name} ({play.live_proxy})",
                "Historical Bias": "HIGH-CONVICTION PLAY",
                "3M Return": f"{metric.change_3m:+.2f}%",
                "50D Confirmation": (
                    "ABOVE 50d SMA"
                    if metric.current > metric.reference
                    else "BELOW 50d SMA"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_research_tab() -> None:
    """Render interactive empirical curves and institutional validation tables."""
    artifacts = load_research_artifacts()
    equity = artifacts["equity"]
    portfolio = artifacts["portfolio"]["portfolio"]
    validation = artifacts["validation"]

    st.subheader("Interactive Portfolio Equity Curves")
    long_frame = equity.melt(
        id_vars="Date",
        var_name="Strategy",
        value_name="Portfolio Value",
    )
    figure = px.line(
        long_frame,
        x="Date",
        y="Portfolio Value",
        color="Strategy",
        labels={"Portfolio Value": "Portfolio Value (USD, $10,000 initial)"},
    )
    figure.update_layout(
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
    )
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("2007–2026 Portfolio Comparison")
    rows = []
    for name, values in portfolio.items():
        rows.append(
            {
                "Portfolio": name,
                "Total Return": f"{values['total_return_pct']:.2f}%",
                "CAGR": f"{values['cagr_pct']:.2f}%",
                "Volatility": f"{values['volatility_pct']:.2f}%",
                "Sharpe": values["sharpe"],
                "Sortino": values["sortino"],
                "Max Drawdown": f"{values['max_drawdown_pct']:.2f}%",
                "Calmar": values["calmar"],
                "Worst Year": f"{values['worst_year_pct']:.2f}%",
                "Best Year": f"{values['best_year_pct']:.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Strategy A vs. B — ATR Stop-Loss Experiment")
    st.dataframe(
        pd.DataFrame(validation["strategy_ab"]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("1976–2026 Blind Out-of-Sample Validation")
    st.dataframe(
        pd.DataFrame(validation["eras"]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Parameter Perturbation Robustness Matrix")
    matrix = pd.DataFrame(validation["robustness"]["matrix"])
    st.dataframe(matrix, use_container_width=True)
    st.metric("Stability Score", validation["robustness"]["stability_score"])
    st.caption(
        "Each matrix cell reports 126-day win rate / median return / valid trades."
    )


def main() -> None:
    """Configure and render the public research application."""
    st.set_page_config(
        page_title="Macro Regime & Quantitative Gold Research Engine",
        page_icon="📈",
        layout="wide",
    )
    st.title("Macro Regime & Quantitative Gold Research Engine")
    st.info(
        "Quantitative macro research tool for educational purposes only — "
        "not financial advice"
    )

    with st.sidebar:
        st.header("Methodology")
        st.markdown(
            "Dual coincident and six-month-forward macro regimes, strict four-factor "
            "Gold inflection signals, point-in-time FRED lags, SMA-confirmed asset "
            "allocation, and anti-overfitting validation across 50 years."
        )
        st.link_button("GitHub Repository", REPOSITORY_URL)
        st.link_button(
            "Support This Research — Buy Me a Coffee",
            "https://buymeacoffee.com",
        )

    live_tab, research_tab = st.tabs(
        [
            "Live Macro & Gold Dashboard",
            "Empirical Backtests & 50-Year Validation",
        ]
    )
    with live_tab:
        render_live_tab()
    with research_tab:
        render_research_tab()


if __name__ == "__main__":
    main()
