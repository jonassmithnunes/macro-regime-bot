"""Latest available daily market prices from Yahoo Finance via yfinance."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import MARKET_SYMBOLS


def fetch_market_prices() -> dict[str, dict[str, Any]]:
    """Fetch each symbol independently so one outage does not hide other prices."""
    import yfinance as yf

    quotes: dict[str, dict[str, Any]] = {}
    for symbol, name in MARKET_SYMBOLS.items():
        try:
            history = yf.Ticker(symbol).history(
                period="5d", interval="1d", auto_adjust=False
            )
            if history.empty or "Close" not in history:
                raise ValueError("no daily closing prices returned")
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if close.empty:
                raise ValueError("no numeric closing prices returned")

            quote: dict[str, Any] = {
                "name": name,
                "price": round(float(close.iloc[-1]), 6),
                "as_of": close.index[-1].date().isoformat(),
            }
            if len(close) > 1 and close.iloc[-2] != 0:
                quote["daily_change_pct"] = round(
                    float((close.iloc[-1] / close.iloc[-2] - 1) * 100), 4
                )
            quotes[symbol] = quote
        except Exception as exc:  # yfinance exposes several provider-specific errors
            quotes[symbol] = {"name": name, "error": str(exc)}
    return quotes
