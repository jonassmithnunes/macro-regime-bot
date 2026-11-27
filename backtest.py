#!/usr/bin/env python3
"""Point-in-time backtests for Gold surge signals and forward macro regimes."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

from asset_playbook import AssetPlaybook, OUTPERFORM, UNDERPERFORM
from regime_predictor import (
    ForwardRegimePredictor,
    PredictorError,
    _socket_timeout,
)


DATA_START = "2007-01-01"
ANALYSIS_START = "2007-03-01"
GOLD_FRED = ("DFII10", "DGS2")
REGIME_FRED = ("PERMIT", "IC4WSA", "T5YIE")
MARKET_TICKERS = (
    "UUP",
    "GLD",
    "HG=F",
    "GC=F",
    "HYG",
    "IEF",
    "CL=F",
    "SPY",
    "TLT",
    "DBC",
    "SLV",
    "BIL",
)
RETURN_ASSETS = ("SPY", "TLT", "GLD", "DBC", "BIL", "UUP")
HORIZONS = (21, 63, 126)
COOLDOWN_DAYS = 21
REGIMES = (
    "GOLDILOCKS",
    "REFLATION",
    "STAGFLATION",
    "DEFLATION / RECESSION",
)
INITIAL_CAPITAL = 10_000.0
ROUND_TRIP_FEES = 0.001
PORTFOLIO_MODERN_START = pd.Timestamp("2007-01-01")
REGIME_BASKETS = {
    "GOLDILOCKS": ("HG=F", "SLV", "GLD"),
    "REFLATION": ("DBC", "HG=F", "SLV", "GLD"),
    "STAGFLATION": ("GLD", "DBC", "UUP"),
    "DEFLATION / RECESSION": ("TLT", "GLD", "UUP"),
}
BASKET_ASSETS = tuple(dict.fromkeys(
    asset for basket in REGIME_BASKETS.values() for asset in basket
))


@dataclass(frozen=True)
class ReturnStats:
    """Forward-return and path-drawdown statistics for one sample."""

    observations: int
    win_rate: float
    mean: float
    median: float
    best: float
    worst: float
    average_max_drawdown: float
    best_date: Optional[pd.Timestamp]
    worst_date: Optional[pd.Timestamp]


@dataclass(frozen=True)
class GoldBacktest:
    """Baseline/strict event dates and comparative statistics by horizon."""

    baseline_dates: pd.DatetimeIndex
    strict_dates: pd.DatetimeIndex
    stats: Mapping[int, Mapping[str, ReturnStats]]


@dataclass(frozen=True)
class RegimeBacktest:
    """Average returns and unfiltered/SMA-confirmed accuracy by regime."""

    mean_returns: pd.DataFrame
    observations: pd.Series
    hit_rates: pd.Series
    overall_hit_rate: float
    unfiltered_predictions: pd.Series
    filtered_predictions: pd.Series
    filtered_hit_rates: pd.Series
    filtered_overall_hit_rate: float


@dataclass(frozen=True)
class Trade:
    """One sequentially executed trade with fees and its exit driver."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    net_return: float
    days_held: int
    exit_reason: str


@dataclass(frozen=True)
class PortfolioResult:
    """Trade distribution and daily portfolio risk/return metrics."""

    trades: Tuple[Trade, ...]
    equity: pd.Series
    total_return: float
    overall_cagr: float
    invested_cagr: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    average_win: float
    average_loss: float
    payoff_ratio: float
    profit_factor: float


@dataclass(frozen=True)
class PortfolioPerformance:
    """Daily-return performance metrics for multi-asset portfolio comparisons."""

    equity: pd.Series
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    worst_calendar_year: float
    best_calendar_year: float


def apply_cooldown(
    crossings: pd.Series,
    cooldown: int = COOLDOWN_DAYS,
) -> pd.DatetimeIndex:
    """Keep crossings at least `cooldown` trading observations apart."""
    positions = np.flatnonzero(crossings.fillna(False).to_numpy(dtype=bool))
    accepted = []
    last_position: Optional[int] = None
    for position in positions:
        if last_position is None or position - last_position >= cooldown:
            accepted.append(position)
            last_position = position
    return crossings.index[accepted]


def compute_strict_gold_filter(
    components: pd.DataFrame,
    real_yield_buffer: pd.Series,
    dollar_buffer: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Require buffered three-day macro signals and mandatory GLD confirmation."""
    real_confirmed = real_yield_buffer.rolling(3).sum() == 3
    dollar_confirmed = dollar_buffer.rolling(3).sum() == 3
    strict_components = components.copy()
    strict_components["real_yields"] = (
        components["real_yields"] & real_confirmed
    )
    strict_components["dollar"] = components["dollar"] & dollar_confirmed
    strict_score = strict_components.sum(axis=1)
    strict_active = (strict_score >= 3) & components["gold_technical"]
    return strict_score, strict_active


def align_fred(
    series: pd.Series,
    calendar: pd.DatetimeIndex,
    lag_business_days: int,
) -> pd.Series:
    """Lag observations, then forward-fill onto a market calendar without backfill."""
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    clean.index = pd.to_datetime(clean.index).normalize()
    clean = clean[~clean.index.duplicated(keep="last")]
    shifted = clean.copy()
    shifted.index = shifted.index + pd.tseries.offsets.BDay(lag_business_days)
    union = calendar.union(shifted.index).sort_values()
    return shifted.reindex(union).ffill().reindex(calendar)


def forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    """Calculate close-to-close forward returns without future price leakage."""
    return prices.shift(-horizon) / prices - 1.0


def summarize_returns(
    returns: pd.Series,
    prices: pd.Series,
    horizon: int,
) -> ReturnStats:
    """Calculate return distribution and average post-entry maximum drawdown."""
    valid = returns.dropna()
    if valid.empty:
        return ReturnStats(
            0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None
        )

    drawdowns = []
    for date in valid.index:
        position = prices.index.get_loc(date)
        path = (
            prices.iloc[position : position + horizon + 1]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(path) < 2:
            continue
        drawdown = path / np.maximum.accumulate(path) - 1.0
        drawdowns.append(float(drawdown.min()))
    average_drawdown = float(np.mean(drawdowns)) if drawdowns else np.nan
    return ReturnStats(
        observations=int(valid.size),
        win_rate=float((valid > 0).mean()),
        mean=float(valid.mean()),
        median=float(valid.median()),
        best=float(valid.max()),
        worst=float(valid.min()),
        average_max_drawdown=average_drawdown,
        best_date=valid.idxmax(),
        worst_date=valid.idxmin(),
    )


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate Wilder-style daily true range followed by a rolling ATR mean."""
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _yfinance_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract one OHLC field from either yfinance MultiIndex orientation."""
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if field in raw.columns.get_level_values(level):
                values = raw.xs(field, axis=1, level=level)
                return values.to_frame() if isinstance(values, pd.Series) else values
    if field in raw:
        return raw[[field]]
    raise PredictorError(f"Yahoo Finance response has no {field} column")


def simulate_portfolio(
    prices: pd.Series,
    signal_dates: Sequence[pd.Timestamp],
    atr: Optional[pd.Series] = None,
    use_atr_exit: bool = False,
    fixed_hold_days: int = 126,
    max_hold_days: int = 252,
) -> PortfolioResult:
    """Simulate sequential next-close entries and daily-close exit rules."""
    clean = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    if len(clean) < 2:
        raise ValueError("At least two price observations are required")
    index = clean.index
    aligned_atr = (
        atr.reindex(index)
        if atr is not None
        else clean.diff().abs().rolling(20).mean()
    )
    sma50 = clean.rolling(50).mean()
    trades = []
    next_allowed = 0
    invested_days = 0

    for signal_date in signal_dates:
        entry_position = int(
            index.searchsorted(pd.Timestamp(signal_date), side="right")
        )
        if entry_position < next_allowed or entry_position >= len(index):
            continue
        entry_date = index[entry_position]
        entry_price = float(clean.iloc[entry_position])
        exit_position = len(index) - 1
        exit_reason = "End of Data"

        if not use_atr_exit:
            scheduled_exit = entry_position + fixed_hold_days
            if scheduled_exit < len(index):
                exit_position = scheduled_exit
                exit_reason = f"{fixed_hold_days}-Day Exit"
        else:
            entry_atr = aligned_atr.iloc[entry_position]
            initial_stop = (
                entry_price - 2.5 * float(entry_atr)
                if pd.notna(entry_atr)
                else entry_price * 0.94
            )
            stop = float(initial_stop)
            highest_close = entry_price
            trailing_active = False
            below_sma_days = 0
            scheduled_exit = entry_position + max_hold_days
            final_position = min(scheduled_exit, len(index) - 1)

            for position in range(entry_position + 1, final_position + 1):
                current = float(clean.iloc[position])
                highest_close = max(highest_close, current)
                current_atr = aligned_atr.iloc[position]
                if highest_close >= entry_price * 1.05 and pd.notna(current_atr):
                    trailing_active = True
                    trailing_stop = highest_close - 3.0 * float(current_atr)
                    stop = max(stop, trailing_stop)

                if pd.notna(sma50.iloc[position]):
                    below_sma_days = (
                        below_sma_days + 1 if current < sma50.iloc[position] else 0
                    )
                if current <= stop:
                    exit_position = position
                    exit_reason = (
                        "Trailing ATR Stop" if trailing_active else "Initial ATR Stop"
                    )
                    break
                if below_sma_days >= 3:
                    exit_position = position
                    exit_reason = "Macro Trend Exit"
                    break
                if position == scheduled_exit:
                    exit_position = position
                    exit_reason = "252-Day Cap"
                    break
            else:
                exit_position = final_position

        exit_date = index[exit_position]
        exit_price = float(clean.iloc[exit_position])
        net_return = (exit_price / entry_price) * (1.0 - ROUND_TRIP_FEES) - 1.0
        days_held = exit_position - entry_position
        trades.append(
            Trade(
                entry_date,
                exit_date,
                entry_price,
                exit_price,
                net_return,
                days_held,
                exit_reason,
            )
        )
        invested_days += days_held
        next_allowed = exit_position + 1

    asset_returns = clean.pct_change(fill_method=None).fillna(0.0)
    daily_returns = pd.Series(0.0, index=index)
    for trade in trades:
        entry_position = int(index.get_loc(trade.entry_date))
        exit_position = int(index.get_loc(trade.exit_date))
        if exit_position > entry_position:
            daily_returns.iloc[entry_position + 1 : exit_position + 1] = (
                asset_returns.iloc[entry_position + 1 : exit_position + 1]
            )
        if exit_position > 0:
            daily_returns.iloc[exit_position] = (
                (1.0 + daily_returns.iloc[exit_position])
                * (1.0 - ROUND_TRIP_FEES)
                - 1.0
            )

    equity = INITIAL_CAPITAL * (1.0 + daily_returns).cumprod()
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0)
    overall_years = len(index) / 252.0
    overall_cagr = (1.0 + total_return) ** (1.0 / overall_years) - 1.0
    invested_years = invested_days / 252.0
    invested_cagr = (
        (1.0 + total_return) ** (1.0 / invested_years) - 1.0
        if invested_years > 0 and total_return > -1.0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    volatility = float(daily_returns.std(ddof=1))
    sharpe = (
        float(np.sqrt(252) * daily_returns.mean() / volatility)
        if volatility > 0
        else np.nan
    )

    net_returns = np.array([trade.net_return for trade in trades], dtype=float)
    winners = net_returns[net_returns > 0]
    losers = net_returns[net_returns <= 0]
    average_win = float(winners.mean()) if winners.size else np.nan
    average_loss = float(losers.mean()) if losers.size else np.nan
    payoff = (
        average_win / abs(average_loss)
        if winners.size and losers.size and average_loss != 0
        else np.nan
    )
    gross_profit = float(winners.sum()) if winners.size else 0.0
    gross_loss = abs(float(losers.sum())) if losers.size else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    return PortfolioResult(
        trades=tuple(trades),
        equity=equity,
        total_return=total_return,
        overall_cagr=float(overall_cagr),
        invested_cagr=float(invested_cagr),
        max_drawdown=float(drawdown.min()),
        sharpe=sharpe,
        win_rate=float((net_returns > 0).mean()) if net_returns.size else np.nan,
        average_win=average_win,
        average_loss=average_loss,
        payoff_ratio=float(payoff),
        profit_factor=float(profit_factor),
    )


class MacroBacktester:
    """Download and empirically evaluate both live signal frameworks."""

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        if not api_key.strip():
            raise PredictorError("FRED_API_KEY is missing")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.prices = pd.DataFrame()
        self.gld_atr: Optional[pd.Series] = None
        self.fred: Dict[str, pd.Series] = {}

    def fetch_data(self) -> None:
        """Fetch full adjusted market history and raw FRED observations."""
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        try:
            raw = yf.download(
                tickers=list(MARKET_TICKERS),
                start=DATA_START,
                end=end.date().isoformat(),
                auto_adjust=True,
                group_by="column",
                progress=False,
                threads=True,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise PredictorError(f"Backtest market download failed: {exc}") from exc
        closes = ForwardRegimePredictor._extract_closes(raw, MARKET_TICKERS)
        self.gld_atr = average_true_range(
            _yfinance_field(raw, "High")["GLD"],
            _yfinance_field(raw, "Low")["GLD"],
            _yfinance_field(raw, "Close")["GLD"],
        )
        closes.index = pd.to_datetime(closes.index)
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)
        closes.index = closes.index.normalize()
        self.prices = closes[~closes.index.duplicated(keep="last")].sort_index()

        client = Fred(api_key=self.api_key)
        try:
            with _socket_timeout(self.timeout):
                for ticker in (*GOLD_FRED, *REGIME_FRED):
                    self.fred[ticker] = client.get_series(
                        ticker,
                        observation_start=DATA_START,
                    )
        except Exception as exc:
            raise PredictorError(f"Backtest FRED download failed: {exc}") from exc

    def market_frame(self) -> pd.DataFrame:
        """Align all prices to the SPY calendar using past values only."""
        if self.prices.empty:
            raise PredictorError("fetch_data() must complete first")
        calendar = self.prices["SPY"].dropna().index
        return self.prices.reindex(calendar).ffill()

    def gold_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reconstruct the exact daily four-factor Gold monitor score."""
        prices = self.market_frame()
        calendar = prices.index
        real_yield = align_fred(self.fred["DFII10"], calendar, 1)
        front_end = align_fred(self.fred["DGS2"], calendar, 1)
        gld = prices["GLD"]
        dollar = prices["UUP"]

        real_sma = real_yield.rolling(50).mean()
        front_sma = front_end.rolling(50).mean()
        dollar_sma = dollar.rolling(20).mean()
        gold_sma = gld.rolling(50).mean()
        real_change = real_yield.diff(10)
        gold_change = gld.pct_change(10, fill_method=None)

        components = pd.DataFrame(
            {
                "real_yields": (real_yield < real_sma) & (real_change < 0),
                "dollar": dollar < dollar_sma,
                "fed_pivot": front_end < front_sma,
                "gold_technical": (gld > gold_sma) & (gold_change > 0),
            },
            index=calendar,
        )
        valid = pd.concat(
            (real_sma, real_change, dollar_sma, front_sma, gold_sma, gold_change),
            axis=1,
        ).notna().all(axis=1)
        score = components.sum(axis=1).astype(float).where(valid)
        real_yield_buffer = real_yield <= real_sma - 0.03
        dollar_buffer = dollar <= dollar_sma * 0.9975
        strict_score, strict_active = compute_strict_gold_filter(
            components,
            real_yield_buffer,
            dollar_buffer,
        )
        signals = pd.DataFrame({"score": score}, index=calendar)
        signals["active"] = score >= 3
        signals["strict_score"] = strict_score.astype(float).where(valid)
        signals["strict_active"] = strict_active & valid
        return signals, prices

    def run_gold_backtest(self) -> GoldBacktest:
        """Compare baseline and strict crossings with identical return samples."""
        signals, prices = self.gold_signals()
        score = signals["score"]
        strict_score = signals["strict_score"]
        analysis = signals.index >= pd.Timestamp(ANALYSIS_START)
        baseline_crossings = (
            (score >= 3)
            & (score.shift(1) < 3)
            & analysis
            & score.notna()
        )
        strict_crossings = (
            signals["strict_active"]
            & ~signals["strict_active"].shift(1, fill_value=False)
            & analysis
            & strict_score.notna()
        )
        baseline_dates = apply_cooldown(baseline_crossings)
        strict_dates = apply_cooldown(strict_crossings)
        gld = prices["GLD"]
        stats: Dict[int, Mapping[str, ReturnStats]] = {}
        for horizon in HORIZONS:
            returns = forward_returns(gld, horizon)
            eligible = analysis & score.notna()
            samples = {
                "Baseline Surge": returns.reindex(baseline_dates).dropna(),
                "Strict Surge": returns.reindex(strict_dates).dropna(),
                "Unconditional": returns.loc[eligible].dropna(),
                "Wait (0/4 or 1/4)": returns.loc[eligible & (score <= 1)].dropna(),
            }
            stats[horizon] = {
                name: summarize_returns(sample, gld, horizon)
                for name, sample in samples.items()
            }
        return GoldBacktest(baseline_dates, strict_dates, stats)

    def regime_signals(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Reconstruct daily forward regimes with macro publication lags."""
        prices = self.market_frame()
        calendar = prices.index
        permits = align_fred(self.fred["PERMIT"], calendar, 20)
        claims = align_fred(self.fred["IC4WSA"], calendar, 5)
        breakeven = align_fred(self.fred["T5YIE"], calendar, 1)

        copper_gold = prices["HG=F"] / prices["GC=F"]
        credit = prices["HYG"] / prices["IEF"]
        gold_equities = prices["GC=F"] / prices["SPY"]
        oil = prices["CL=F"]
        copper_change = copper_gold.pct_change(63, fill_method=None)
        credit_change = credit.pct_change(63, fill_method=None)
        oil_change = oil.pct_change(63, fill_method=None)
        gold_equity_change = gold_equities.pct_change(63, fill_method=None)
        copper_sma = copper_gold.rolling(50).mean()
        credit_sma = credit.rolling(50).mean()
        oil_sma = oil.rolling(50).mean()
        gold_equity_sma = gold_equities.rolling(50).mean()
        permit_sma = permits.rolling(63).mean()
        claims_minimum = claims.rolling(126).min()
        breakeven_change = breakeven.diff(63)

        growth_components = pd.DataFrame(
            {
                "copper_gold": (copper_change > 0) & (copper_gold > copper_sma),
                "credit": (credit_change > 0) & (credit > credit_sma),
                "permits": permits > permit_sma,
                "claims": claims <= claims_minimum * 1.10,
            }
        )
        inflation_components = pd.DataFrame(
            {
                "breakeven": (breakeven_change > 0) & (breakeven > 2.25),
                "oil": (oil_change > 0) & (oil > oil_sma),
                "gold_equities": (gold_equity_change > 0)
                & (gold_equities > gold_equity_sma),
            }
        )
        valid = pd.concat(
            (
                copper_change,
                copper_sma,
                credit_change,
                credit_sma,
                permit_sma,
                claims_minimum,
                breakeven_change,
                oil_change,
                oil_sma,
                gold_equity_change,
                gold_equity_sma,
            ),
            axis=1,
        ).notna().all(axis=1)
        growth_score = growth_components.sum(axis=1)
        inflation_score = inflation_components.sum(axis=1)
        growth_rising = pd.Series(
            np.select(
                (growth_score >= 3, growth_score <= 1),
                (True, False),
                default=copper_change > 0,
            ),
            index=calendar,
        )
        inflation_rising = inflation_score >= 2
        regimes = pd.Series(index=calendar, dtype=object)
        regimes.loc[growth_rising & ~inflation_rising] = "GOLDILOCKS"
        regimes.loc[growth_rising & inflation_rising] = "REFLATION"
        regimes.loc[~growth_rising & inflation_rising] = "STAGFLATION"
        regimes.loc[~growth_rising & ~inflation_rising] = "DEFLATION / RECESSION"
        regimes = regimes.where(valid & (calendar >= pd.Timestamp(ANALYSIS_START)))
        return regimes, prices

    def run_regime_backtest(self) -> RegimeBacktest:
        """Compare raw and SMA-confirmed directional playbook accuracy."""
        regimes, prices = self.regime_signals()
        returns = pd.DataFrame(
            {asset: forward_returns(prices[asset], 63) for asset in RETURN_ASSETS}
        )
        smas = {asset: prices[asset].rolling(50).mean() for asset in RETURN_ASSETS}
        mean_returns = pd.DataFrame(index=REGIMES, columns=RETURN_ASSETS, dtype=float)
        observations = pd.Series(0, index=REGIMES, dtype=int)
        hit_rates = pd.Series(np.nan, index=REGIMES, dtype=float)
        unfiltered_predictions = pd.Series(0, index=REGIMES, dtype=int)
        filtered_predictions = pd.Series(0, index=REGIMES, dtype=int)
        filtered_hit_rates = pd.Series(np.nan, index=REGIMES, dtype=float)
        total_hits = 0
        total_predictions = 0
        total_filtered_hits = 0
        total_filtered_predictions = 0

        for regime in REGIMES:
            sample = returns.loc[regimes == regime]
            observations.loc[regime] = int((regimes == regime).sum())
            mean_returns.loc[regime] = sample.mean()
            playbook = AssetPlaybook.for_regime(regime)
            proxy_names = {asset.proxy: asset.name for asset in playbook.core_assets}
            biases = {asset.name: asset.bias for asset in playbook.core_assets}
            hits = 0
            predictions = 0
            filtered_hits = 0
            filtered_count = 0
            for asset in RETURN_ASSETS:
                if asset == "UUP":
                    bias = OUTPERFORM if regime == AssetPlaybook.DEFLATION else None
                else:
                    bias = biases[proxy_names[asset]]
                if bias not in {OUTPERFORM, UNDERPERFORM}:
                    continue

                asset_returns = sample[asset].dropna()
                if bias == OUTPERFORM:
                    asset_hits = asset_returns > 0
                    confirmed = prices[asset] > smas[asset]
                else:
                    asset_hits = asset_returns < 0
                    confirmed = prices[asset] < smas[asset]
                hits += int(asset_hits.sum())
                predictions += int(asset_hits.size)

                confirmed_returns = sample[asset].where(confirmed).dropna()
                if bias == OUTPERFORM:
                    confirmed_hits = confirmed_returns > 0
                else:
                    confirmed_hits = confirmed_returns < 0
                filtered_hits += int(confirmed_hits.sum())
                filtered_count += int(confirmed_hits.size)

            hit_rates.loc[regime] = hits / predictions if predictions else np.nan
            unfiltered_predictions.loc[regime] = predictions
            filtered_predictions.loc[regime] = filtered_count
            filtered_hit_rates.loc[regime] = (
                filtered_hits / filtered_count if filtered_count else np.nan
            )
            total_hits += hits
            total_predictions += predictions
            total_filtered_hits += filtered_hits
            total_filtered_predictions += filtered_count

        overall = total_hits / total_predictions if total_predictions else np.nan
        filtered_overall = (
            total_filtered_hits / total_filtered_predictions
            if total_filtered_predictions
            else np.nan
        )
        return RegimeBacktest(
            mean_returns,
            observations,
            hit_rates,
            overall,
            unfiltered_predictions,
            filtered_predictions,
            filtered_hit_rates,
            filtered_overall,
        )


def compare_gold_strategies(
    prices: pd.Series,
    signal_dates: Sequence[pd.Timestamp],
    atr: Optional[pd.Series] = None,
) -> Mapping[str, PortfolioResult]:
    """Run fixed-hold and ATR/trailing variants on identical signals and prices."""
    return {
        "A Fixed 126d": simulate_portfolio(
            prices,
            signal_dates,
            use_atr_exit=False,
        ),
        "B ATR + Trail": simulate_portfolio(
            prices,
            signal_dates,
            atr=atr,
            use_atr_exit=True,
        ),
    }


def target_basket_weights(
    regime: str,
    prices: Mapping[str, float],
    smas: Mapping[str, float],
    strict_gold_active: bool,
) -> Dict[str, float]:
    """Allocate normalized winner slots and route every inactive slot to BIL."""
    candidates = REGIME_BASKETS.get(regime, ())
    eligible = {
        asset: asset in candidates and prices[asset] > smas[asset]
        for asset in candidates
    }
    double_gold = (
        strict_gold_active
        and "GLD" in candidates
        and eligible.get("GLD", False)
    )
    denominator = len(candidates) + (1 if double_gold else 0)
    weights = {asset: 0.0 for asset in (*BASKET_ASSETS, "BIL")}
    if not denominator:
        weights["BIL"] = 1.0
        return weights
    for asset in candidates:
        if eligible[asset]:
            units = 2 if asset == "GLD" and double_gold else 1
            weights[asset] = units / denominator
    weights["BIL"] = 1.0 - sum(weights.values())
    return weights


def simulate_regime_basket(
    prices: pd.DataFrame,
    regimes: pd.Series,
    strict_gold_active: pd.Series,
    rebalance_days: int = 21,
    sma_window: int = 50,
) -> pd.Series:
    """Simulate monthly SMA-filtered regime baskets with turnover fees."""
    assets = list((*BASKET_ASSETS, "BIL"))
    clean = prices[assets].sort_index().ffill()
    smas = clean.rolling(sma_window).mean()
    asset_returns = clean.pct_change(fill_method=None).fillna(0.0)
    aligned_regimes = regimes.reindex(clean.index)
    aligned_strict = strict_gold_active.reindex(clean.index).fillna(False)
    valid_regimes = aligned_regimes.dropna()
    if valid_regimes.empty:
        raise ValueError("No valid regimes are available for basket simulation")

    start_date = valid_regimes.index[0]
    weights = {asset: 0.0 for asset in assets}
    weights["BIL"] = 1.0
    daily_returns = pd.Series(index=clean.loc[start_date:].index, dtype=float)
    for step, date in enumerate(daily_returns.index):
        gross_return = sum(
            weights[asset] * float(asset_returns.loc[date, asset])
            for asset in assets
        )
        if step % rebalance_days == 0:
            regime = aligned_regimes.loc[date]
            target = target_basket_weights(
                regime,
                clean.loc[date].to_dict(),
                smas.loc[date].to_dict(),
                bool(aligned_strict.loc[date]),
            )
            turnover = 0.5 * sum(
                abs(target[asset] - weights[asset]) for asset in assets
            )
            gross_return = (
                (1.0 + gross_return) * (1.0 - ROUND_TRIP_FEES * turnover) - 1.0
            )
            weights = target
        daily_returns.loc[date] = gross_return
    return daily_returns


def gold_with_cash_returns(
    gold: pd.Series,
    cash: pd.Series,
    signal_dates: Sequence[pd.Timestamp],
) -> pd.Series:
    """Run fixed-hold Gold trades and earn adjusted BIL returns while flat."""
    result = simulate_portfolio(gold, signal_dates, use_atr_exit=False)
    index = gold.index
    gold_returns = gold.pct_change(fill_method=None).fillna(0.0)
    cash_returns = cash.reindex(index).pct_change(fill_method=None).fillna(0.0)
    daily_returns = cash_returns.copy()
    for trade in result.trades:
        entry_position = int(index.get_loc(trade.entry_date))
        exit_position = int(index.get_loc(trade.exit_date))
        if exit_position > entry_position:
            daily_returns.iloc[entry_position + 1 : exit_position + 1] = (
                gold_returns.iloc[entry_position + 1 : exit_position + 1]
            )
        if exit_position > 0:
            daily_returns.iloc[exit_position] = (
                (1.0 + daily_returns.iloc[exit_position])
                * (1.0 - ROUND_TRIP_FEES)
                - 1.0
            )
    return daily_returns


def performance_from_returns(daily_returns: pd.Series) -> PortfolioPerformance:
    """Calculate total return, risk ratios, drawdown, and calendar-year extremes."""
    returns = daily_returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("Portfolio return series is empty")
    equity = INITIAL_CAPITAL * (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0)
    years = len(returns) / 252.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(252))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if returns.std(ddof=1) > 0
        else np.nan
    )
    downside_deviation = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2) * 252))
    sortino = (
        float(returns.mean() * 252 / downside_deviation)
        if downside_deviation > 0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    calendar_returns = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    return PortfolioPerformance(
        equity=equity,
        total_return=total_return,
        cagr=float(cagr),
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=float(calmar),
        worst_calendar_year=float(calendar_returns.min()),
        best_calendar_year=float(calendar_returns.max()),
    )


def _pct(value: float, digits: int = 2) -> str:
    """Format decimal returns as percentages, preserving missing values."""
    return "N/A" if pd.isna(value) else f"{value * 100:.{digits}f}%"


def generate_report(gold: GoldBacktest, regime: RegimeBacktest) -> str:
    """Render baseline/strict Gold A/B results and regime accuracy tables."""
    baseline_count = len(gold.baseline_dates)
    strict_count = len(gold.strict_dates)
    noise_reduction = (
        1.0 - strict_count / baseline_count if baseline_count else np.nan
    )
    lines = [
        "=" * 112,
        f"MACRO SIGNAL BACKTEST | {ANALYSIS_START} to "
        f"{datetime.now().date().isoformat()}",
        "Adjusted closes; FRED lagged before forward-filling; "
        "overlapping forward returns.",
        "=" * 112,
        "",
        "GOLD BOUNCE MONITOR — BASELINE VS STRICT FILTER A/B",
        f"Baseline Trigger Events ({COOLDOWN_DAYS}d cooldown): {baseline_count}",
        f"Strict Trigger Events ({COOLDOWN_DAYS}d cooldown): {strict_count}",
        f"Signal Noise Reduction: {_pct(noise_reduction)}",
    ]
    for horizon in HORIZONS:
        lines.extend(
            (
                "",
                f"Forward Horizon: {horizon} trading days",
                "Sample                  N    Win Rate      Mean    Median  "
                "    Best     Worst   Avg Max DD",
                "-" * 96,
            )
        )
        for sample_name, stats in gold.stats[horizon].items():
            lines.append(
                f"{sample_name:<22} {stats.observations:>4}  "
                f"{_pct(stats.win_rate):>10}  {_pct(stats.mean):>8}  "
                f"{_pct(stats.median):>9}  {_pct(stats.best):>8}  "
                f"{_pct(stats.worst):>8}  {_pct(stats.average_max_drawdown):>10}"
            )
        for sample_name in ("Baseline Surge", "Strict Surge"):
            event_stats = gold.stats[horizon][sample_name]
            if event_stats.observations:
                lines.append(
                    f"{sample_name} Best/Worst Dates: "
                    f"{event_stats.best_date.date()} / "
                    f"{event_stats.worst_date.date()}"
                )

    lines.extend(
        (
            "",
            "=" * 112,
            "6-MONTH FORWARD REGIME PREDICTOR — AVERAGE FORWARD 63D RETURNS",
            "=" * 112,
            "Regime                       N       SPY       TLT       GLD   "
            "    DBC       BIL       UUP",
            "-" * 96,
        )
    )
    for name in REGIMES:
        returns = "  ".join(
            f"{_pct(regime.mean_returns.loc[name, asset]):>8}"
            for asset in RETURN_ASSETS
        )
        lines.append(
            f"{name:<24} {regime.observations.loc[name]:>5}  {returns}"
        )

    lines.extend(
        (
            "",
            "FORWARD REGIME DIRECTIONAL ACCURACY — RAW VS SMA-CONFIRMED",
            "Regime                    Unfiltered N   Hit Rate   "
            "SMA-Confirmed N   Hit Rate",
            "-" * 88,
        )
    )
    for name in REGIMES:
        lines.append(
            f"{name:<24} {regime.unfiltered_predictions.loc[name]:>13}  "
            f"{_pct(regime.hit_rates.loc[name]):>10}  "
            f"{regime.filtered_predictions.loc[name]:>16}  "
            f"{_pct(regime.filtered_hit_rates.loc[name]):>10}"
        )
    lines.extend(
        (
            "-" * 88,
            f"Overall Unfiltered Hit Rate: {_pct(regime.overall_hit_rate)}",
            f"Overall SMA-Confirmed Hit Rate: "
            f"{_pct(regime.filtered_overall_hit_rate)}",
            "Hit = winner return > 0 or laggard return < 0; confirmed samples "
            "also require the matching 50d SMA trend.",
            "Gold max drawdown is measured from entry through each forward "
            "horizon; returns are not annualized.",
        )
    )
    return "\n".join(lines)


def generate_portfolio_report(
    results: Mapping[str, Mapping[str, PortfolioResult]],
) -> str:
    """Render the modern/full-history head-to-head portfolio simulation."""
    lines = [
        "",
        "=" * 156,
        "STRICT GOLD PORTFOLIO SIMULATION — $10,000 INITIAL CAPITAL / "
        "0.10% ROUND-TRIP FEES",
        "Next-business-day close entry; sequential all-in trades; 0% return on cash.",
        "=" * 156,
        "Era / Strategy             Trades   Win Rate    Avg Win   Avg Loss  "
        " Payoff   Profit Factor   Total Return   CAGR Overall   CAGR Invested  "
        " Max Drawdown   Sharpe",
        "-" * 156,
    ]
    for era, strategies in results.items():
        for strategy, result in strategies.items():
            lines.append(
                f"{era + ' / ' + strategy:<25} {len(result.trades):>6}  "
                f"{_pct(result.win_rate):>9}  {_pct(result.average_win):>9}  "
                f"{_pct(result.average_loss):>9}  {result.payoff_ratio:>6.2f}  "
                f"{result.profit_factor:>13.2f}  {_pct(result.total_return):>13}  "
                f"{_pct(result.overall_cagr):>12}  {_pct(result.invested_cagr):>13}  "
                f"{_pct(result.max_drawdown):>12}  {result.sharpe:>6.2f}"
            )

    for era, strategies in results.items():
        baseline = strategies["A Fixed 126d"]
        atr = strategies["B ATR + Trail"]
        sharpe_improves = atr.sharpe > baseline.sharpe
        drawdown_improves = abs(atr.max_drawdown) < abs(baseline.max_drawdown)
        if sharpe_improves and drawdown_improves:
            verdict = "IMPROVES"
        elif not sharpe_improves and not drawdown_improves:
            verdict = "DOES NOT IMPROVE"
        else:
            verdict = "MIXED / DOES NOT CLEARLY IMPROVE"
        lines.append(
            f"{era} Verdict: Strategy B {verdict} risk-adjusted returns "
            f"(Sharpe {baseline.sharpe:.2f} -> {atr.sharpe:.2f}; "
            f"MaxDD {_pct(baseline.max_drawdown)} -> {_pct(atr.max_drawdown)})."
        )
    lines.append(
        "50-year ATR uses a 20d mean absolute close-change proxy; "
        "modern GLD uses true ATR."
    )
    return "\n".join(lines)


def generate_multi_asset_report(
    performances: Mapping[str, PortfolioPerformance],
) -> str:
    """Render Gold-only, regime-basket, and buy-and-hold performance."""
    lines = [
        "",
        "=" * 142,
        f"MULTI-ASSET PORTFOLIO COMPARISON — {ANALYSIS_START} TO PRESENT / "
        f"$10,000 / 0.10% FEES",
        "=" * 142,
        "Portfolio                       Total Return      CAGR   Volatility   "
        "Sharpe   Sortino   Max Drawdown   Calmar   Worst Year   Best Year",
        "-" * 142,
    ]
    for name, result in performances.items():
        lines.append(
            f"{name:<28} {_pct(result.total_return):>13}  {_pct(result.cagr):>7}  "
            f"{_pct(result.annual_volatility):>10}  {result.sharpe:>7.2f}  "
            f"{result.sortino:>7.2f}  {_pct(result.max_drawdown):>12}  "
            f"{result.calmar:>6.2f}  {_pct(result.worst_calendar_year):>10}  "
            f"{_pct(result.best_calendar_year):>9}"
        )

    gold = performances["Gold Only + BIL"]
    basket = performances["Regime Basket + SMA"]
    sharpe_improves = basket.sharpe > gold.sharpe
    calmar_improves = basket.calmar > gold.calmar
    drawdown_improves = abs(basket.max_drawdown) < abs(gold.max_drawdown)
    if sharpe_improves and calmar_improves and drawdown_improves:
        verdict = "IMPROVES"
    elif not sharpe_improves and not calmar_improves and not drawdown_improves:
        verdict = "DOES NOT IMPROVE"
    else:
        verdict = "MIXED / DOES NOT CLEARLY IMPROVE"
    lines.extend(
        (
            "-" * 142,
            f"Verdict: Regime-Matched Basket {verdict} risk-adjusted returns "
            f"(Sharpe {gold.sharpe:.2f} -> {basket.sharpe:.2f}; "
            f"Calmar {gold.calmar:.2f} -> {basket.calmar:.2f}; "
            f"MaxDD {_pct(gold.max_drawdown)} -> {_pct(basket.max_drawdown)}).",
        )
    )
    return "\n".join(lines)


def main() -> int:
    """Run both full-history backtests and print empirical results."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print("Error: set FRED_API_KEY before running backtest.py", file=sys.stderr)
        return 1
    try:
        backtester = MacroBacktester(api_key)
        backtester.fetch_data()
        gold = backtester.run_gold_backtest()
        regime = backtester.run_regime_backtest()
        print(generate_report(gold, regime))

        signals, modern_market = backtester.gold_signals()
        modern_crossings = (
            signals["strict_active"]
            & ~signals["strict_active"].shift(1, fill_value=False)
            & (signals.index >= PORTFOLIO_MODERN_START)
            & signals["strict_score"].notna()
        )
        modern_events = apply_cooldown(modern_crossings)
        modern_prices = modern_market["GLD"].loc[PORTFOLIO_MODERN_START:]
        modern_results = compare_gold_strategies(
            modern_prices,
            modern_events,
            atr=backtester.gld_atr,
        )

        from validation_50yr import (
            ANALYSIS_START as LONG_ANALYSIS_START,
            FiftyYearValidation,
            strict_event_dates,
        )

        long_validation = FiftyYearValidation(api_key)
        long_validation.fetch_data()
        long_prices = long_validation.frame["gold"].loc[LONG_ANALYSIS_START:]
        long_events = strict_event_dates(long_validation.frame)
        long_results = compare_gold_strategies(long_prices, long_events)
        portfolio_results = {
            "Modern 2007+": modern_results,
            "Full 50-Year": long_results,
        }
        print(generate_portfolio_report(portfolio_results))

        modern_gold = modern_market["GLD"].loc[ANALYSIS_START:]
        modern_bil = modern_market["BIL"].reindex(modern_gold.index)
        gold_cash_returns = gold_with_cash_returns(
            modern_gold,
            modern_bil,
            gold.strict_dates,
        )
        regime_series, basket_market = backtester.regime_signals()
        basket_returns = simulate_regime_basket(
            basket_market,
            regime_series,
            signals["strict_active"],
        )
        buy_gold_returns = modern_gold.pct_change(fill_method=None).fillna(0.0)
        buy_dbc = basket_market["DBC"].loc[ANALYSIS_START:]
        buy_dbc_returns = buy_dbc.pct_change(fill_method=None).fillna(0.0)
        for buy_returns in (buy_gold_returns, buy_dbc_returns):
            buy_returns.iloc[-1] = (
                (1.0 + buy_returns.iloc[-1]) * (1.0 - ROUND_TRIP_FEES) - 1.0
            )
        performances = {
            "Gold Only + BIL": performance_from_returns(gold_cash_returns),
            "Regime Basket + SMA": performance_from_returns(basket_returns),
            "Buy & Hold GLD": performance_from_returns(buy_gold_returns),
            "Buy & Hold DBC": performance_from_returns(buy_dbc_returns),
        }
        print(generate_multi_asset_report(performances))
    except (PredictorError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
