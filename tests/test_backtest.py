import pandas as pd
import pytest

from backtest import (
    align_fred,
    apply_cooldown,
    compute_strict_gold_filter,
    forward_returns,
    simulate_portfolio,
    simulate_regime_basket,
    summarize_returns,
    target_basket_weights,
)


def test_cooldown_enforces_twenty_one_trading_day_spacing() -> None:
    index = pd.bdate_range("2024-01-01", periods=50)
    crossings = pd.Series(False, index=index)
    crossings.iloc[[0, 10, 21, 22, 42]] = True
    events = apply_cooldown(crossings)
    assert list(events) == [index[0], index[21], index[42]]


def test_fred_alignment_lags_before_forward_filling_without_backfill() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=5)
    raw = pd.Series(
        [10.0, 20.0],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
    )
    aligned = align_fred(raw, calendar, lag_business_days=1)
    assert pd.isna(aligned.iloc[0])
    assert aligned.iloc[1] == 10.0
    assert aligned.iloc[2] == 20.0
    assert aligned.iloc[3] == 20.0


def test_forward_returns_and_drawdown_use_only_post_entry_path() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.Series([100.0, 120.0, 60.0], index=index)
    returns = forward_returns(prices, 1)
    assert returns.iloc[0] == pytest.approx(0.20)
    assert returns.iloc[1] == pytest.approx(-0.50)
    assert pd.isna(returns.iloc[2])

    stats = summarize_returns(returns.iloc[[0]], prices, horizon=2)
    assert stats.win_rate == 1.0
    assert stats.average_max_drawdown == pytest.approx(-0.50)


def test_strict_gold_filter_requires_buffers_and_price_confirmation() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    components = pd.DataFrame(
        {
            "real_yields": True,
            "dollar": True,
            "fed_pivot": False,
            "gold_technical": True,
        },
        index=index,
    )
    real_buffer = pd.Series(True, index=index)
    dollar_buffer = pd.Series([False, True, True, True, True], index=index)

    score, active = compute_strict_gold_filter(
        components,
        real_buffer,
        dollar_buffer,
    )
    assert score.iloc[2] == 2
    assert not active.iloc[2]
    assert score.iloc[3] == 3
    assert active.iloc[3]

    without_price = components.copy()
    without_price["gold_technical"] = False
    _, price_filtered = compute_strict_gold_filter(
        without_price,
        real_buffer,
        dollar_buffer,
    )
    assert not price_filtered.any()

    interrupted_real_buffer = real_buffer.copy()
    interrupted_real_buffer.iloc[3] = False
    _, real_filtered = compute_strict_gold_filter(
        components,
        interrupted_real_buffer,
        dollar_buffer,
    )
    assert not real_filtered.iloc[3]


def test_fixed_hold_simulator_enters_next_close_and_charges_fees() -> None:
    index = pd.bdate_range("2024-01-01", periods=130)
    prices = pd.Series([100.0 + value for value in range(130)], index=index)
    result = simulate_portfolio(
        prices,
        [index[0]],
        fixed_hold_days=126,
    )

    trade = result.trades[0]
    assert trade.entry_date == index[1]
    assert trade.exit_date == index[127]
    assert trade.days_held == 126
    assert trade.exit_reason == "126-Day Exit"
    expected_return = (227.0 / 101.0) * 0.999 - 1.0
    assert trade.net_return == pytest.approx(expected_return)
    assert result.total_return == pytest.approx(expected_return)


def test_initial_atr_stop_cuts_a_false_breakout() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.Series([99.0, 100.0, 96.0, 110.0, 111.0, 112.0], index=index)
    atr = pd.Series(1.0, index=index)

    result = simulate_portfolio(prices, [index[0]], atr=atr, use_atr_exit=True)

    trade = result.trades[0]
    assert trade.entry_price == 100.0
    assert trade.exit_date == index[2]
    assert trade.exit_reason == "Initial ATR Stop"
    assert trade.net_return == pytest.approx(0.96 * 0.999 - 1.0)


def test_trailing_atr_stop_locks_profits_without_moving_down() -> None:
    index = pd.bdate_range("2024-01-01", periods=7)
    prices = pd.Series([99.0, 100.0, 106.0, 105.0, 101.0, 110.0, 111.0], index=index)
    atr = pd.Series(1.0, index=index)

    result = simulate_portfolio(prices, [index[0]], atr=atr, use_atr_exit=True)

    trade = result.trades[0]
    assert trade.exit_date == index[4]
    assert trade.exit_reason == "Trailing ATR Stop"
    assert trade.net_return == pytest.approx(1.01 * 0.999 - 1.0)


def test_basket_weights_double_gold_and_route_inactive_slots_to_bil() -> None:
    assets = ("HG=F", "SLV", "GLD", "DBC", "UUP", "TLT", "BIL")
    prices = {asset: 100.0 for asset in assets}
    smas = {asset: 100.0 for asset in assets}
    prices.update({"HG=F": 110.0, "GLD": 110.0})

    weights = target_basket_weights("GOLDILOCKS", prices, smas, True)

    assert weights["HG=F"] == pytest.approx(0.25)
    assert weights["GLD"] == pytest.approx(0.50)
    assert weights["SLV"] == 0.0
    assert weights["BIL"] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0)

    defensive = target_basket_weights(
        "DEFLATION / RECESSION",
        prices,
        smas,
        False,
    )
    assert defensive["TLT"] == 0.0
    assert defensive["GLD"] == pytest.approx(1.0 / 3.0)
    assert defensive["BIL"] == pytest.approx(2.0 / 3.0)


def test_basket_rebalancer_applies_new_weights_and_turnover_fees() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    assets = ("HG=F", "SLV", "GLD", "DBC", "UUP", "TLT", "BIL")
    prices = pd.DataFrame(100.0, index=index, columns=assets)
    prices["HG=F"] = [100.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    regimes = pd.Series("GOLDILOCKS", index=index)
    strict = pd.Series(False, index=index)

    returns = simulate_regime_basket(
        prices,
        regimes,
        strict,
        rebalance_days=2,
        sma_window=2,
    )

    assert returns.iloc[2] == pytest.approx(-0.001 / 3.0)
    assert returns.iloc[3] == pytest.approx((102.0 / 101.0 - 1.0) / 3.0)
