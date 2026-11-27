import pandas as pd

from macro_regime_bot.config import FRED_SERIES
from macro_regime_bot.regime import Regime
from macro_regime_bot.report import build_report
from test_regime import full_data


def test_report_contains_all_requested_observations_and_markets() -> None:
    data = full_data()
    for group in FRED_SERIES.values():
        for series_id in group:
            data.setdefault(series_id, pd.Series([1.0], index=[pd.Timestamp("2025-01-01")]))
    from macro_regime_bot.regime import analyze_regime

    result = build_report(
        data,
        analyze_regime(data),
        markets={"SPY": {"price": 500, "as_of": "2025-01-02"}},
    )
    assert result["regime"] == Regime.REFLATION.value
    assert result["observations"]["inflation"]["CPIAUCSL"]["value"] == 112
    assert result["observations"]["credit_spreads"]["BAMLH0A0HYM2"]["value"] == 1
    assert result["markets"]["SPY"]["price"] == 500
