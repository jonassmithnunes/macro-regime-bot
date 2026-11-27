import pytest

from asset_playbook import (
    AssetPlaybook,
    NEUTRAL,
    OUTPERFORM,
    UNDERPERFORM,
)


@pytest.mark.parametrize(
    ("regime", "winners", "laggards", "neutral"),
    [
        (
            AssetPlaybook.GOLDILOCKS,
            {"Stocks", "Silver", "Copper"},
            {"Cash / T-Bills", "Commodities"},
            {"Long-Term Bonds", "Gold"},
        ),
        (
            AssetPlaybook.REFLATION,
            {"Stocks", "Commodities", "Silver", "Copper"},
            {"Long-Term Bonds", "Cash / T-Bills"},
            {"Gold"},
        ),
        (
            AssetPlaybook.STAGFLATION,
            {"Cash / T-Bills", "Commodities", "Gold"},
            {"Stocks", "Long-Term Bonds", "Copper"},
            {"Silver"},
        ),
        (
            AssetPlaybook.DEFLATION,
            {"Long-Term Bonds", "Cash / T-Bills", "Gold"},
            {"Stocks", "Commodities", "Silver", "Copper"},
            set(),
        ),
    ],
)
def test_complete_core_asset_matrix(regime, winners, laggards, neutral) -> None:
    assets = AssetPlaybook.for_regime(regime).core_assets
    assert len(assets) == 7
    assert {asset.name for asset in assets if asset.bias == OUTPERFORM} == winners
    assert {asset.name for asset in assets if asset.bias == UNDERPERFORM} == laggards
    assert {asset.name for asset in assets if asset.bias == NEUTRAL} == neutral


@pytest.mark.parametrize(
    ("regime", "expected_proxies"),
    [
        (AssetPlaybook.GOLDILOCKS, {"QQQ", "XLY", "HYG"}),
        (AssetPlaybook.REFLATION, {"XLE", "XLF", "TIP"}),
        (AssetPlaybook.STAGFLATION, {"DBMF", "XLE", "SGOV"}),
        (AssetPlaybook.DEFLATION, {"TLT", "XLU", "UUP"}),
    ],
)
def test_top_three_plays_and_live_proxies(regime, expected_proxies) -> None:
    plays = AssetPlaybook.for_regime(regime).plays
    assert len(plays) == 3
    assert {play.live_proxy for play in plays} == expected_proxies
    assert all(play.rationale for play in plays)


def test_required_tickers_are_unique_and_include_core_and_plays() -> None:
    tickers = AssetPlaybook.required_tickers(AssetPlaybook.DEFLATION)
    assert len(tickers) == len(set(tickers))
    assert {"SPY", "TLT", "BIL", "DBC", "GLD", "SLV", "CPER"}.issubset(tickers)
    assert {"XLU", "UUP"}.issubset(tickers)


def test_coincident_recession_label_maps_to_deflation_playbook() -> None:
    playbook = AssetPlaybook.for_regime("RECESSION")
    assert playbook.regime == AssetPlaybook.DEFLATION


def test_boom_to_bust_transition_note_is_explicit() -> None:
    note = AssetPlaybook.transition_note(
        AssetPlaybook.REFLATION,
        AssetPlaybook.DEFLATION,
    )
    assert note is not None
    assert "Cash" in note
    assert "Gold" in note
    assert "50-day SMA" in note
    assert AssetPlaybook.transition_note(
        AssetPlaybook.GOLDILOCKS,
        AssetPlaybook.DEFLATION,
    ) is None


def test_unknown_regime_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported macro regime"):
        AssetPlaybook.for_regime("UNKNOWN")
