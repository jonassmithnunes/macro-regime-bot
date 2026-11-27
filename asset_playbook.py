"""Historical macro asset biases and regime-specific allocation playbooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


OUTPERFORM = "OUTPERFORM"
UNDERPERFORM = "UNDERPERFORM"
NEUTRAL = "NEUTRAL / MIXED"


@dataclass(frozen=True)
class CoreAsset:
    """A core asset class, its live ETF proxy, and historical regime bias."""

    name: str
    proxy: str
    bias: str


@dataclass(frozen=True)
class RegimePlay:
    """A high-conviction regime play and its rationale."""

    name: str
    proxies: Tuple[str, ...]
    rationale: str

    @property
    def live_proxy(self) -> str:
        """Return the primary proxy used for live momentum confirmation."""
        return self.proxies[0]


@dataclass(frozen=True)
class RegimePlaybook:
    """Complete historical allocation map for one forward regime."""

    regime: str
    core_assets: Tuple[CoreAsset, ...]
    plays: Tuple[RegimePlay, ...]


class AssetPlaybook:
    """Immutable macro matrix with live-proxy and transition helpers."""

    GOLDILOCKS = "GOLDILOCKS"
    REFLATION = "REFLATION"
    STAGFLATION = "STAGFLATION"
    DEFLATION = "DEFLATION / RECESSION"

    _CORE_PROXIES = (
        ("Stocks", "SPY"),
        ("Long-Term Bonds", "TLT"),
        ("Cash / T-Bills", "BIL"),
        ("Commodities", "DBC"),
        ("Gold", "GLD"),
        ("Silver", "SLV"),
        ("Copper", "CPER"),
    )
    _BIASES: Mapping[str, Mapping[str, str]] = {
        GOLDILOCKS: {
            "Stocks": OUTPERFORM,
            "Long-Term Bonds": NEUTRAL,
            "Cash / T-Bills": UNDERPERFORM,
            "Commodities": UNDERPERFORM,
            "Gold": NEUTRAL,
            "Silver": OUTPERFORM,
            "Copper": OUTPERFORM,
        },
        REFLATION: {
            "Stocks": OUTPERFORM,
            "Long-Term Bonds": UNDERPERFORM,
            "Cash / T-Bills": UNDERPERFORM,
            "Commodities": OUTPERFORM,
            "Gold": NEUTRAL,
            "Silver": OUTPERFORM,
            "Copper": OUTPERFORM,
        },
        STAGFLATION: {
            "Stocks": UNDERPERFORM,
            "Long-Term Bonds": UNDERPERFORM,
            "Cash / T-Bills": OUTPERFORM,
            "Commodities": OUTPERFORM,
            "Gold": OUTPERFORM,
            "Silver": NEUTRAL,
            "Copper": UNDERPERFORM,
        },
        DEFLATION: {
            "Stocks": UNDERPERFORM,
            "Long-Term Bonds": OUTPERFORM,
            "Cash / T-Bills": OUTPERFORM,
            "Commodities": UNDERPERFORM,
            "Gold": OUTPERFORM,
            "Silver": UNDERPERFORM,
            "Copper": UNDERPERFORM,
        },
    }
    _PLAYS: Mapping[str, Tuple[RegimePlay, ...]] = {
        GOLDILOCKS: (
            RegimePlay(
                "Growth & Tech Equities",
                ("QQQ", "XLK"),
                "Lower discount rates and rising earnings drive multiple expansion.",
            ),
            RegimePlay(
                "Consumer Discretionary & Industrials",
                ("XLY", "XLI"),
                "Expanding real consumer purchasing power supports cyclical demand.",
            ),
            RegimePlay(
                "High-Yield Corporate Bonds",
                ("HYG",),
                "Credit spreads tighten as default risk declines.",
            ),
        ),
        REFLATION: (
            RegimePlay(
                "Energy & Materials Equities",
                ("XLE", "XLB"),
                "Rising input and commodity prices directly support earnings.",
            ),
            RegimePlay(
                "Financials / Banks",
                ("XLF",),
                "Steepening curves and loan growth can boost net interest margins.",
            ),
            RegimePlay(
                "Treasury Inflation-Protected Securities",
                ("TIP",),
                "Principal adjusts upward with CPI while real growth remains firm.",
            ),
        ),
        STAGFLATION: (
            RegimePlay(
                "Managed Futures / CTA Trend-Following",
                ("DBMF", "KMLM"),
                "Can go long commodities and the dollar while shorting bonds "
                "and equities.",
            ),
            RegimePlay(
                "Energy & Defensive Equity",
                ("XLE", "XLP"),
                "Inelastic necessities provide pricing power and resilient demand.",
            ),
            RegimePlay(
                "Short-Duration T-Bills",
                ("SGOV", "BIL"),
                "Captures front-end yields while avoiding duration losses.",
            ),
        ),
        DEFLATION: (
            RegimePlay(
                "Long-Duration US Treasuries",
                ("TLT", "ZROZ"),
                "Duration convexity benefits when growth, inflation, and yields fall.",
            ),
            RegimePlay(
                "Defensive Equity Sectors",
                ("XLU", "XLP", "XLV"),
                "Utilities, staples, and healthcare earnings resist cyclical "
                "contraction.",
            ),
            RegimePlay(
                "US Dollar & Investment-Grade Credit",
                ("UUP", "LQD"),
                "Flight-to-safety demand supports the dollar and high-grade duration.",
            ),
        ),
    }
    _BOOM_TO_BUST_NOTE = (
        "BOOM-TO-BUST INFLECTION: Cash (BIL/SGOV) and Gold (GC=F; live proxy "
        "GLD) often lead early. Long-Term Bonds (TLT) typically trigger their "
        "largest rally only after inflation confirms its peak and yields break "
        "below their 50-day SMA."
    )

    @classmethod
    def normalize_regime(cls, regime: str) -> str:
        """Normalize coincident and forward labels to a canonical regime name."""
        normalized = regime.strip().upper()
        if normalized in {"RECESSION", "DEFLATION", "DEFLATION/RECESSION"}:
            return cls.DEFLATION
        if normalized not in cls._BIASES:
            raise ValueError(f"Unsupported macro regime: {regime}")
        return normalized

    @classmethod
    def for_regime(cls, regime: str) -> RegimePlaybook:
        """Return the complete core-asset and high-conviction regime map."""
        canonical = cls.normalize_regime(regime)
        biases = cls._BIASES[canonical]
        core = tuple(
            CoreAsset(name=name, proxy=proxy, bias=biases[name])
            for name, proxy in cls._CORE_PROXIES
        )
        return RegimePlaybook(canonical, core, cls._PLAYS[canonical])

    @classmethod
    def required_tickers(cls, regime: str) -> Tuple[str, ...]:
        """Return unique core and primary play proxies for market downloads."""
        playbook = cls.for_regime(regime)
        proxies = [asset.proxy for asset in playbook.core_assets]
        proxies.extend(play.live_proxy for play in playbook.plays)
        return tuple(dict.fromkeys(proxies))

    @classmethod
    def transition_note(cls, current: str, forward: str) -> Optional[str]:
        """Return the boom-to-bust warning when that transition is detected."""
        current_name = cls.normalize_regime(current)
        forward_name = cls.normalize_regime(forward)
        if current_name == cls.REFLATION and forward_name == cls.DEFLATION:
            return cls._BOOM_TO_BUST_NOTE
        return None


__all__ = [
    "AssetPlaybook",
    "CoreAsset",
    "NEUTRAL",
    "OUTPERFORM",
    "RegimePlay",
    "RegimePlaybook",
    "UNDERPERFORM",
]
