"""External data series and report metadata."""

FRED_SERIES = {
    "inflation": {
        "CPIAUCSL": "Headline CPI",
        "CPILFESL": "Core CPI",
        "T5YIE": "Five-year breakeven inflation",
    },
    "growth_and_labor": {
        "GDPC1": "Real GDP",
        "ICSA": "Weekly initial jobless claims",
        "INDPRO": "Industrial production",
    },
    "rates_and_yield_curve": {
        "T10Y2Y": "10-year minus 2-year Treasury spread",
        "DFF": "Effective federal funds rate",
    },
    "consumer_spending": {
        "RSXFS": "Real retail and food services sales",
    },
    "credit_spreads": {
        "BAMLH0A0HYM2": "US high-yield option-adjusted spread",
    },
}

COINCIDENT_SERIES = {
    "WEI": "Weekly Economic Index",
    "USPHCI": "Philadelphia Fed coincident activity index",
}

MARKET_SYMBOLS = {
    "CL=F": "WTI crude oil",
    "HG=F": "Copper",
    "SPY": "S&P 500 ETF",
    "TLT": "20+ year Treasury bond ETF",
    "DX-Y.NYB": "US Dollar Index",
}


def series_ids(coincident_series: str = "WEI") -> list[str]:
    """Return every FRED series required for one report."""
    ids = [series_id for group in FRED_SERIES.values() for series_id in group]
    ids.append(coincident_series)
    return ids
