"""Command-line entry point for the macro regime bot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .config import series_ids
from .fred import FredClient, FredError
from .markets import fetch_market_prices
from .regime import analyze_regime
from .report import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="macro-regime",
        description="Classify the US economy into a four-quadrant macro regime.",
    )
    parser.add_argument(
        "--coincident-series",
        choices=("WEI", "USPHCI"),
        default=os.getenv("COINCIDENT_SERIES", "WEI").upper(),
        help="Coincident growth proxy (default: WEI)",
    )
    parser.add_argument(
        "--start-date",
        default="2000-01-01",
        help="Earliest FRED observation date (default: 2000-01-01)",
    )
    parser.add_argument(
        "--skip-markets",
        action="store_true",
        help="Do not fetch Yahoo Finance market prices",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the JSON report to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        parser.error(
            "FRED_API_KEY is not set. Create a key at "
            "https://fredaccount.stlouisfed.org/apikeys and export it first."
        )

    try:
        client = FredClient(api_key)
        data = client.fetch_many(
            series_ids(args.coincident_series), start_date=args.start_date
        )
        assessment = analyze_regime(data, coincident_series=args.coincident_series)
        markets = None if args.skip_markets else fetch_market_prices()
        report = build_report(data, assessment, markets)
        rendered = json.dumps(report, indent=2, sort_keys=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except (FredError, ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
