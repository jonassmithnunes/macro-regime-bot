# Research artifacts

Public, non-sensitive empirical snapshots generated from `backtest.py` and `validation_50yr.py`.

- `portfolio_equity.csv`: actual daily adjusted-close portfolio equity curves, normalized to $10,000 at inception.
- `portfolio_summary.json`: published 2007–2026 comparison metrics referenced by the web dashboard and project README.
- `validation_summary.json`: Strategy A/B, blind-era, and perturbation results from the 1976–2026 research run.

These files contain no API keys, Telegram identifiers, alert timestamps, or personal state. Regenerate them from a fresh research run when updating published results; small differences can occur as Yahoo Finance revises adjusted closes.
