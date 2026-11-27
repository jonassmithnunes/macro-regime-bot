# Macro Regime & Quantitative Gold Research Engine

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/pytest-67_passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-orange.svg)](https://buymeacoffee.com)

An open-source quantitative macro research platform combining **coincident economic classification**, **six-month-forward regime forecasting**, **strict Gold inflection signals**, **cross-asset allocation**, and **institutional-style anti-overfitting validation**.

Built for transparent, reproducible research—not black-box trading signals.

> **Educational/research use only. Nothing in this repository is financial advice.**

## Research Architecture

### 1. Coincident vs. Forward Macro Regimes

| Engine | Objective | Core Inputs |
|---|---|---|
| **Coincident Regime** | Classify the economy using current growth and inflation momentum | Real GDP, Industrial Production, WEI/USPHCI, headline/core CPI |
| **6-Month Forward Regime** | Classify forward growth/inflation using leading market and macro indicators | Copper/Gold, HYG/IEF, permits, jobless claims, breakevens, oil, Gold/SPY, BTC/DXY, yield curve |

Both engines map conditions into **GOLDILOCKS**, **REFLATION**, **STAGFLATION**, or **DEFLATION / RECESSION**, exposing every signal and threshold for inspection.

### 2. Strict Four-Factor Gold Inflection Filter

A Gold surge requires at least three aligned catalysts, mandatory Gold price confirmation, and anti-head-fake buffers:

1. **Real yields:** below the 50-day SMA by 0.03 percentage points for three days, with a negative 10-day change.
2. **US dollar:** at least 0.25% below its 20-day SMA for three days.
3. **Fed pivot pricing:** two-year Treasury yield below its 50-day SMA.
4. **Gold technical trend:** price above its 50-day SMA with positive 10-day ROC—**mandatory**.

The project also evaluates fixed-horizon exits, ATR stops, trailing exits, SMA-confirmed commodity baskets, and BIL-based cash management.

## Published Empirical Results

### 2007–2026 Portfolio Comparison

Adjusted daily closes, $10,000 initial capital, 0.10% round-trip fees, and BIL returns while flat.

| Portfolio | Total Return | CAGR | Volatility | Sharpe | Sortino | Max Drawdown | Worst Year | Best Year |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gold Only + BIL** | **576.02%** | **10.18%** | 16.15% | **0.68** | **0.98** | -30.62% | **-11.11%** | 51.37% |
| Regime Basket + SMA | 187.67% | 5.61% | **12.71%** | 0.49 | 0.68 | **-25.98%** | -11.51% | 30.42% |
| Buy & Hold GLD | 403.30% | 8.54% | 18.27% | 0.54 | 0.76 | -45.56% | -28.33% | 63.68% |
| Buy & Hold DBC | 45.16% | 1.91% | 19.26% | 0.19 | 0.27 | -76.36% | -31.80% | 41.36% |

The diversified basket reduced drawdown, but did **not** improve Sharpe or Calmar relative to Gold plus BIL. Results are reported without cherry-picking a preferred conclusion.

### 50-Year Anti-Overfitting Validation · 1976–2026

The frozen strict rules were evaluated on blind 1976–2006 data, modern 2007+ data, and the combined sample. Gold history is return-stitched from archived LBMA prices and Yahoo futures; CPI uses a 20-business-day lag.

| Horizon | Blind 1976–2006 Win Rate | Modern 2007+ Win Rate | Full-Sample Win Rate | Full-Sample Mean | Full-Sample Median |
|---|---:|---:|---:|---:|---:|
| 21 days | 44.62% | 52.94% | 48.28% | +0.85% | -0.14% |
| 63 days | 51.54% | 57.84% | 54.31% | +2.64% | +1.15% |
| 126 days | 55.38% | **71.29%** | **62.34%** | +5.18% | +3.01% |

**Perturbation stability score: 0.00%** across 15 SMA/confirmation combinations using a 65% six-month win-rate threshold. The modern edge therefore does not establish robust blind-era performance.

| Gold Trend SMA | 2-Day Confirmation | 3-Day Confirmation | 4-Day Confirmation |
|---|---:|---:|---:|
| 40d | 62.85% | 62.40% | 64.38% |
| 45d | 62.45% | 61.70% | 63.72% |
| 50d | 62.40% | **62.34%** | 63.84% |
| 55d | 63.18% | 63.44% | 64.71% |
| 60d | 63.60% | 63.56% | 64.55% |

## Interactive Web Dashboard

The Streamlit application provides:

- Live coincident/forward regime cards and transition path.
- Strict Gold catalyst badges with `ACTIVE`, `PENDING`, and `INACTIVE` states.
- Asset playbook returns and 50-day confirmation tags.
- Interactive Plotly portfolio equity curves.
- Portfolio, stop-loss, blind-era, and robustness research tables.

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, set `FRED_API_KEY` in application secrets to enable the live tab. The empirical tab works from public, non-sensitive research artifacts without credentials or Telegram configuration.

## Quickstart

Python 3.11+ is recommended.

```bash
git clone https://github.com/jonassmithnunes/macro-regime-bot.git
cd macro-regime-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the terminal dashboard, research engines, or test suite:

```bash
export FRED_API_KEY='your_fred_api_key'
python dashboard.py
python backtest.py
python validation_50yr.py
pytest
```

On-demand Telegram report commands:

```bash
python alert_monitor.py --command run
python alert_monitor.py --command regime
python alert_monitor.py --command forecast
python alert_monitor.py --command assets
python alert_monitor.py --command gold
python alert_monitor.py --command macro
python alert_monitor.py --command help
```

## Private GitHub Actions + Telegram Alerts

The public showcase never exposes Telegram credentials or personal alert state. Private automation remains available through encrypted GitHub Secrets:

```dotenv
FRED_API_KEY=your_fred_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

1. Copy `.env.example` to `.env` and fill in your own credentials.
2. Test delivery with `python alert_monitor.py --test-alert`.
3. Add the same three values under **GitHub → Settings → Secrets and variables → Actions**.
4. Run **Actions → Macro Inflection Alerts → Run workflow** and select a report.

Scheduled runs execute Monday–Friday at 15:30 and 20:30 UTC. `actions/cache` restores `data/last_state.json` across scheduled runs; alert state, `.env`, logs, virtual environments, and the entire local `data/` directory are excluded from Git. The workflow needs only `contents: read` permission and never commits personal state to the public branch.

## Repository Map

| File | Responsibility |
|---|---|
| `app.py` | Streamlit live dashboard and interactive research showcase |
| `dashboard.py` | Unified terminal regime, allocation, and Gold report |
| `regime_predictor.py` | Six-month-forward leading-signal classifier |
| `gold_monitor.py` | Live four-factor Gold catalyst monitor |
| `asset_playbook.py` | Historical regime/asset mapping |
| `alert_monitor.py` | Stateful inflection detection and on-demand commands |
| `notifier.py` | Secret-safe private Telegram and local alert delivery |
| `backtest.py` | Signal, portfolio, ATR, and multi-asset simulations |
| `validation_50yr.py` | Blind-era and parameter-perturbation research |
| `research/` | Public empirical curves and result snapshots |
| `tests/` | Offline unit tests for rules, accounting, alerts, and validation |

## Data & Research Methodology

- **FRED / fredapi:** economic activity, inflation, Treasury yields, real yields, permits, claims, and historical CPI.
- **Yahoo Finance / yfinance:** adjusted ETF, futures, FX, commodity, and crypto prices.
- **Point-in-time controls:** explicit macro publication lags, forward-only fills, next-close portfolio entries, and no backfilled market history.
- **Accounting:** adjusted total-return series, turnover/trade fees, BIL cash returns, daily marked-to-market drawdowns, and zero risk-free rate for Sharpe/Sortino comparisons.
- **Research artifacts:** public snapshots contain no credentials, Telegram identifiers, or personal alert state.

Historical results use overlapping observations in several signal studies and latest-vintage FRED history unless explicitly noted. They are not independently executable trading records.

## Testing & Security

```bash
pytest
```

The offline suite covers regime mappings, Gold scoring and strict filters, FRED lag alignment, trade execution, ATR exits, basket rebalancing, Telegram payload formatting, state transitions, stitched data, and robustness matrices. No live API access is required for the tests.

Before publishing, tracked files are scanned for credentials and personal identifiers. Never commit `.env`, real API keys, Telegram chat IDs, `data/`, or log files.

## License & Support

Released under the [MIT License](LICENSE).

If this research is useful, [support future development on Buy Me a Coffee](https://buymeacoffee.com).
