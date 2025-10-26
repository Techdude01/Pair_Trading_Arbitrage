# StatArb

This repository explores whether two stocks maintain a stable long-term relationship and whether a dollar-neutral pairs trading strategy can profit from temporary deviations around that relationship. The workflow discovers cointegrated pairs using statistical tests, validates the relationship quality, and runs a conservative backtest with realistic transaction costs. While pairs are initially identified based on empirical cointegration, the final trading signals may involve stocks from different industries.

## Quickstart

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install pandas numpy statsmodels matplotlib seaborn yfinance fastparquet python-dotenv jupyterlab ipykernel
```

Tested with Python 3.12/3.13.

## Overview

- Start with daily adjusted prices for a tech‑heavy universe.
- Identify pairs whose log prices are cointegrated (Engle–Granger + ADF on residuals).
- Evaluate quality using half‑life, regression fit (R²), and rolling correlation.
- Trade the log‑spread using a z‑score rule, dollar‑neutral weights, and turnover‑based costs.

## How it works (three parts)

1. Find cointegrated pairs

   - Engle–Granger on log prices with an intercept; residuals tested at 5%.
   - Save the top‑N strongest pairs to `analysis/cointegrated_pairs.pkl`.

2. Analyze and gate

   - Compute half‑life, spread mean/std, rolling correlation, regression R².
   - Filter by sensible thresholds (see `utils/config.get_default_criteria`).

3. Backtest the spread
   - Spread: log(p1) − (alpha + beta·log(p2)); z‑score window ≈ 60d.
   - Entry when |z| > 1.5; exit near 0; execute at t+1 (no lookahead).
   - Dollar‑neutral weights; costs = 5 bps per unit weight turnover.

## Results snapshot (from included notebooks)

- 10 pairs saved; mean Sharpe ≈ 0.73; 10/10 pairs profitable over the sample.
- Best pair (example): V vs ADBE — Sharpe ≈ 1.44, Total Return ≈ 64.6%, Max DD ≈ −15.7%, ~8 trades, win rate ~88.9%.
- Highest absolute return (example): UBER vs SNAP — Total Return ≈ 107%, Sharpe ≈ 1.26.
- Industry similarity doesn't guarantee superior performance; noise and distributional differences can lead to better Sharpe ratios, and proves Cross Sector Pairs can have same/better results.

### 2) Prepare market data

Two options:

- Use the helper notebook: open `data/pullyfinance.ipynb` and run all cells. It will download selected tickers via yfinance and write `data/stock_data.parquet` and `data/stock_data.csv`.
- Or provide your own dataset with the same column convention used here (e.g., `Date`, `Close__AAPL`, `Close__MSFT`, ...).

### 3) Configure the data path

Set the absolute path to your data file (Parquet or CSV) using an environment variable. You can use a `.env` file in the project root.

Create `.env`:

```bash
cat > .env <<'EOF'
stock_data_path=/Users/you/Projects/StatArb/data/stock_data.parquet
EOF
```

Notes:

- The variable name is `stock_data_path` (lowercase), read by `utils/config.py`.
- `.env` is git-ignored.
- Parquet reading uses the `fastparquet` engine.

## How to run

### Notebooks

Launch Jupyter and open the notebooks:

```bash
jupyter lab
```

- `analysis/cointegration.ipynb`
  - Discovers cointegrated pairs (Engle-Granger), saves the top pairs to `analysis/cointegrated_pairs.pkl`, prints summary stats, and can draw a p-value heatmap.
- `signal/signals.ipynb`
  - Loads saved pairs and runs a weight-based pairs backtest with basic turnover costs, plus summary and charts.

Recommended order:

1. Run `data/pullyfinance.ipynb` (first time only) to generate `data/stock_data.parquet`.
2. Run `analysis/cointegration.ipynb` to create/update `analysis/cointegrated_pairs.pkl`.
3. Run `signal/signals.ipynb` to backtest and visualize.

## Project layout

```
analysis/
  cointegration.ipynb         # Discover pairs, save top-N, heatmap
  cointegrated_pairs.pkl      # Saved by the analysis step
  pvalue_heatmap_sorted.png   # Example output image (if generated)

data/
  pullyfinance.ipynb          # Download data and write parquet/csv
  stock_data.parquet          # Market data (git-ignored pattern)
  stock_data.csv              # Market data (git-ignored pattern)

signal/
  signals.ipynb               # Backtest and charts

utils/
  *.py                        # IO, config, stats, plotting, spread, analysis helpers
```

## Assumptions

- Data: daily adjusted prices from yfinance; columns follow `Date`, `Close__{TICKER}` convention.
- Universe: fixed set of large‑cap tech/adjacent tickers defined in `utils/config.get_default_tickers`.
- Cointegration: Engle–Granger on log prices with intercept; residuals tested via ADF at 0.05.
- Hedge ratio: constant over the test horizon (estimated once in analysis, reused in backtest).
- Execution: signals computed on day t, positions executed at t+1 (no lookahead).
- Positioning: dollar‑neutral via weights `w1=pos/(1+|hr|)`, `w2=‑hr*pos/(1+|hr|)`.
- Costs: turnover‑based 5 bps per unit of weight change per day; no borrow/financing fees modeled.
- Signals: z‑score of log‑spread with 60‑day window; typical thresholds entry=1.5, exit=0.0 (configurable).
- Risk/limits: no leverage limits or hard stop‑loss; portfolio capacity and short availability not modeled.

## Roadmap / TODO

- Modularize signals/backtest into `utils`:

  - Move `compute_spread_with_intercept`, `calculate_zscore`, `generate_positions`,
    `backtest_pairs_weights`, and `compute_trade_metrics` from `signal/signals.ipynb`
    to `utils/signals.py` with full type hints and docstrings.
  - Add a `BacktestConfig` dataclass and a single `run_backtest(pair, config)` entrypoint.
  - Write unit tests for signals and PnL math; deterministic fixtures with small synthetic series.
  - Expose a small CLI: `python -m utils.strategy --pairs analysis/cointegrated_pairs.pkl`.

- Strategy refinements:
  - Walk‑forward re‑estimation of hedge ratio/intercept; rolling ADF gating for regime changes.
  - Parameter sweep or Bayesian optimization for entry/exit and z‑window with OOS validation.
  - Volatility targeting and max exposure caps; optional stop‑out and time‑exit.
  - Richer costs: side‑specific fees, borrow rate, slippage model (bps per turnover).
  - Data robustness: outlier winsorization, holiday/calendar alignment, missing‑data handling.
  - Metrics: add Sortino, Calmar, Omega, CAGR, annualized return/vol, drawdown duration,
    VaR/CVaR, profit factor, hit rate, turnover, time‑in‑market; export to CSV.

## Troubleshooting

- If Parquet reading fails, ensure `fastparquet` is installed (or write CSV and set `stock_data_path` to the CSV file).
- Use an absolute path for `stock_data_path`.
- If imports fail inside notebooks, restart kernel and run all cells from the top so `sys.path` is set correctly.
