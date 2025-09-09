pairs-trading
================

A compact, end-to-end research workflow for statistical pairs trading.
It covers data acquisition, pair selection (cointegration), signal generation,
vectorized backtesting with basic costs, performance analytics (incl. turnover),
and publication-ready figures.

-----------------------------------------------------------------------
FEATURES
-----------------------------------------------------------------------
- Data pipeline
  - Yahoo Finance via yfinance (Adjusted Close, Volume)
  - Alignment to a wide price matrix; capped forward-fill and sparse-ticker drop
- Pair selection
  - OLS hedge ratio + Engle–Granger cointegration and ADF on residuals
  - Half-life estimation (AR(1) on the spread)
  - Optional same-sector filter and return-correlation prefilter
- Strategy
  - Spread construction, rolling z-score
  - Band logic with configurable entry/exit, optional z-stop and time stop
- Backtesting
  - Vectorized, next-close execution; linear fees + slippage
  - Equity, drawdowns, exposures, daily/annualized turnover
- Visualization
  - Spread with signals and z-score overlay
  - Equity + drawdown with concise date formatting
- Tests
  - Synthetic, deterministic unit tests (no network)

-----------------------------------------------------------------------
PROJECT STRUCTURE
-----------------------------------------------------------------------
pairs-trading/
  README.md
  pyproject.toml
  requirements.txt
  main.py
  src/
    __init__.py
    data.py        # download, align, validate, drop-sparse, persist
    pairs.py       # correlation, cointegration, half-life, ranking
    strategy.py    # hedge ratio, spread, z-score, signal engine
    backtest.py    # vectorized backtest + metrics (incl. turnover)
    viz.py         # plots with titles and ticker-aware legends
  tests/
    test_data.py
    test_pairs.py
    test_strategy.py
    test_backtest.py
    test_viz.py
    test_smoke.py
  data/
    raw/
    processed/
  results/
    figures/
  outputs/
    csv/

-----------------------------------------------------------------------
REQUIREMENTS
-----------------------------------------------------------------------
- Python 3.10+ (tested on 3.10–3.12)
- See requirements.txt for libraries:
  pandas, numpy, statsmodels, scikit-learn, matplotlib, yfinance, pyarrow

-----------------------------------------------------------------------
INSTALLATION
-----------------------------------------------------------------------
1) Clone
   git clone <your-fork-or-repo-url>.git
   cd pairs-trading

2) Create and activate a virtual environment (example: venv)
   python -m venv .venv
   # macOS/Linux:
   source .venv/bin/activate
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1

3) Install dependencies
   pip install -r requirements.txt

4) Install the package in editable mode so 'src/' is importable
   pip install -e .

-----------------------------------------------------------------------
QUICKSTART
-----------------------------------------------------------------------
python main.py

Outputs:
- Console summary with top-ranked pairs and metrics (CAGR, Sharpe, Sortino,
  max drawdown, annual turnover).
- Figures saved to results/figures/:
  <X>_<Y>_spread_signals.png
  <X>_<Y>_equity_drawdown.png

Default tickers in main.py: ("XLP", "XLY", "XLV", "XLI").
Change these or the date range under the Configuration section in main.py.

-----------------------------------------------------------------------
CONFIGURATION (in main.py)
-----------------------------------------------------------------------
# Universe and dates
TICKERS = ("XLP", "XLY", "XLV", "XLI")
START_DATE = "2018-01-01"
END_DATE   = "2025-01-01"

# Data hygiene
MAX_FFILL_DAYS = 5      # cap forward-fill
MAX_NAN_FRAC   = 0.02   # drop tickers with >2% missing

# Pair selection filters
MIN_RET_CORR = 0.10     # log-return corr prefilter (None to disable)
USE_SAME_SECTOR_ONLY = True

# Signals
ZSCORE_WINDOW   = 60
Z_ENTRY, Z_EXIT = 2.0, 0.5
Z_STOP          = 3.0   # hard stop (abs(z) >= 3)
TIME_STOP_BARS  = 20    # optional bar-based stop

# Backtest
FEE_BPS, SLIPPAGE_BPS = 1.0, 1.0
START_CAPITAL = 100_000.0
DOLLAR_PER_LEG = 50_000.0

The sector filter uses a simple placeholder in main.py:
def _build_sector_map(cols):
    return {c: "sector" for c in cols}  # replace with real sector mapping if desired

-----------------------------------------------------------------------
DATA AND PERSISTENCE
-----------------------------------------------------------------------
- Prices are downloaded with yfinance. Vendor data may have gaps; the pipeline:
  - Caps forward-fill to MAX_FFILL_DAYS
  - Drops tickers with missing fraction > MAX_NAN_FRAC
- Helpers:
  - src/data.py::save_csv(...) — save per-ticker raw CSVs to data/raw/
  - src/data.py::to_processed_csv(...) — save the processed wide matrix to
    data/processed/adj_close.csv

-----------------------------------------------------------------------
METHODOLOGY (BRIEF)
-----------------------------------------------------------------------
- Pair selection
  - Regress y_t = alpha + beta x_t + eps_t (OLS)
  - Engle–Granger cointegration test and ADF on residuals
  - Half-life via AR(1) on spread changes
  - Score combines EG p-value, ADF p-value, and capped half-life; lower is better
  - Optional screens: same sector; absolute log-return correlation threshold
- Signals
  - Construct spread s_t = y_t - beta x_t, compute rolling z-score
  - Enter when |z| crosses Z_ENTRY, exit when |z| returns within Z_EXIT
  - Optional hard z-stop and time stop
  - Unit dollar-neutral weights (scaled in backtest by DOLLAR_PER_LEG)
- Backtest
  - Positions target prior-close; fills happen at next close (daily)
  - Linear fees and slippage (bps)
  - Equity, returns, drawdown, exposures, daily turnover (traded notional / prior equity)
  - Metrics: CAGR, annualized vol, Sharpe, Sortino, max drawdown, hit rate,
    payoff ratio, annual turnover

-----------------------------------------------------------------------
TESTING
-----------------------------------------------------------------------
Deterministic unit tests (no network) live in tests/.

Ensure package is importable:
  pip install -e .

Run tests:
  pytest -q

If you see 'ModuleNotFoundError: No module named "src"', re-run 'pip install -e .'
from the project root (where pyproject.toml lives).

-----------------------------------------------------------------------
FIGURES
-----------------------------------------------------------------------
- Spread & Signals: spread line, mean and ±2σ guides, optional entry/exit markers,
  z-score overlay
- Equity & Drawdown: equity curve with drawdown shaded; concise date formatting

Filenames include tickers for clarity, e.g. XLP_XLY_equity_drawdown.png.

-----------------------------------------------------------------------
ROADMAP (OPTIONAL)
-----------------------------------------------------------------------
- On-disk cache/read of processed price matrix
- Sector/industry mapping from a fundamentals API or static file
- Multi-pair portfolio loop with capital allocation
- Parameter sweep and walk-forward evaluation
- Transaction-cost sensitivity analysis

-----------------------------------------------------------------------
NOTES & DISCLAIMER
-----------------------------------------------------------------------
This repository is for educational and research purposes only.
It is not investment advice and does not constitute a recommendation
to buy or sell any security. Data quality, execution assumptions, and
statistical screening are intentionally simple.

-----------------------------------------------------------------------
ACKNOWLEDGMENTS
-----------------------------------------------------------------------
- yfinance for market data access
- statsmodels for econometrics (OLS, tests)
- pandas, numpy, matplotlib for the core stack

-----------------------------------------------------------------------
LICENSE
-----------------------------------------------------------------------
Choose a license appropriate for your use (e.g., MIT) and add it as LICENSE
at the project root.
