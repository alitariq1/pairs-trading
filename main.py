# Entry point for the pairs-trading research workflow:
#   1) Download & clean prices
#   2) Rank candidate pairs with light realism filters
#   3) Build spread & z-score signals with basic safeguards
#   4) Vectorized backtest with fees/slippage
#   5) Report core performance metrics (incl. turnover)
#
# The orchestration logic stays lean; computational work lives in src/ modules.

from __future__ import annotations
from typing import Sequence

from src.data import (
    download_prices,
    align_prices,
    validate_prices,
    drop_sparse_tickers,
)
from src.pairs import rank_pairs
from src.strategy import compute_spread, zscore, generate_signals
from src.backtest import run_backtest, compute_metrics
from src.viz import plot_spread_with_signals, plot_equity_and_drawdown


# Configuration
TICKERS: Sequence[str] = ("KO", "PEP")
START_DATE: str = "2018-01-01"
END_DATE: str = "2025-01-01"

# Data hygiene
MAX_FFILL_DAYS: int = 5          # cap forward-fill to avoid long imputations
MAX_NAN_FRAC: float = 0.02       # drop tickers with >2% missing after cleaning

# Pair selection filters
MIN_RET_CORR: float | None = 0.10  # log-return correlation prefilter (None to disable)
USE_SAME_SECTOR_ONLY: bool = True  # use simple sector map below

# Signal parameters
ZSCORE_WINDOW: int = 60
Z_ENTRY: float = 2.0
Z_EXIT: float = 0.5
Z_STOP: float | None = 3.0
TIME_STOP_BARS: int | None = 20

# Backtest assumptions
FEE_BPS: float = 1.0
SLIPPAGE_BPS: float = 1.0
START_CAPITAL: float = 100_000.0
DOLLAR_PER_LEG: float = 50_000.0


def _build_sector_map(cols: Sequence[str]) -> dict[str, str]:
    """
    Minimal sector map placeholder.
    For heterogeneous universes, replace with a real sector/industry mapping.
    """
    return {c: "sector" for c in cols}


def main() -> None:
    # 1) Data pipeline
    raw = download_prices(list(TICKERS), start=START_DATE, end=END_DATE)
    prices = align_prices(raw)
    prices = validate_prices(prices, max_ffill_days=MAX_FFILL_DAYS)
    prices = drop_sparse_tickers(prices, max_nan_frac=MAX_NAN_FRAC)

    if prices.empty or len(prices.columns) < 2:
        raise ValueError("Insufficient clean price data to form at least one pair.")

    # 2) Pair ranking
    sector_map = _build_sector_map(prices.columns) if USE_SAME_SECTOR_ONLY else None
    ranked = rank_pairs(
        prices,
        max_pairs=3,
        sector_map=sector_map,
        min_ret_corr=MIN_RET_CORR,
    )
    if ranked.empty:
        raise ValueError("No viable pairs found after screening.")

    # Use top-ranked pair for a single-run demo
    x = str(ranked.loc[0, "x"])
    y = str(ranked.loc[0, "y"])
    beta = float(ranked.loc[0, "beta"])

    # 3) Signals
    spread = compute_spread(prices[x], prices[y], beta)
    z = zscore(spread, window=ZSCORE_WINDOW)
    signals = generate_signals(
        z,
        z_entry=Z_ENTRY,
        z_exit=Z_EXIT,
        z_stop=Z_STOP,
        time_stop_bars=TIME_STOP_BARS,
    )

    # 4) Backtest
    bt = run_backtest(
        prices=prices,
        signals=signals,
        pair=(x, y),
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        capital=START_CAPITAL,
        dollar_per_leg=DOLLAR_PER_LEG,
    )

    # 5) Metrics
    portfolio = bt["portfolio"]
    metrics = compute_metrics(
        equity=portfolio["equity"],
        daily_ret=portfolio["daily_ret"],
        rf=0.0,
        daily_turnover=portfolio.get("daily_turnover"),
    )

    # Console report
    print("===== Pairs Trading Summary =====")
    print(f"Selected pair: {x} vs {y}  |  beta = {beta:.3f}")
    print("\nTop-ranked pairs:")
    print(ranked.to_string(index=False))

    print("\nPerformance metrics:")
    for k in (
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hit_rate",
        "payoff_ratio",
        "annual_turnover",
    ):
        v = metrics.get(k, None)
        if v is not None:
            print(f"{k:16s}: {v:.6f}")

    # Figures (optional)
    try:
        plot_spread_with_signals(
            spread=spread,
            signals=signals,
            out_path="results/figures/spread_signals.png",
        )
        plot_equity_and_drawdown(
            portfolio=portfolio,
            out_path="results/figures/equity_drawdown.png",
        )
        print("\nSaved figures to results/figures/")
    except Exception:
        # Visualization should not block a headless or minimal environment.
        pass

if __name__ == "__main__":
    main()
