import numpy as np
import pandas as pd
from src.strategy import zscore, generate_signals
from src.backtest import run_backtest, compute_metrics

def _mk_prices_and_signals():
    idx = pd.date_range("2020-01-01", periods=250, freq="B")
    # Two synthetic price series with mild co-movement
    x = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, size=len(idx))), index=idx, name="X")
    y = pd.Series( 98 + np.cumsum(np.random.default_rng(1).normal(0, 1, size=len(idx))), index=idx, name="Y")
    prices = pd.concat([x, y], axis=1)

    # Simple decision series: z of (y - x)
    spread = y - x
    z = zscore(spread, window=30)
    signals = generate_signals(z, z_entry=1.5, z_exit=0.5, z_stop=3.0, time_stop_bars=20)
    return prices, signals

def test_run_backtest_and_metrics():
    prices, signals = _mk_prices_and_signals()
    bt = run_backtest(
        prices=prices,
        signals=signals,
        pair=("X", "Y"),
        fee_bps=1.0,
        slippage_bps=1.0,
        capital=100_000.0,
        dollar_per_leg=50_000.0,
    )
    assert set(bt.keys()) == {"trades", "portfolio"}
    portfolio = bt["portfolio"]
    # Required columns
    for c in ("equity", "daily_ret", "drawdown", "gross_exposure", "daily_turnover"):
        assert c in portfolio.columns
    # Metrics compute without error
    m = compute_metrics(
        equity=portfolio["equity"],
        daily_ret=portfolio["daily_ret"],
        rf=0.0,
        daily_turnover=portfolio["daily_turnover"],
    )
    for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "annual_turnover"):
        assert k in m
