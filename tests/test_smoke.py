# tests/test_smoke.py
"""
Simple smoke tests for pairs trading project.
Checks imports, basic spread/signal generation, and metrics keys.
"""

import pandas as pd
import numpy as np

# Imports
import src.data as data
import src.pairs as pairs
import src.strategy as strategy
import src.backtest as backtest
import src.viz as viz


def test_basic_flow():
    # Tiny synthetic data: two correlated price series
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    x = pd.Series(np.linspace(100, 120, 100), index=dates)
    y = 2.0 * x + np.random.normal(0, 1, size=100)

    # Hedge ratio and spread
    beta = strategy.hedge_ratio_ols(x, y)
    spread = strategy.compute_spread(x, y, beta)
    assert isinstance(spread, pd.Series)
    assert not spread.empty

    # Generate signals
    sig = strategy.generate_signals(spread, z_entry=1.0, z_exit=0.5)
    assert isinstance(sig, pd.DataFrame)
    assert set(["spread", "z", "side", "entry_flag", "exit_flag", "w_x", "w_y"]).issubset(sig.columns)

    # Fake prices DataFrame for backtest
    prices = pd.DataFrame({ "X": x, "Y": y })
    res = backtest.run_backtest(prices, sig, pair=("X", "Y"))
    assert "portfolio" in res and "trades" in res

    port = res["portfolio"]
    metrics = backtest.compute_metrics(port["equity"], port["daily_ret"])
    for key in ["cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "hit_rate", "payoff_ratio"]:
        assert key in metrics
