import pandas as pd
import numpy as np
from src.viz import plot_spread_with_signals, plot_equity_and_drawdown

def test_plot_spread_and_equity(tmp_path):
    idx = pd.date_range("2023-01-01", periods=60, freq="D")
    spread = pd.Series(np.sin(np.linspace(0, 6.0, len(idx))), index=idx, name="spread")

    # Minimal signals DataFrame; markers are optional in viz
    signals = pd.DataFrame(index=idx)
    signals["z"] = (spread - spread.mean()) / spread.std(ddof=1)

    p1 = tmp_path / "spread.png"
    plot_spread_with_signals(spread=spread, signals=signals, out_path=str(p1), title="Demo Spread")
    assert p1.exists() and p1.stat().st_size > 0

    # Portfolio frame for equity/drawdown
    equity = pd.Series(100_000 + np.cumsum(np.random.default_rng(0).normal(0, 100, len(idx))), index=idx)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0, np.nan)
    portfolio = pd.DataFrame({"equity": equity, "drawdown": drawdown}, index=idx)

    p2 = tmp_path / "equity.png"
    plot_equity_and_drawdown(portfolio=portfolio, out_path=str(p2), title="Demo Equity")
    assert p2.exists() and p2.stat().st_size > 0
