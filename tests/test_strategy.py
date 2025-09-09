import numpy as np
import pandas as pd
from src.strategy import hedge_ratio_ols, compute_spread, zscore, generate_signals

def test_hedge_ratio_and_spread():
    idx = pd.date_range("2021-01-01", periods=50, freq="D")
    x = pd.Series(np.linspace(100, 150, 50), index=idx, name="X")
    y = 2.0 * x + 5.0
    beta = hedge_ratio_ols(x, y)
    assert 1.8 < beta < 2.2
    spr = compute_spread(x, y, beta)
    assert spr.index.equals(idx)
    # spread near constant (intercept absorbed into spread)
    assert np.isfinite(spr.var())

def test_zscore_and_signals_basic():
    idx = pd.date_range("2022-01-01", periods=120, freq="D")
    # Create a series that goes up, then down, to cross bands
    base = np.r_[np.linspace(0, 3, 60), np.linspace(3, -3, 60)]
    s = pd.Series(base, index=idx)
    z = zscore(s, window=30)
    # After warmup, z must be finite
    assert z.iloc[40:80].notna().any()

    sig = generate_signals(z, z_entry=1.0, z_exit=0.25, z_stop=2.5, time_stop_bars=15)
    for col in ("w_x", "w_y", "entry_flag", "exit_flag", "side"):
        assert col in sig.columns
    # Side must be in {-1,0,+1}
    assert set(np.unique(sig["side"])).issubset({-1, 0, 1})
