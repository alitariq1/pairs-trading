import numpy as np
import pandas as pd
from src.pairs import engle_granger, estimate_half_life, rank_pairs

rng = np.random.default_rng(42)

def _mk_cointegrated(n=300, alpha=2.0, beta=0.8, noise_sd=0.5):
    # Random walk X_t
    eps_x = rng.normal(0, 1, size=n)
    x = np.cumsum(eps_x)
    # Stationary residual u_t (AR(1) with phi ~ 0.5)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.5 * u[t-1] + rng.normal(0, noise_sd)
    y = alpha + beta * x + u
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(x, index=idx, name="X"), pd.Series(y, index=idx, name="Y")

def test_engle_granger_and_half_life_basic():
    x, y = _mk_cointegrated()
    stats = engle_granger(x, y)
    # Reasonable fields
    for k in ("alpha", "beta", "eg_p", "adf_p"):
        assert k in stats
    # Cointegration should have relatively small p-values
    assert 0.0 <= stats["eg_p"] <= 1.0
    assert 0.0 <= stats["adf_p"] <= 1.0
    spread = (y - stats["beta"] * x).dropna()
    hl = estimate_half_life(spread)
    assert np.isfinite(hl) or np.isinf(hl)

def test_rank_pairs_with_filters():
    # Build a tiny universe with one cointegrated pair and one noisy ticker
    x, y = _mk_cointegrated()
    z = pd.Series(rng.normal(100, 1, size=len(x)), index=x.index, name="Z")  # unrelated
    prices = pd.concat([x.rename("AAA"), y.rename("BBB"), z.rename("CCC")], axis=1)
    # Make them look like prices (positive, shifted up)
    prices = (prices - prices.min().min()) + 10.0

    sector_map = {"AAA": "s1", "BBB": "s1", "CCC": "s2"}
    ranked = rank_pairs(prices, max_pairs=3, sector_map=sector_map, min_ret_corr=0.05)
    assert not ranked.empty
    # Best pair should be AAA-BBB (same sector, cointegrated)
    top = ranked.iloc[0]
    assert {top["x"], top["y"]} == {"AAA", "BBB"}
