# Statistical tools for identifying and ranking candidate pairs.
#
# Provides:
#   - corr_matrix:        Pearson correlation of log returns (sanity screen)
#   - save_corr_heatmap:  quick heatmap export for the correlation matrix
#   - engle_granger:      OLS hedge ratio + Engle–Granger cointegration test
#   - estimate_half_life: mean-reversion half-life via AR(1) on spread changes
#   - rank_pairs:         loop over universe, score, and return top pairs
#   - plot_best_spread:   convenience plot of the best-ranked pair’s spread
#
# Notes:
# - This module focuses on light, interpretable statistics suitable for a pairs
#   workflow. It avoids heavy multiple-testing machinery on purpose; that belongs
#   in a separate research pass if you scale the universe materially.

from __future__ import annotations

import os
import itertools
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


# Utilities
def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from an adjusted-close price matrix.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide matrix of prices (Date x Ticker).

    Returns
    -------
    pd.DataFrame
        Log returns with same shape (rows reduced by 1 due to diff).
    """
    return np.log(prices / prices.shift(1)).dropna(how="all")


# Correlation screens (sanity only; correlation != cointegration)
def corr_matrix(adj_close: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation of daily log returns.

    Useful as an initial sanity screen to remove obviously unrelated pairs
    before running more expensive/econometric tests.

    Parameters
    ----------
    adj_close : pd.DataFrame
        Adjusted close matrix.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    logret = _log_returns(adj_close)
    return logret.corr(method="pearson")


def save_corr_heatmap(corr: pd.DataFrame, out_path: str = "results/figures/corr.png") -> None:
    """
    Save a simple heatmap of a correlation matrix.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix.
    out_path : str
        Destination PNG path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation of Log Returns")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Pairwise statistics
def engle_granger(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """
    Engle–Granger two-step cointegration test with OLS hedge ratio.

    Steps
    -----
    1) Regress y ~ const + beta * x  → obtain alpha, beta.
    2) Test for cointegration between x and y (EG test).
       As a cross-check, also run ADF on regression residuals.

    Parameters
    ----------
    x : pd.Series
        Explanatory price series.
    y : pd.Series
        Dependent price series.

    Returns
    -------
    dict
        {
          "alpha": alpha (float),
          "beta":  beta  (float),
          "eg_p":  Engle–Granger p-value,
          "adf_p": ADF p-value on residuals
        }
    """
    xy = pd.concat([x, y], axis=1).dropna()
    X = sm.add_constant(xy.iloc[:, 0])
    yv = xy.iloc[:, 1]

    ols = sm.OLS(yv, X).fit()
    alpha = float(ols.params["const"])
    beta = float(ols.params[xy.columns[0]])

    # EG cointegration test (step 2)
    _, eg_p, _ = coint(yv, xy.iloc[:, 0])

    # ADF on residuals of the OLS regression (stationarity of the spread)
    resid = yv - (alpha + beta * xy.iloc[:, 0])
    adf_p = adfuller(resid.dropna(), regression="c", autolag="AIC")[1]

    return {"alpha": alpha, "beta": beta, "eg_p": float(eg_p), "adf_p": float(adf_p)}


def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life using an AR(1) on spread changes.

    Model
    -----
    Δs_t = a + ρ * s_{t-1} + ε_t
      ⇒ half-life hl = -ln(2)/ρ if ρ < 0 else ∞

    Parameters
    ----------
    spread : pd.Series
        Spread time series (y - beta*x).

    Returns
    -------
    float
        Estimated half-life in trading days (np.inf if not mean-reverting).
    """
    s = spread.dropna()
    if len(s) < 3:
        return np.inf

    s_lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    yv = ds.loc[s_lag.index]
    X = sm.add_constant(s_lag)

    try:
        res = sm.OLS(yv, X).fit()
        rho = float(res.params[s_lag.name])
    except Exception:
        return np.inf

    return float(-np.log(2) / rho) if rho < 0 else float("inf")


# Ranking and convenience plotting
def rank_pairs(
    adj_close: pd.DataFrame,
    max_pairs: int = 5,
    sector_map: dict[str, str] | None = None,
    min_ret_corr: float | None = 0.10,
) -> pd.DataFrame:
    """
    Score and rank unordered pairs from a price matrix.

    For each pair (x, y):
      - OLS: y ~ const + beta*x  → alpha, beta
      - define spread = y - beta*x
      - cointegration checks: Engle–Granger p-value, ADF p-value on residuals
      - estimate half-life (mean reversion speed)
      - score = 0.4*eg_p + 0.4*adf_p + 0.2*(min(half_life, 100)/100)
        (lower is better: favors stronger cointegration and shorter half-life)

    Optional filters:
      - sector_map: keep only same-sector pairs (reduces spurious combinations)
      - min_ret_corr: quick log-return correlation prefilter; set None to disable

    Parameters
    ----------
    adj_close : pd.DataFrame
        Adjusted close matrix (Date x Ticker).
    max_pairs : int
        Number of top pairs to return.
    sector_map : dict[str, str] | None
        Mapping ticker -> sector label. If provided, only same-sector pairs pass.
    min_ret_corr : float | None
        Absolute Pearson correlation threshold on log returns. None disables.

    Returns
    -------
    pd.DataFrame
        Ranked pairs with columns:
        ['x','y','alpha','beta','eg_p','adf_p','half_life','score'].
    """
    cols = list(adj_close.columns)
    rows = []

    for a, b in itertools.combinations(cols, 2):
        # Same-sector screen (optional)
        if sector_map is not None and sector_map.get(a) != sector_map.get(b):
            continue

        x, y = adj_close[a], adj_close[b]

        try:
            # Quick, cheap prefilter on log-return correlation (optional)
            if min_ret_corr is not None:
                lr = np.log(pd.concat([x, y], axis=1)).diff().dropna()
                if lr.shape[0] >= 20:
                    rc = float(lr.corr(method="pearson").iloc[0, 1])
                    if np.isnan(rc) or abs(rc) < min_ret_corr:
                        continue

            stats = engle_granger(x, y)
            spread = (y - stats["beta"] * x).dropna()
            hl = estimate_half_life(spread)

            # Normalize half-life to [0, 100] range for scoring
            hl_cap = min(hl, 100.0)
            score = 0.4 * stats["eg_p"] + 0.4 * stats["adf_p"] + 0.2 * (hl_cap / 100.0)

            rows.append(
                {
                    "x": a,
                    "y": b,
                    "alpha": stats["alpha"],
                    "beta": stats["beta"],
                    "eg_p": stats["eg_p"],
                    "adf_p": stats["adf_p"],
                    "half_life": hl,
                    "score": score,
                }
            )
        except Exception:
            # Skip pathological pairs (e.g., insufficient data for regression/tests)
            continue

    rank_df = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)
    return rank_df.head(max_pairs)


def plot_best_spread(
    adj_close: pd.DataFrame,
    rank_df: pd.DataFrame,
    out_path: str = "results/figures/best_spread.png",
) -> None:
    """
    Plot the spread and ±1σ bands for the best-ranked pair.

    Parameters
    ----------
    adj_close : pd.DataFrame
        Adjusted close matrix used to produce rank_df.
    rank_df : pd.DataFrame
        Output of rank_pairs (top rows are best).
    out_path : str
        Destination PNG path.
    """
    if rank_df.empty:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    best = rank_df.iloc[0]
    x, y = adj_close[best["x"]], adj_close[best["y"]]
    spread = (y - best["beta"] * x).dropna()

    mu, sd = spread.mean(), spread.std(ddof=1)
    z = (spread - mu) / sd if sd > 0 else pd.Series(0.0, index=spread.index)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(spread.index, spread.values, label="Spread")
    ax1.axhline(mu, linestyle="--", linewidth=1, label="Mean")
    ax1.axhline(mu + sd, linestyle=":", linewidth=1, label="+1σ")
    ax1.axhline(mu - sd, linestyle=":", linewidth=1, label="-1σ")
    ax1.set_title(f"Best Spread: {best['y']} - {best['beta']:.3f} × {best['x']}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Spread")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
