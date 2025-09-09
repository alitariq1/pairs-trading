# Strategy utilities for pairs trading.
#
# Provides:
#   - hedge_ratio_ols: estimate hedge ratio beta via OLS
#   - compute_spread:  construct spread = y - beta * x
#   - zscore:          rolling z-score for mean-reversion decisions
#   - generate_signals:entry/exit logic using z-bands, hard stop, time stop
#
# Notes
# -----
# - The signal engine is agnostic to whether you pass raw spread or a
#   pre-computed z-score series. Many workflows compute z outside and feed it in.
# - Weights are dollar-neutral unit magnitudes (+1 / -1) suitable for a
#   vectorized backtest that scales by target dollars per leg.

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------
# Hedge ratio and spread construction
# ---------------------------------------------------------------------
def hedge_ratio_ols(x: pd.Series, y: pd.Series) -> float:
    """
    Estimate hedge ratio (beta) by regressing y on x with an intercept.

    Parameters
    ----------
    x : pd.Series
        Explanatory price series.
    y : pd.Series
        Dependent price series.

    Returns
    -------
    float
        Estimated hedge ratio beta.

    Raises
    ------
    ValueError
        If inputs are None or insufficient length after alignment.
    """
    if x is None or y is None:
        raise ValueError("Input series cannot be None.")
    xy = pd.concat([x, y], axis=1).dropna()
    if xy.shape[0] < 2:
        raise ValueError("Not enough aligned observations for regression.")
    X = sm.add_constant(xy.iloc[:, 0])
    yv = xy.iloc[:, 1]
    model = sm.OLS(yv, X).fit()
    return float(model.params[xy.columns[0]])


def compute_spread(x: pd.Series, y: pd.Series, beta: float) -> pd.Series:
    """
    Compute spread = y - beta * x, aligned on common index.

    Parameters
    ----------
    x : pd.Series
        Explanatory price series.
    y : pd.Series
        Dependent price series.
    beta : float
        Hedge ratio.

    Returns
    -------
    pd.Series
        Spread series aligned to x/y with NaNs dropped.

    Raises
    ------
    TypeError
        If beta is not numeric.
    """
    if not isinstance(beta, (int, float)):
        raise TypeError("Beta must be numeric.")
    spread = y - beta * x
    return spread.dropna()


# ---------------------------------------------------------------------
# Rolling z-score
# ---------------------------------------------------------------------
def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute a rolling z-score using a custom rolling apply.

    The value at time t is (x_t - mean_{t-window+1..t}) / std_{t-window+1..t}.
    Returns NaN when there is insufficient data or zero variance.

    Parameters
    ----------
    series : pd.Series
        Input series (e.g., spread).
    window : int
        Rolling window length.

    Returns
    -------
    pd.Series
        Rolling z-score with the same index as input, NaN until window fills.
    """
    s = pd.Series(series, dtype="float64").replace([np.inf, -np.inf], np.nan)

    def _last_z(arr: np.ndarray) -> float:
        a = arr[np.isfinite(arr)]
        if a.size < window:
            return np.nan
        mu = a.mean()
        sd = a.std(ddof=1)
        if sd <= 0:
            return np.nan
        return (a[-1] - mu) / sd

    return s.rolling(window=window, min_periods=window).apply(_last_z, raw=True)


# ---------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------
def generate_signals(
    spread: pd.Series,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: Optional[float] = None,
    time_stop_bars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Turn a one-dimensional decision series into trading signals.

    By default the series is treated as the z-score of the spread; callers
    often pass the z-score directly. The engine applies symmetric entry/exit
    bands, an optional hard z-stop, and an optional time stop. Positions are
    encoded as unit, dollar-neutral weights per leg (+1 / -1).

    Parameters
    ----------
    spread : pd.Series
        Decision series (commonly the z-score of the spread). If you pass the
        raw spread, consider computing z separately for better scaling.
    z_entry : float
        Enter when z >= +z_entry (short spread) or z <= -z_entry (long spread).
    z_exit : float
        Exit when |z| <= z_exit while in a position.
    z_stop : float, optional
        Hard stop: exit immediately if |z| >= z_stop while in a position.
        None disables the hard stop.
    time_stop_bars : int, optional
        Close the position if it has been open this many bars. None disables.

    Returns
    -------
    pd.DataFrame
        Index = Date. Columns:
          - spread : the input series values (for traceability)
          - z      : same as spread (caller may overwrite downstream)
          - side   : {-1, 0, +1} where +1 = long y/short x, -1 = short y/long x
          - entry_flag, exit_flag : {0, 1} event markers
          - w_x, w_y : unit-magnitude dollar-neutral weights per leg
    """
    s = pd.Series(spread, dtype="float64")
    z = s.copy()                     # keep a copy; caller may pass z already
    z_dec = z.fillna(0.0)            # decision series with NaNs treated as flat

    side = np.zeros(len(s), dtype=int)        # current position state over time
    entry_flag = np.zeros(len(s), dtype=int)  # 1 when we open a position
    exit_flag = np.zeros(len(s), dtype=int)   # 1 when we close a position

    current = 0           # -1, 0, +1
    bars_in_trade = 0     # for optional time stop

    for i, zi in enumerate(z_dec.values):
        # Hard stop takes precedence if in a position
        if current != 0 and z_stop is not None and abs(zi) >= z_stop:
            current = 0
            exit_flag[i] = 1
            bars_in_trade = 0

        # Exit band while in a position
        elif current != 0 and abs(zi) <= z_exit:
            current = 0
            exit_flag[i] = 1
            bars_in_trade = 0

        # Entries from flat
        elif current == 0 and zi >= z_entry:
            current = -1            # short y / long x (short spread)
            entry_flag[i] = 1
            bars_in_trade = 0

        elif current == 0 and zi <= -z_entry:
            current = +1            # long y / short x (long spread)
            entry_flag[i] = 1
            bars_in_trade = 0

        # Optional time stop while in a position
        if current != 0 and time_stop_bars is not None:
            bars_in_trade += 1
            if bars_in_trade >= int(time_stop_bars):
                current = 0
                exit_flag[i] = 1
                bars_in_trade = 0

        side[i] = current

    # Map side to unit, dollar-neutral leg weights
    # Convention: w_y = side, w_x = -side  (so net is ~0 when scaled equally)
    w_y = np.where(side == +1, +1.0, np.where(side == -1, -1.0, 0.0))
    w_x = -w_y

    out = pd.DataFrame(
        {
            "spread": s.values,
            "z": z.values,
            "side": side,
            "entry_flag": entry_flag,
            "exit_flag": exit_flag,
            "w_x": w_x,
            "w_y": w_y,
        },
        index=s.index,
    )
    out.index.name = "Date"
    return out
