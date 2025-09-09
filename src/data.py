# Data access and preparation utilities for pairs trading.
#
# Responsibilities
# ----------------
# - Robust daily price download for a small universe of tickers (Adj Close, Volume)
# - Convenience helpers to save raw CSVs and quick-plot for sanity checks
# - Alignment into a wide price matrix with a clean Date index
# - Lightweight validation/cleaning with capped forward-fill
# - Optional filter to drop sparse tickers
# - Persist processed matrix to disk
#
# Notes
# -----
# - Uses yfinance for convenience. APIs can change; callers should cache outputs.
# - All dates are tz-naive pandas Timestamps.
# - Keep transformations minimal and explicit; avoid “magical” data fabrication.

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


# Internal helpers
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    If yfinance returns a MultiIndex for columns, drop the top level
    (e.g., ('Adj Close', 'AAPL') -> 'Adj Close') or join levels with a space.
    """
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.droplevel(0, axis=1)
        except Exception:
            # Fallback: join all levels as strings
            df.columns = [" ".join(map(str, c)).strip() for c in df.columns]
    return df


# Download / Save / Quick Plot
def download_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Robust daily downloader for ETFs/stocks using yfinance.
    Tries multiple endpoints and column variants.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ticker -> DataFrame with columns:
        - 'Adj Close' (float)
        - 'Volume' (float, may be NaN if unavailable)
        Index is tz-naive Date.
    """

    def _extract(df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        df = _flatten_cols(df)

        # Normalize column name lookup (case-insensitive)
        cl = {str(c).strip().lower(): c for c in df.columns}

        # Prefer 'Adj Close', otherwise fall back to 'Close'
        adj = None
        if "adj close" in cl:
            adj = df[cl["adj close"]].astype(float).rename("Adj Close")
        elif "close" in cl:
            adj = df[cl["close"]].astype(float).rename("Adj Close")

        vol = (
            df[cl["volume"]].astype(float).rename("Volume")
            if "volume" in cl
            else pd.Series(index=df.index, dtype="float64", name="Volume")
        )

        if adj is None:
            return None

        out = pd.concat([adj, vol], axis=1)
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out.index.name = "Date"
        return out

    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        # Attempt 1: per-ticker history(auto_adjust=False) → expect 'Adj Close'
        df = _extract(yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=False))

        # Attempt 2: per-ticker history(auto_adjust=True) → use 'Close' as adjusted
        if df is None:
            df = _extract(yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True))

        # Attempt 3: batch download(auto_adjust=False)
        if df is None:
            df = _extract(
                yf.download(
                    t,
                    start=start,
                    end=end,
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    group_by="column",
                    threads=True,
                )
            )

        # Attempt 4: batch download(auto_adjust=True)
        if df is None:
            df = _extract(
                yf.download(
                    t,
                    start=start,
                    end=end,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    group_by="column",
                    threads=True,
                )
            )

        if df is None or df.empty:
            raise ValueError(f"No usable price data for {t}")

        out[t] = df

    return out


def save_csv(data: Dict[str, pd.DataFrame], out_dir: str = "data/raw") -> None:
    """
    Save each ticker's DataFrame to CSV under out_dir/<TICKER>.csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    for t, df in data.items():
        path = os.path.join(out_dir, f"{t}.csv")
        df.index.name = "Date"
        df.to_csv(path)


def quick_plot_close(data: Dict[str, pd.DataFrame], out_path: str = "results/figures/raw_close.png") -> None:
    """
    Quick line plot of adjusted closes for a sanity check after download.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for t, df in data.items():
        if "Adj Close" in df.columns:
            plt.plot(df.index, df["Adj Close"], label=t)
    plt.title("Adjusted Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Alignment / Validation / Save
def align_prices(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a wide DataFrame of adjusted close prices.
    Columns = tickers, Index = Date (tz-naive, sorted).
    """
    cols = []
    for ticker, df in data.items():
        s = df["Adj Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = ticker
        cols.append(s)
    wide = pd.concat(cols, axis=1).sort_index()
    wide.index.name = "Date"
    return wide


def validate_prices(df: pd.DataFrame, max_ffill_days: int = 5) -> pd.DataFrame:
    """
    Lightweight cleaning for a wide adjusted-close matrix.

    Steps
    -----
    - Coerce numerics; drop ±inf
    - Drop rows where *all* tickers are non-positive
    - Forward-fill within columns, but only up to `max_ffill_days`
    - Drop initial rows until all columns have data

    Parameters
    ----------
    df : pd.DataFrame
        Wide price matrix (Date x Ticker).
    max_ffill_days : int
        Maximum consecutive days to forward-fill within a column.

    Returns
    -------
    pd.DataFrame
        Cleaned matrix with Date index.
    """
    clean = df.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([float("inf"), float("-inf")], pd.NA)

    # Remove rows where every ticker is non-positive (bad vendor rows)
    mask_bad = (clean <= 0).all(axis=1)
    clean = clean.loc[~mask_bad]

    # Cap forward-fill to avoid fabricating long stretches of data
    clean = clean.ffill(limit=max_ffill_days)

    # Trim to first date where all columns have data
    valid_row_mask = clean.notna().all(axis=1)
    if valid_row_mask.any():
        first_full_idx = valid_row_mask.idxmax()
        clean = clean.loc[first_full_idx:]
    else:
        return clean.iloc[0:0]

    clean.index.name = "Date"
    return clean


def drop_sparse_tickers(df: pd.DataFrame, max_nan_frac: float = 0.02) -> pd.DataFrame:
    """
    Drop columns (tickers) with too many NaNs after cleaning/alignment.

    Parameters
    ----------
    df : pd.DataFrame
        Wide price matrix.
    max_nan_frac : float
        Maximum allowed fraction of missing values per ticker (e.g., 0.02 = 2%).

    Returns
    -------
    pd.DataFrame
        Matrix restricted to tickers passing the missingness filter.
    """
    if df.empty:
        return df
    frac = df.isna().mean()
    keep = frac[frac <= max_nan_frac].index
    return df[keep]


def to_processed_csv(df: pd.DataFrame, path: str = "data/processed/adj_close.csv") -> None:
    """
    Persist the processed adjusted-close matrix to CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(path)