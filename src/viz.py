# Visualization utilities for the pairs-trading workflow.
#
# Provides:
#   - plot_spread_with_signals: spread line with mean/±2σ guides, optional
#       z-score overlay on a secondary axis, and optional entry/exit markers.
#   - plot_equity_and_drawdown: equity curve with drawdown shaded on a
#       secondary axis.
#
# Notes:
# - Functions save figures to disk and close them (non-interactive).
# - Optional `title` lets callers include tickers/beta in a recruiter-friendly caption.
# - Output directories are created if they do not exist.
# - Uses concise date formatting for multi-year horizons.

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Spread & signals plot
def plot_spread_with_signals(
    spread: pd.Series,
    signals: pd.DataFrame,
    out_path: str,
    title: str | None = None,
) -> None:
    """
    Plot the spread series with optional z-score overlay and trading markers.

    The primary axis (left) shows the raw spread with horizontal guides at the
    mean and ±2 standard deviations. If a finite z-score can be computed, the
    secondary axis (right) overlays the z-score. If `signals` contains any of
    {"long_entry","long_exit","short_entry","short_exit"}, markers are placed
    at those dates on the spread line.

    Parameters
    ----------
    spread : pd.Series
        Spread series (typically y - beta*x), indexed by datetime.
    signals : pd.DataFrame
        Index aligned to `spread`. Optional boolean/{0,1} columns:
        - long_entry, long_exit, short_entry, short_exit
        Missing columns are ignored.
    out_path : str
        Destination PNG path. Parent directories will be created if missing.
    title : str, optional
        Figure title (e.g., "Spread & Signals | XLY vs XLP — beta=0.870").

    Returns
    -------
    None
        The plot is saved to `out_path` and the figure is closed.
    """
    # Clean series and compute z-score for overlay/reference
    s = spread.dropna()
    z = (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else pd.Series(index=s.index)

    # Ensure output directory exists (also handle bare filenames)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Base figure and primary axis
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(s.index, s.values, label="Spread")

    # Mean and ±2σ reference lines
    mu, sd = s.mean(), s.std(ddof=1)
    ax1.axhline(mu, color="black", linestyle="--", alpha=0.7)
    ax1.axhline(mu + 2 * sd, color="red", linestyle="--", alpha=0.5)
    ax1.axhline(mu - 2 * sd, color="red", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Spread")

    # Optional z-score overlay
    if z.notna().any():
        ax2 = ax1.twinx()
        ax2.plot(z.index, z.values, alpha=0.4, label="z-score")
        ax2.set_ylabel("z")

    # Optional signal markers
    if "long_entry" in signals:
        ax1.plot(
            signals.index[signals["long_entry"]],
            s.loc[signals["long_entry"]],
            "^",
            markersize=6,
            color="green",
            label="Long Entry",
        )
    if "long_exit" in signals:
        ax1.plot(
            signals.index[signals["long_exit"]],
            s.loc[signals["long_exit"]],
            "v",
            markersize=6,
            color="darkgreen",
            label="Long Exit",
        )
    if "short_entry" in signals:
        ax1.plot(
            signals.index[signals["short_entry"]],
            s.loc[signals["short_entry"]],
            "v",
            markersize=6,
            color="red",
            label="Short Entry",
        )
    if "short_exit" in signals:
        ax1.plot(
            signals.index[signals["short_exit"]],
            s.loc[signals["short_exit"]],
            "^",
            markersize=6,
            color="darkred",
            label="Short Exit",
        )

    # Merge legends across axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    if z.notna().any():
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles, labels = handles1 + handles2, labels1 + labels2
    else:
        handles, labels = handles1, labels1
    plt.legend(handles, labels, loc="upper left")

    # Title and concise date ticks
    if title:
        plt.title(title)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    # Save and close
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Equity & drawdown plot
def plot_equity_and_drawdown(
    portfolio: pd.DataFrame,
    out_path: str,
    title: str | None = None,
) -> None:
    """
    Plot equity with drawdown shaded on a secondary axis.

    The primary axis (left) shows the equity curve. The secondary axis (right)
    shades the drawdown series for quick inspection of underwater periods.

    Parameters
    ----------
    portfolio : pd.DataFrame
        Must include 'equity' and 'drawdown' columns indexed by datetime.
    out_path : str
        Destination PNG path. Parent directories will be created if missing.
    title : str, optional
        Figure title (e.g., "Equity & Drawdown | XLY vs XLP — beta=0.870").

    Returns
    -------
    None
        The plot is saved to `out_path` and the figure is closed.
    """
    df = portfolio.dropna(subset=["equity", "drawdown"])

    # Ensure output directory exists (also handle bare filenames)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Base figure and primary axis (equity)
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(df.index, df["equity"].values, label="Equity")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity")

    # Secondary axis (drawdown shading)
    ax2 = ax1.twinx()
    ax2.fill_between(df.index, df["drawdown"].values, 0.0, alpha=0.3, label="Drawdown")
    ax2.set_ylabel("Drawdown")

    # Merge legends across axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    # Title and concise date ticks
    if title:
        plt.title(title)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    # Save and close
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
