# Backtesting utilities for pairs trading strategies.
#
# Provides:
#   - run_backtest: vectorized simulation of trading a single pair
#   - compute_metrics: summary performance statistics
#
# Assumptions:
#   - Trades execute at the close of the signal day + 1
#   - Linear fees and slippage in basis points
#   - Signals contain dollar weights for each leg (long/short)
#   - Strategy is market-neutral (dollar_per_leg exposure each side)

from __future__ import annotations
import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    pair: tuple[str, str],
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    capital: float = 100_000.0,
    dollar_per_leg: float = 50_000.0,
) -> dict[str, pd.DataFrame]:
    """
    Vectorized backtest for a single pair of tickers.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted Close prices with Date index and ticker columns.
    signals : pd.DataFrame
        Must contain 'w_x' and 'w_y' columns (weights per leg).
        Positive = long, negative = short, unit magnitude.
    pair : (str, str)
        Tickers (x, y) to trade.
    fee_bps : float
        One-way transaction fee in basis points.
    slippage_bps : float
        One-way slippage in basis points.
    capital : float
        Starting cash.
    dollar_per_leg : float
        Target dollar exposure per leg.

    Returns
    -------
    dict
        {
          "trades":   executed trades table
          "portfolio":time series of equity, PnL, exposures, drawdown, turnover
        }
    """
    x, y = pair
    px = prices[[x, y]].dropna().copy()
    sig = signals.reindex(px.index).fillna(0.0)

    # Target dollar exposure per leg at time t
    tgt_val = pd.DataFrame(
        {x: sig["w_x"] * dollar_per_leg, y: sig["w_y"] * dollar_per_leg},
        index=px.index,
    )

    # Convert dollar exposures into quantities using prior close
    prior_close = px.shift(1)
    tgt_qty = tgt_val.divide(prior_close, axis=0)

    # Effective positions (shift by 1 day)
    qty = tgt_qty.shift(1).fillna(0.0)

    # Trades = day-over-day change in position (executed at today's close)
    dqty = qty.diff().fillna(qty)
    trade_price = px

    # Trade notional and costs
    notional = dqty * trade_price
    fees = notional.abs() * (fee_bps / 1e4)
    slippage = notional.abs() * (slippage_bps / 1e4)
    cash_flow = -(notional + fees + slippage).sum(axis=1)

    # Cash account and equity curve
    cash = capital + cash_flow.cumsum()
    pos_val = (qty * px).sum(axis=1)
    equity = (cash + pos_val).astype(float)

    # Daily returns
    prev = equity.shift(1)
    daily_ret = ((equity - prev) / prev).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Drawdowns
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0, np.nan)

    # Exposures
    gross = (qty.abs() * px).sum(axis=1)
    net = (qty * px).sum(axis=1)

    # Daily turnover = traded notional / prior-day equity
    traded_notional = notional.abs().sum(axis=1)
    equity_prev = pd.Series(capital, index=px.index)
    equity_prev.update(equity.shift(1))
    daily_turnover = (traded_notional / equity_prev.replace(0, np.nan)).fillna(0.0)

    # Portfolio time series
    portfolio = pd.DataFrame(
        {
            "cash": cash,
            "pos_value": pos_val,
            "equity": equity,
            "daily_ret": daily_ret,
            "gross_exposure": gross,
            "net_exposure": net,
            "drawdown": drawdown,
            "daily_turnover": daily_turnover,
        },
        index=px.index,
    )
    portfolio.index.name = "Date"

    # Trade blotter: one row per fill
    tx_list = []
    for col in [x, y]:
        nonzero = dqty[col].ne(0)
        if nonzero.any():
            df = pd.DataFrame(
                {
                    "ticker": col,
                    "side": np.sign(dqty[col].loc[nonzero]).astype(int),  # +1=buy, -1=sell
                    "qty": dqty[col].loc[nonzero].values,
                    "price": trade_price[col].loc[nonzero].values,
                    "notional": notional[col].loc[nonzero].values,
                    "fees": fees[col].loc[nonzero].values,
                    "slippage": slippage[col].loc[nonzero].values,
                },
                index=dqty.index[nonzero],
            )
            tx_list.append(df)

    trades = (
        pd.concat(tx_list).sort_index()
        if tx_list
        else pd.DataFrame(
            columns=["ticker", "side", "qty", "price", "notional", "fees", "slippage"]
        )
    )
    trades.index.name = "Date"

    return {"trades": trades, "portfolio": portfolio}


def _annualized_turnover(daily_turnover: pd.Series | None) -> float:
    """
    Annualize daily turnover = average daily turnover × 252.
    """
    if daily_turnover is None or len(daily_turnover) == 0:
        return 0.0
    dt = pd.Series(daily_turnover).replace([np.inf, -np.inf], np.nan).dropna()
    return float(252.0 * dt.mean()) if len(dt) > 0 else 0.0


def compute_metrics(
    equity: pd.Series,
    daily_ret: pd.Series,
    rf: float = 0.0,
    daily_turnover: pd.Series | None = None,
) -> dict:
    """
    Compute common performance metrics.

    Parameters
    ----------
    equity : pd.Series
        Portfolio equity curve.
    daily_ret : pd.Series
        Daily returns aligned with equity.
    rf : float
        Risk-free annual rate.
    daily_turnover : pd.Series, optional
        Daily turnover series for turnover metric.

    Returns
    -------
    dict
        Metrics: CAGR, volatility, Sharpe, Sortino, max drawdown,
        hit rate, payoff ratio, annual turnover.
    """
    eq = equity.dropna().astype(float)
    r = daily_ret.reindex(eq.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    n = max(len(eq), 1)
    if n <= 1 or eq.iloc[0] <= 0:
        cagr = 0.0
    else:
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / n) - 1.0

    # Excess returns
    r_ex = r - (rf / 252.0)
    sd = r.std(ddof=1)
    ann_vol = float(sd * np.sqrt(252.0)) if sd > 0 else 0.0
    sharpe = float(r_ex.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0

    downside = r[r < 0]
    dd = downside.std(ddof=1)
    sortino = float(r_ex.mean() / dd * np.sqrt(252.0)) if dd > 0 else 0.0

    run_max = eq.cummax()
    dd_series = (eq - run_max) / run_max.replace(0, np.nan)
    max_dd = float(dd_series.min()) if not dd_series.empty else 0.0

    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    payoff = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    return {
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "hit_rate": float(win_rate),
        "payoff_ratio": float(payoff),
        "annual_turnover": _annualized_turnover(daily_turnover),
    }
