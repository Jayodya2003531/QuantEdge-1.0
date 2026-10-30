"""Historical replay, explicit asset shocks, and volatility scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting import backtest_portfolio
from data import validate_returns, validate_weights
from performance import maximum_drawdown

HISTORICAL_SCENARIOS = {
    "financial_crisis_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
}


def hypothetical_stress(
    weights: pd.Series | dict[str, float],
    asset_shocks: pd.Series | dict[str, float],
    portfolio_value: float = 100_000.0,
) -> pd.Series:
    """Apply simultaneous decimal shocks; omitted assets receive zero shock.

    Interest-rate shocks must first be mapped to asset returns, e.g.
    bond_return ~= -modified_duration * yield_change.
    """
    weights = pd.Series(weights, dtype=float)
    w = validate_weights(weights, weights.index)
    shocks = pd.Series(asset_shocks, dtype=float)
    if shocks.index.has_duplicates or not set(shocks.index).issubset(w.index):
        raise ValueError("Shocks contain duplicate or unknown asset labels.")
    if not np.isfinite(shocks).all() or (shocks < -1).any():
        raise ValueError("Shocks must be finite returns >= -1.")
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive and finite.")
    result = float(w @ shocks.reindex(w.index, fill_value=0))
    return pd.Series(
        {
            "portfolio_return": result,
            "portfolio_pnl": portfolio_value * result,
            "ending_value": portfolio_value * (1 + result),
        }
    )


def historical_stress(
    asset_returns: pd.DataFrame,
    weights: pd.Series | dict[str, float],
    start: str,
    end: str,
    initial_value: float = 100_000.0,
) -> pd.Series:
    """Replay an inclusive historical window as a cost-free buy-and-hold.

    Require data spanning the requested dates, allowing seven calendar days
    for boundary holidays. Partial windows are rejected rather than relabeled.
    Weights selected later are counterfactual exposures, not an ex-ante test.
    """
    frame = validate_returns(asset_returns, minimum_rows=1).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Historical stress requires dates.")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError("start must not follow end.")
    tolerance = pd.Timedelta(days=7)
    window = frame.loc[first:last]
    if (
        window.empty
        or window.index.min() > first + tolerance
        or (window.index.max() < last - tolerance)
    ):
        raise ValueError("Data do not cover the complete stress window.")
    bt = backtest_portfolio(window, weights, initial_value)
    history = bt["history"]
    return pd.Series(
        {
            "start": str(window.index.min().date()),
            "end": str(window.index.max().date()),
            "observations": len(window),
            "period_return": float(bt["summary"]["total_return"]),
            "ending_value": float(bt["summary"]["ending_value"]),
            "maximum_drawdown": maximum_drawdown(history["portfolio_return"]),
        }
    )


def volatility_stress(
    asset_returns: pd.DataFrame, multiplier: float = 2.0
) -> pd.DataFrame:
    """Scale demeaned daily returns, preserving sample means/correlations.

    This is a mechanical sensitivity scenario, not a market forecast.
    Reject scaling that implies asset losses beyond total loss.
    """
    frame = validate_returns(asset_returns)
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("multiplier must be finite and nonnegative.")
    result = frame.mean() + multiplier * (frame - frame.mean())
    return validate_returns(result)
