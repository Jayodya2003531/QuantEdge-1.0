"""Self-financing portfolio simulation and out-of-sample VaR exceptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import chi2

from data import validate_returns, validate_weights
from risk_metrics import _clean_returns, _validate_confidence, historical_var


def backtest_portfolio(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series | dict[str, float],
    initial_value: float = 100_000.0,
    rebalance: str = "none",
    transaction_cost_bps: float = 0.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Simulate fixed targets with drifting holdings and net daily returns.

    Trades precede the first return of a new month/quarter, using previous
    closing holdings. Initial purchases incur costs too. A cost of c applies
    to every dollar bought or sold (two-sided traded notional). Post-trade
    wealth solves V_after + c*sum(abs(target*V_after-holdings)) = V_before.
    Fractional holdings and adjusted total-return reinvestment are assumed.
    """
    returns = validate_returns(asset_returns, minimum_rows=1).copy()
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Backtest requires a DatetimeIndex.")
    if returns.index.hasnans or returns.index.tz is not None:
        raise ValueError("Use valid timezone-naive dates.")
    returns = returns.sort_index()
    if (returns <= -1).any().any():
        raise ValueError("Backtesting requires asset returns > -1.")
    target = validate_weights(target_weights, returns.columns).to_numpy()
    if rebalance not in {"none", "monthly", "quarterly"}:
        raise ValueError("rebalance must be none, monthly, or quarterly.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite.")
    if not np.isfinite(transaction_cost_bps) or not (
        0 <= transaction_cost_bps < 10000
    ):
        raise ValueError("Costs must be finite and in [0, 10000) bps.")
    rate = transaction_cost_bps / 10000
    holdings = np.zeros(len(target))
    value = float(initial_value)
    previous_period = None
    history, logs = [], []
    for i, (date, row) in enumerate(returns.iterrows()):
        period = date.to_period("M" if rebalance == "monthly" else "Q")
        trade = i == 0 or (rebalance != "none" and period != previous_period)
        before = value
        cost = 0.0
        if trade:

            def balance(after: float) -> float:
                """Self-financing constraint including both trade sides."""
                return (
                    after
                    + rate * np.abs(target * after - holdings).sum()
                    - before
                )

            after = (
                before
                if rate == 0
                else brentq(
                    balance, 0, before, xtol=max(1e-12, before * 1e-14)
                )
            )
            trades = target * after - holdings
            cost = float(rate * np.abs(trades).sum())
            for asset, amount in zip(returns.columns, trades):
                logs.append(
                    {
                        "date": date,
                        "event": "initial" if i == 0 else "rebalance",
                        "asset": asset,
                        "trade_notional": float(amount),
                        "cost": float(rate * abs(amount)),
                        "turnover": float(abs(amount) / before),
                    }
                )
            holdings = target * after
        holdings *= 1 + row.to_numpy()
        value = float(holdings.sum())
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "Portfolio wealth overflowed or became nonpositive."
            )
        history.append(
            {
                "date": date,
                "portfolio_return": value / before - 1,
                "portfolio_value": value,
                "transaction_cost": cost,
            }
        )
        previous_period = period
    history_frame = pd.DataFrame(history).set_index("date")
    return {
        "history": history_frame,
        "rebalancing_log": pd.DataFrame(logs),
        "summary": pd.Series(
            {
                "initial_value": initial_value,
                "ending_value": value,
                "total_return": value / initial_value - 1,
                "total_cost": float(history_frame.transaction_cost.sum()),
            }
        ),
    }


def rolling_var_backtest(
    portfolio_returns: pd.Series, confidence: float = 0.95, window: int = 250
) -> pd.DataFrame:
    """Forecast historical loss VaR from preceding observations only."""
    _clean_returns(portfolio_returns)
    _validate_confidence(confidence)
    returns = pd.Series(portfolio_returns, dtype=float)
    if (
        returns.index.has_duplicates
        or not returns.index.is_monotonic_increasing
    ):
        raise ValueError("Forecast observations must be uniquely ordered.")
    if not isinstance(window, int) or window < 20 or len(returns) <= window:
        raise ValueError("Need window >=20 and more observations than window.")
    rows = []
    for i in range(window, len(returns)):
        var = historical_var(returns.iloc[i - window : i], confidence)
        realized = float(returns.iloc[i])
        rows.append(
            {
                "date": returns.index[i],
                "var": var,
                "realized_return": realized,
                "exception": realized < -var,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def kupiec_test(
    exceptions: pd.Series, confidence: float = 0.95
) -> dict[str, float | int]:
    """Unconditional coverage likelihood ratio, with stable boundary counts.

    Chi-square p-values are asymptotic and unreliable for short samples.
    This tests frequency, not serial independence of exceptions.
    """
    _validate_confidence(confidence)
    flags = pd.Series(exceptions)
    if (
        flags.empty
        or flags.isna().any()
        or not flags.isin([True, False]).all()
    ):
        raise ValueError("exceptions must be nonempty binary observations.")
    n = len(flags)
    count = int(flags.sum())
    observed = count / n
    expected = 1 - confidence
    null = xlogy(count, expected) + xlogy(n - count, 1 - expected)
    alternative = xlogy(count, observed) + xlogy(n - count, 1 - observed)
    statistic = max(0.0, float(2 * (alternative - null)))
    return {
        "observations": n,
        "exceptions": count,
        "exception_rate": observed,
        "lr_statistic": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
    }
