"""Compounded wealth and annualized portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_returns(returns: pd.Series) -> pd.Series:
    """Require nonempty finite simple returns strictly above total loss."""
    series = pd.Series(returns, dtype=float)
    if series.empty or not np.isfinite(series).all() or (series <= -1).any():
        raise ValueError("Require finite nonempty simple returns > -1.")
    return series


def wealth_index(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Compound returns from a positive finite starting value."""
    series = _validate_returns(returns)
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite.")
    return initial_value * (1 + series).cumprod()


def maximum_drawdown(returns: pd.Series) -> float:
    """Minimum fractional drawdown including the initial wealth peak."""
    wealth = wealth_index(returns)
    peaks = wealth.cummax().clip(lower=1.0)
    return float((wealth / peaks - 1).min())


def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    mar: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """CAGR, sample volatility, Sharpe, Sortino, drawdown, and Calmar.

    Rates are annual effective decimals, converted to per-observation rates.
    CAGR uses observation count, not calendar days. Sortino downside deviation
    includes all observations. Undefined zero-denominator ratios are NaN.
    """
    series = _validate_returns(returns)
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if (
        not np.isfinite([risk_free_rate, mar]).all()
        or min(risk_free_rate, mar) <= -1
    ):
        raise ValueError("Annual rates must be finite and > -1.")
    daily_rf = np.expm1(np.log1p(risk_free_rate) / periods_per_year)
    daily_mar = np.expm1(np.log1p(mar) / periods_per_year)
    log_growth = float(np.log1p(series).sum())
    cagr = float(np.expm1(log_growth * periods_per_year / len(series)))
    daily_vol = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    annual_vol = daily_vol * np.sqrt(periods_per_year)
    downside = np.minimum(series.to_numpy() - daily_mar, 0)
    downside_vol = float(np.sqrt(np.mean(downside**2)))
    mdd = maximum_drawdown(series)
    return {
        "total_return": float(np.expm1(log_growth)),
        "annualized_return_cagr": cagr,
        "annualized_volatility": float(annual_vol),
        "sharpe_ratio": float(
            (series.mean() - daily_rf) / daily_vol * np.sqrt(periods_per_year)
        )
        if daily_vol > 1e-15
        else float("nan"),
        "sortino_ratio": float(
            (series.mean() - daily_mar)
            / downside_vol
            * np.sqrt(periods_per_year)
        )
        if downside_vol > 1e-15
        else float("nan"),
        "maximum_drawdown": mdd,
        "calmar_ratio": cagr / abs(mdd) if mdd < 0 else float("nan"),
    }
