"""Long-only constrained sample mean/covariance portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import validate_returns


class OptimizationError(RuntimeError):
    """Optimization failed or returned an infeasible numerical result."""


@dataclass(frozen=True)
class OptimizationResult:
    """Target allocation and annualized in-sample model diagnostics."""

    weights: pd.Series
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    objective: str
    success: bool
    message: str


def estimate_covariance(
    returns: pd.DataFrame, periods_per_year: int = 252
) -> pd.DataFrame:
    """Estimate annual sample covariance from complete aligned returns."""
    aligned = validate_returns(returns)
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    return aligned.cov() * periods_per_year


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = "min_variance",
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    periods_per_year: int = 252,
) -> OptimizationResult:
    """Minimize variance or maximize arithmetic annual excess-return Sharpe.

    Use training data only. SLSQP enforces sum(weights)=1 and box bounds.
    Max-Sharpe uses deterministic multistarts; no global optimum is promised.
    risk_free_rate is annual arithmetic for this mean/covariance model.
    """
    aligned = validate_returns(returns)
    if objective not in {"min_variance", "max_sharpe"}:
        raise ValueError("Unknown objective.")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and > -1.")
    n_assets = aligned.shape[1]
    if not 0 < max_weight <= 1 or n_assets * max_weight < 1 - 1e-12:
        raise ValueError("Weight cap cannot fund a fully invested portfolio.")
    cov = estimate_covariance(aligned, periods_per_year).to_numpy()
    mu = aligned.mean().to_numpy() * periods_per_year
    initial = np.full(n_assets, 1 / n_assets)
    scale = max(float(np.max(np.diag(cov))), 1e-12)
    if objective == "max_sharpe" and np.max(np.diag(cov)) <= 1e-24:
        raise OptimizationError("Sharpe is undefined for zero variance data.")

    def loss(weights: np.ndarray) -> float:
        """Evaluate a scaled variance or negative Sharpe objective."""
        variance = max(float(weights @ cov @ weights), 0.0)
        if objective == "min_variance":
            return variance / scale
        if variance <= 1e-24:
            return 1e12
        return -float((weights @ mu - risk_free_rate) / np.sqrt(variance))

    starts = [initial]
    if objective == "max_sharpe" and n_assets > 1:
        for i in range(n_assets):
            candidate = np.full(n_assets, (1 - max_weight) / (n_assets - 1))
            candidate[i] = max_weight
            if candidate.max() <= max_weight + 1e-12:
                starts.append(candidate)
    results = [
        minimize(
            loss,
            start,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n_assets,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        for start in starts
    ]
    valid = [
        r
        for r in results
        if r.success
        and np.isfinite(r.x).all()
        and abs(r.x.sum() - 1) <= 1e-8
        and r.x.min() >= -1e-8
        and r.x.max() <= max_weight + 1e-8
        and np.isfinite(r.fun)
    ]
    if not valid:
        raise OptimizationError(str(results[0].message))
    result = min(valid, key=lambda r: r.fun)
    weights = result.x
    volatility = float(np.sqrt(max(float(weights @ cov @ weights), 0)))
    if objective == "max_sharpe" and volatility <= 1e-12:
        raise OptimizationError("Optimal Sharpe has zero volatility.")
    expected = float(weights @ mu)
    return OptimizationResult(
        pd.Series(weights, index=aligned.columns, name="weight"),
        expected,
        volatility,
        (expected - risk_free_rate) / volatility
        if volatility > 1e-12
        else float("nan"),
        objective,
        True,
        str(result.message),
    )
