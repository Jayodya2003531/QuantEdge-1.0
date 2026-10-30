"""Analytical optimizer checks, feasibility, and invalid inputs."""

import numpy as np
import pandas as pd
import pytest

from portfolio import (
    OptimizationError,
    estimate_covariance,
    optimize_portfolio,
)


def orthogonal_returns() -> pd.DataFrame:
    """Uncorrelated assets with variance ratio 1:4."""
    return pd.DataFrame(
        {"A": [0.01, -0.01, 0.01, -0.01], "B": [0.02, 0.02, -0.02, -0.02]}
    )


def test_known_minimum_variance() -> None:
    """Inverse-variance weights match the analytical two-asset solution."""
    frame = orthogonal_returns()
    result = optimize_portfolio(frame)
    assert result.weights["A"] == pytest.approx(0.8, abs=1e-5)
    assert estimate_covariance(frame).loc["A", "B"] == pytest.approx(0)
    capped = optimize_portfolio(frame, max_weight=0.6)
    assert capped.weights["A"] == pytest.approx(0.6, abs=1e-6)


def test_max_sharpe() -> None:
    """Independent equal-mean assets yield inverse-variance Sharpe weights."""
    result = optimize_portfolio(orthogonal_returns() + 0.001, "max_sharpe")
    assert result.weights.sum() == pytest.approx(1)
    assert result.weights["A"] == pytest.approx(0.8, abs=1e-4)


@pytest.mark.parametrize("objective", ["min_variance", "max_sharpe"])
def test_zero_variance(objective: str) -> None:
    """Variance optimization allows constants; Sharpe requires risk."""
    frame = pd.DataFrame({"A": [0.01] * 5, "B": [0.01] * 5})
    if objective == "max_sharpe":
        with pytest.raises(OptimizationError):
            optimize_portfolio(frame, objective)
    else:
        assert optimize_portfolio(frame).annual_volatility == pytest.approx(0)


@pytest.mark.parametrize("cap", [0, 0.4, 1.1, np.nan])
def test_infeasible_cap(cap: float) -> None:
    """Invalid or infeasible allocation bounds fail before optimization."""
    with pytest.raises(ValueError):
        optimize_portfolio(orthogonal_returns(), max_weight=cap)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.1])
def test_invalid_returns(bad: float) -> None:
    """Nonfinite or impossible returns are rejected."""
    frame = orthogonal_returns()
    frame.iloc[0, 0] = bad
    with pytest.raises(ValueError):
        optimize_portfolio(frame)
