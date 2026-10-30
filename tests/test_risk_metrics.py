"""Known loss distributions, performance metrics, and stress identities."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from performance import maximum_drawdown, performance_summary
from risk_metrics import (
    historical_es,
    historical_var,
    parametric_es,
    parametric_var,
)
from stress_testing import (
    historical_stress,
    hypothetical_stress,
    volatility_stress,
)


@pytest.mark.parametrize(
    "fn", [historical_var, historical_es, parametric_var, parametric_es]
)
@pytest.mark.parametrize("confidence", [0, 1, np.nan])
def test_invalid_confidence(
    fn: Callable[[pd.Series, float], float], confidence: float
) -> None:
    """Every risk estimator enforces open confidence bounds."""
    with pytest.raises(ValueError):
        fn(pd.Series([0.01, -0.01]), confidence)


def test_empirical_tail_mass_and_ties() -> None:
    """ES integrates exactly the worst 1.5 observations, including ties."""
    returns = pd.Series([-0.1, -0.04, -0.04, 0.02])
    assert historical_var(returns, 0.625) == pytest.approx(0.04)
    assert historical_es(returns, 0.625) == pytest.approx(
        (0.1 + 0.5 * 0.04) / 1.5
    )
    assert historical_es(returns, 0.99) == pytest.approx(0.1)


def test_normal_formula_and_constant_returns() -> None:
    """Normal estimates agree with closed forms; constant returns are valid."""
    returns = pd.Series([-0.02, 0, 0.02])
    sigma = returns.std()
    assert parametric_var(returns) == pytest.approx(sigma * norm.ppf(0.95))
    assert parametric_es(returns) == pytest.approx(
        sigma * norm.pdf(norm.ppf(0.95)) / 0.05
    )
    for fn in [historical_var, historical_es, parametric_var, parametric_es]:
        assert fn(pd.Series([-0.02] * 5)) == pytest.approx(0.02)
        assert fn(pd.Series([0.02] * 5)) == 0
        with pytest.raises(ValueError):
            fn(pd.Series([0.01, np.nan]))


def test_initial_drawdown_and_cagr() -> None:
    """First-day losses count toward drawdown and CAGR compounds returns."""
    returns = pd.Series([-0.1, 0.1])
    assert maximum_drawdown(returns) == pytest.approx(-0.1)
    summary = performance_summary(returns, periods_per_year=2)
    assert summary["annualized_return_cagr"] == pytest.approx(-0.01)
    assert summary["calmar_ratio"] == pytest.approx(-0.1)
    assert np.isnan(performance_summary(pd.Series([0.0] * 3))["sharpe_ratio"])
    with pytest.raises(ValueError):
        performance_summary(returns, risk_free_rate=-1)


def test_stress_models() -> None:
    """Dollar shocks, historical replay, and volatility scaling reconcile."""
    weights = {"A": 0.5, "B": 0.5}
    assert hypothetical_stress(weights, {"A": -0.2}, 100)["ending_value"] == 90
    frame = pd.DataFrame(
        {"A": [-0.1, 0.1, 0], "B": [0, 0, 0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    result = historical_stress(frame, weights, "2024-01-01", "2024-01-03", 100)
    assert result["ending_value"] == pytest.approx(99.5)
    scaled = volatility_stress(frame, 2)
    np.testing.assert_allclose(scaled.mean(), frame.mean(), atol=1e-15)
    np.testing.assert_allclose(scaled.std(), frame.std() * 2)
    with pytest.raises(ValueError):
        hypothetical_stress(weights, {"UNKNOWN": -0.2})
    with pytest.raises(ValueError):
        historical_stress(frame, weights, "2008-01-01", "2008-12-31")
