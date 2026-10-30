"""One-period loss VaR and ES, floored at zero, in return-fraction units."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _clean_returns(returns: pd.Series) -> np.ndarray:
    """Require finite observations; missing values are never discarded."""
    values = pd.Series(returns, dtype=float).to_numpy()
    if not values.size or not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("Require nonempty finite simple returns >= -1.")
    return values


def _validate_confidence(confidence: float) -> None:
    """Require a confidence strictly between zero and one."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1.")


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Empirical inverse-CDF loss quantile, floored at zero."""
    values = _clean_returns(returns)
    _validate_confidence(confidence)
    return max(
        0.0, float(np.quantile(-values, confidence, method="inverted_cdf"))
    )


def historical_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Integrate empirical quantiles over the worst 1-confidence mass.

    Fractional weight on the boundary order statistic makes ES correct for
    small samples and tied losses, rather than averaging all tied points.
    """
    values = _clean_returns(returns)
    _validate_confidence(confidence)
    losses = np.sort(-values)[::-1]
    mass = len(losses) * (1 - confidence)
    full = min(int(np.floor(mass)), len(losses) - 1)
    total = losses[:full].sum() + (mass - full) * losses[full]
    return max(0.0, float(total / mass))


def parametric_var(
    returns: pd.Series, confidence: float = 0.95, distribution: str = "normal"
) -> float:
    """Normal loss quantile using sample mean and standard deviation."""
    values = _clean_returns(returns)
    _validate_confidence(confidence)
    if distribution != "normal":
        raise ValueError("Only the normal distribution is supported.")
    sigma = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return max(0.0, float(-values.mean() + sigma * norm.ppf(confidence)))


def parametric_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Normal upper-tail loss expectation; the model is not loss bounded."""
    values = _clean_returns(returns)
    _validate_confidence(confidence)
    sigma = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return max(
        0.0,
        float(
            -values.mean()
            + sigma * norm.pdf(norm.ppf(confidence)) / (1 - confidence)
        ),
    )


def var_es_table(
    returns: pd.Series, confidence_levels: tuple[float, ...] = (0.95, 0.99)
) -> pd.DataFrame:
    """Tabulate historical and normal VaR/ES for each confidence."""
    if not confidence_levels:
        raise ValueError("Provide at least one confidence level.")
    return pd.DataFrame(
        [
            {
                "confidence": c,
                "historical_var": historical_var(returns, c),
                "historical_es": historical_es(returns, c),
                "parametric_var": parametric_var(returns, c),
                "parametric_es": parametric_es(returns, c),
            }
            for c in confidence_levels
        ]
    ).set_index("confidence")
