"""Peaks-over-threshold GPD loss-tail VaR and Expected Shortfall."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from risk_metrics import _clean_returns


def fit_evt_var_es(
    returns: pd.Series,
    confidence: float = 0.99,
    threshold_quantile: float = 0.95,
    min_exceedances: int = 20,
) -> dict[str, float | int]:
    """Fit excess losses by maximum likelihood with GPD location fixed at 0.

    Tail fraction uses strict empirical exceedances, including ties correctly.
    ES requires shape < 1; divergent ES raises ValueError. No iid assumption
    diagnostics or parameter uncertainty are implied by a successful fit.
    """
    values = _clean_returns(returns)
    if not 0 < threshold_quantile < confidence < 1:
        raise ValueError("Require 0 < threshold_quantile < confidence < 1.")
    if not isinstance(min_exceedances, int) or min_exceedances < 5:
        raise ValueError("min_exceedances must be an integer >= 5.")
    losses = -values
    threshold = float(np.quantile(losses, threshold_quantile))
    excesses = losses[losses > threshold] - threshold
    n_exc = len(excesses)
    if n_exc < min_exceedances or np.ptp(excesses) == 0:
        raise ValueError("Insufficient or degenerate threshold exceedances.")
    fraction = n_exc / len(values)
    if 1 - confidence >= fraction:
        raise ValueError("Requested quantile is outside the fitted tail.")
    try:
        shape, _, scale = genpareto.fit(excesses, floc=0)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        raise ValueError("GPD maximum-likelihood fit failed.") from exc
    if not np.isfinite([shape, scale]).all() or scale <= 0:
        raise ValueError("GPD fit returned invalid parameters.")
    if not np.isfinite(genpareto.logpdf(excesses, shape, scale=scale)).all():
        raise ValueError("Fitted GPD does not support observed exceedances.")
    if shape >= 1:
        raise ValueError("GPD shape >= 1 implies infinite ES.")
    log_ratio = np.log(fraction / (1 - confidence))
    excess_quantile = (
        scale * log_ratio
        if abs(shape) < 1e-8
        else scale * np.expm1(shape * log_ratio) / shape
    )
    var = threshold + excess_quantile
    es = (var + scale - shape * threshold) / (1 - shape)
    if not np.isfinite([var, es]).all():
        raise ValueError("Tail extrapolation overflowed.")
    return {
        "confidence": float(confidence),
        "threshold_quantile": float(threshold_quantile),
        "threshold_loss": threshold,
        "shape": float(shape),
        "scale": float(scale),
        "observations": len(values),
        "exceedances": n_exc,
        "tail_fraction": float(fraction),
        "evt_var": max(0.0, float(var)),
        "evt_es": max(0.0, float(es)),
    }
