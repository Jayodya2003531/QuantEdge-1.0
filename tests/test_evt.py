"""EVT closed-form limits, support checks, and seeded recovery tests."""

import numpy as np
import pandas as pd
import pytest

import evt
from evt import fit_evt_var_es


def tail_sample() -> pd.Series:
    """Bounded deterministic losses spanning a nondegenerate upper tail."""
    return pd.Series(-np.linspace(0, 0.2, 1000))


@pytest.mark.parametrize("shape", [0.0, 0.2, -0.1])
def test_gpd_closed_form(
    monkeypatch: pytest.MonkeyPatch, shape: float
) -> None:
    """POT extrapolation matches the GPD inverse survival function."""
    monkeypatch.setattr(evt.genpareto, "fit", lambda *a, **k: (shape, 0, 0.02))
    result = fit_evt_var_es(tail_sample(), threshold_quantile=0.9)
    ratio = result["tail_fraction"] / 0.01
    excess = (
        0.02 * np.log(ratio)
        if shape == 0
        else 0.02 / shape * (ratio**shape - 1)
    )
    var = result["threshold_loss"] + excess
    assert result["evt_var"] == pytest.approx(var)
    assert result["evt_es"] == pytest.approx(
        (var + 0.02 - shape * result["threshold_loss"]) / (1 - shape)
    )


def test_infinite_es(monkeypatch: pytest.MonkeyPatch) -> None:
    """Infinite fitted first moments are explicitly rejected."""
    monkeypatch.setattr(evt.genpareto, "fit", lambda *a, **k: (1.1, 0, 0.02))
    with pytest.raises(ValueError, match="infinite ES"):
        fit_evt_var_es(tail_sample(), threshold_quantile=0.9)


def test_exponential_quantile_recovery() -> None:
    """Seeded exponential tail recovers a known extreme loss quantile."""
    losses = np.random.default_rng(44).exponential(0.01, 20000)
    result = fit_evt_var_es(pd.Series(-losses), threshold_quantile=0.9)
    assert result["evt_var"] == pytest.approx(-0.01 * np.log(0.01), rel=0.12)
    assert result["evt_es"] >= result["evt_var"]


@pytest.mark.parametrize(
    "values", [[0.0] * 1000, [-0.01] * 3, [np.nan] * 1000]
)
def test_insufficient_or_degenerate_data(values: list[float]) -> None:
    """Missing, constant, and undersized samples cannot support a tail fit."""
    with pytest.raises(ValueError):
        fit_evt_var_es(pd.Series(values))
