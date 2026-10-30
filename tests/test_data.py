"""Data validation, provider schemas, availability, and return identities."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data import (
    DataValidationError,
    availability_report,
    calculate_returns,
    clean_prices,
    descriptive_statistics,
    download_adjusted_prices,
)


def sample_prices() -> pd.DataFrame:
    """Small positive price frame with independently calculable returns."""
    return pd.DataFrame(
        {"SPY": [100.0, 110.0, 99.0], "AGG": [50.0, 52.0, 52.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )


def test_return_identities() -> None:
    """Simple/log returns preserve the input price ratios."""
    result = calculate_returns(sample_prices())
    assert result.simple_returns.iloc[0, 0] == pytest.approx(0.1)
    np.testing.assert_allclose(
        np.expm1(result.log_returns), result.simple_returns
    )
    stats = descriptive_statistics(result.simple_returns)
    pd.testing.assert_series_equal(
        stats["annualized_mean"], result.simple_returns.mean() * 252
    )


def test_alignment_is_explicit() -> None:
    """Missing data are reported; return calculation does not fill gaps."""
    frame = sample_prices().iloc[::-1]
    frame.iloc[1, 1] = np.nan
    assert availability_report(frame).loc["AGG", "missing"] == 1
    assert clean_prices(frame).index.is_monotonic_increasing
    assert len(clean_prices(frame)) == 2
    with pytest.raises(DataValidationError):
        calculate_returns(frame)
    assert not clean_prices(frame, forward_fill_limit=1).isna().any().any()


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_invalid_prices(bad: float) -> None:
    """Invalid observed prices cannot masquerade as missing data."""
    frame = sample_prices()
    frame.iloc[1, 0] = bad
    with pytest.raises(DataValidationError):
        clean_prices(frame)


def test_duplicate_dates_and_columns() -> None:
    """Ambiguous labels are rejected."""
    frame = sample_prices()
    with pytest.raises(DataValidationError):
        clean_prices(pd.concat([frame, frame]))
    frame.columns = ["SPY", "SPY"]
    with pytest.raises(DataValidationError):
        clean_prices(frame)


@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("multi", [True, False])
def test_download_adjusted_field(
    monkeypatch: pytest.MonkeyPatch, adjust: bool, multi: bool
) -> None:
    """Provider schemas select adjusted closes in both adjustment modes."""
    raw = pd.DataFrame(
        {"Close": [10.0, 12.0], "Adj Close": [5.0, 6.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    if multi:
        raw.columns = pd.MultiIndex.from_tuples([(c, "SPY") for c in raw])
    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: raw)
    )
    result = download_adjusted_prices(
        ["SPY"], "2024-01-01", "2024-02-01", adjust
    )
    assert result.iloc[0, 0] == (10 if adjust else 5)


def test_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty provider responses produce actionable errors."""
    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: None)
    )
    with pytest.raises(DataValidationError):
        download_adjusted_prices(["SPY"], "2024-01-01", "2024-02-01")
