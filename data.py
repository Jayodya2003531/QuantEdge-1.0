"""Adjusted daily ETF prices, explicit alignment, and return statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ("SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK")


class DataValidationError(ValueError):
    """Market data violate the requested schema or numerical constraints."""


@dataclass(frozen=True)
class ReturnData:
    """Aligned adjusted prices and corresponding simple and log returns."""

    prices: pd.DataFrame
    simple_returns: pd.DataFrame
    log_returns: pd.DataFrame


def validate_returns(
    returns: pd.DataFrame, minimum_rows: int = 2
) -> pd.DataFrame:
    """Require finite simple returns, unique labels, and no missing values.

    Returns below -100% are invalid. A complete loss is permitted here;
    consumers requiring strictly positive wealth must reject it separately.
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] == 0:
        raise DataValidationError("returns must be a DataFrame with assets.")
    if len(returns) < minimum_rows:
        raise DataValidationError(
            f"At least {minimum_rows} rows are required."
        )
    if returns.columns.has_duplicates or returns.index.has_duplicates:
        raise DataValidationError("Return labels must be unique.")
    result = returns.astype(float)
    if not np.isfinite(result.to_numpy()).all() or (result < -1).any().any():
        raise DataValidationError("Require finite simple returns >= -1.")
    return result


def validate_weights(
    weights: pd.Series | dict[str, float], columns: pd.Index
) -> pd.Series:
    """Require exactly matching, long-only, fully invested asset weights."""
    result = pd.Series(weights, dtype=float)
    if result.index.has_duplicates or set(result.index) != set(columns):
        raise DataValidationError("Weights must match asset labels exactly.")
    result = result.reindex(columns)
    if not np.isfinite(result).all() or (result < 0).any():
        raise DataValidationError("Weights must be finite and non-negative.")
    if not np.isclose(result.sum(), 1.0, atol=1e-10, rtol=0):
        raise DataValidationError("Weights must sum to one.")
    return result / result.sum()


def download_adjusted_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2007-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download adjusted closes; end is exclusive and gaps are preserved.

    With auto_adjust=False, explicitly select Adj Close rather than Close.
    Network/provider failures are raised with context, never replaced by data.
    """
    import yfinance as yf

    if isinstance(tickers, str):
        raise DataValidationError("tickers must be an iterable of symbols.")
    symbols = list(dict.fromkeys(str(t).strip().upper() for t in tickers))
    if not symbols or any(not t for t in symbols):
        raise DataValidationError("Provide non-empty ticker symbols.")
    stop = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    if pd.Timestamp(start) >= stop:
        raise DataValidationError("start must precede end.")
    try:
        raw = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            interval="1d",
            group_by="column",
            progress=False,
            threads=True,
            timeout=20,
            keepna=True,
        )
    except Exception as exc:
        raise RuntimeError("Yahoo Finance download failed.") from exc
    if raw is None or raw.empty:
        raise DataValidationError("Provider returned no observations.")
    field = "Close" if auto_adjust else "Adj Close"
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            raise DataValidationError(f"Missing adjusted price field: {field}")
        prices = raw[field].copy()
    else:
        if len(symbols) != 1 or field not in raw.columns:
            raise DataValidationError("Unexpected provider column layout.")
        prices = raw[[field]].copy()
        prices.columns = symbols
    if set(symbols) - set(prices.columns):
        raise DataValidationError("Provider omitted requested symbols.")
    prices = clean_prices(prices[symbols], drop_incomplete_rows=False)
    if prices.isna().all().any():
        raise DataValidationError("An asset has no usable prices.")
    return prices


def clean_prices(
    prices: pd.DataFrame,
    *,
    drop_incomplete_rows: bool = True,
    forward_fill_limit: int = 0,
) -> pd.DataFrame:
    """Sort dates, validate positive prices, and explicitly align observations.

    Missing values remain unless dropping or bounded forward filling is
    requested. No backfill is performed. Dropping internal gaps creates
    multi-session returns; inspect availability before doing so.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise DataValidationError("prices must be a non-empty DataFrame.")
    if prices.columns.has_duplicates:
        raise DataValidationError("Price columns must be unique.")
    if not isinstance(forward_fill_limit, int) or forward_fill_limit < 0:
        raise DataValidationError(
            "forward_fill_limit must be a nonnegative int."
        )
    result = prices.astype(float).copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if result.index.hasnans or result.index.has_duplicates:
        raise DataValidationError("Dates must be valid and unique.")
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.sort_index()
    observed = result.to_numpy()
    if np.isinf(observed).any() or np.any(
        observed[np.isfinite(observed)] <= 0
    ):
        raise DataValidationError(
            "Observed prices must be positive and finite."
        )
    if forward_fill_limit:
        result = result.ffill(limit=forward_fill_limit)
    if drop_incomplete_rows:
        result = result.dropna(how="any")
    if result.empty or result.isna().all(axis=None):
        raise DataValidationError("No usable price observations.")
    return result


def calculate_returns(prices: pd.DataFrame) -> ReturnData:
    """Compute returns without filling or silently bridging missing prices."""
    clean = clean_prices(prices, drop_incomplete_rows=False)
    if len(clean) < 2 or clean.isna().any().any():
        raise DataValidationError("Need >=2 complete aligned price rows.")
    simple = clean.pct_change(fill_method=None).iloc[1:]
    logs = np.log(clean).diff().iloc[1:]
    validate_returns(simple, minimum_rows=1)
    if not np.isfinite(logs.to_numpy()).all():
        raise DataValidationError("Non-finite log returns.")
    return ReturnData(clean, simple, logs)


def availability_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw coverage and missingness before alignment."""
    clean = clean_prices(prices, drop_incomplete_rows=False)
    rows = []
    for ticker, series in clean.items():
        observed = series.dropna()
        rows.append(
            {
                "ticker": ticker,
                "observations": int(series.count()),
                "missing": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "first_date": observed.index.min(),
                "last_date": observed.index.max(),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def descriptive_statistics(
    returns: pd.DataFrame, periods_per_year: int = 252
) -> dict[str, pd.Series | pd.DataFrame]:
    """Annualize arithmetic means/sample volatility and report correlation."""
    aligned = validate_returns(returns)
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise DataValidationError("periods_per_year must be a positive int.")
    return {
        "annualized_mean": aligned.mean() * periods_per_year,
        "annualized_volatility": aligned.std(ddof=1)
        * np.sqrt(periods_per_year),
        "correlation": aligned.corr(),
    }
