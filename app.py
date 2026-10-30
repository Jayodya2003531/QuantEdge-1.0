"""QuantEdge command-line orchestration with reproducible offline input."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import backtest_portfolio, kupiec_test, rolling_var_backtest
from data import (
    DEFAULT_TICKERS,
    availability_report,
    calculate_returns,
    clean_prices,
    descriptive_statistics,
    download_adjusted_prices,
)
from evt import fit_evt_var_es
from performance import performance_summary
from portfolio import optimize_portfolio
from risk_metrics import var_es_table
from stress_testing import (
    HISTORICAL_SCENARIOS,
    historical_stress,
    hypothetical_stress,
    volatility_stress,
)

LOGGER = logging.getLogger("quantedge")


def demo_prices(observations: int = 3500, seed: int = 42) -> pd.DataFrame:
    """Generate seeded synthetic correlated prices, not market history."""
    if observations < 20:
        raise ValueError("Demo requires at least 20 observations.")
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.008, (observations, 1))
    beta = np.array([1, 0.9, 1.1, 0.05, 0.1, 1.2, 1.1])
    noise = rng.normal(0, 0.005, (observations, len(beta)))
    logs = 0.00015 + market * beta + noise
    return pd.DataFrame(
        100 * np.exp(np.cumsum(logs, axis=0)),
        columns=DEFAULT_TICKERS,
        index=pd.bdate_range("2007-01-01", periods=observations, name="date"),
    )


def run_pipeline(
    output_dir: Path,
    start: str = "2007-01-01",
    end: str | None = None,
    risk_free_rate: float = 0.0,
    initial_value: float = 100_000.0,
    *,
    prices: pd.DataFrame | None = None,
    source: str = "yahoo",
    transaction_cost_bps: float = 5.0,
) -> dict[str, Path]:
    """Run a 70/30 chronological train/test analysis and save CSV/JSON outputs.

    Input gaps fail explicitly after raw availability is saved. Supply an
    already aligned local frame if a different alignment policy is desired.
    Historical stresses using training-selected weights are counterfactual.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = (
        download_adjusted_prices(start=start, end=end)
        if prices is None
        else prices
    )
    availability_report(raw).to_csv(output_dir / "data_availability.csv")
    clean = clean_prices(raw, drop_incomplete_rows=False)
    if set(clean.columns) != set(DEFAULT_TICKERS):
        raise ValueError("Pipeline requires the seven default assets.")
    clean = clean.loc[:, list(DEFAULT_TICKERS)]
    bundle = calculate_returns(clean)
    returns = bundle.simple_returns
    split = int(len(returns) * 0.7)
    train, test = returns.iloc[:split], returns.iloc[split:]
    if len(train) < 20 or len(test) < 20:
        raise ValueError("Require at least 20 training and 20 test returns.")
    bundle.prices.to_csv(output_dir / "adjusted_prices.csv")
    returns.to_csv(output_dir / "simple_returns.csv")
    bundle.log_returns.to_csv(output_dir / "log_returns.csv")
    for name, table in descriptive_statistics(returns).items():
        table.to_csv(output_dir / f"{name}.csv")
    optimized = optimize_portfolio(train, max_weight=0.4)
    optimized.weights.to_csv(output_dir / "minimum_variance_weights.csv")
    benchmark = pd.Series(0.0, index=returns.columns)
    benchmark.loc["SPY"] = 1
    strategies = {
        "minimum_variance": optimized.weights,
        "equal_weight": pd.Series(
            1 / len(returns.columns), index=returns.columns
        ),
        "SPY_benchmark": benchmark,
    }
    summaries = {}
    strategy_returns = {}
    for name, weights in strategies.items():
        result = backtest_portfolio(
            test, weights, initial_value, "monthly", transaction_cost_bps
        )
        history = result["history"]
        strategy_returns[name] = history["portfolio_return"]
        summaries[name] = performance_summary(
            history["portfolio_return"], risk_free_rate
        )
        history.to_csv(output_dir / f"{name}_backtest.csv")
        result["rebalancing_log"].to_csv(
            output_dir / f"{name}_rebalancing_log.csv", index=False
        )
    pd.DataFrame(summaries).T.to_csv(output_dir / "performance_summary.csv")
    selected = strategy_returns["minimum_variance"]
    var_es_table(selected).to_csv(output_dir / "var_es.csv")
    try:
        evt_result = fit_evt_var_es(selected, threshold_quantile=0.90)
    except ValueError as exc:
        evt_result = {"status": "unavailable", "reason": str(exc)}
        LOGGER.warning("EVT unavailable: %s", exc)
    (output_dir / "evt_results.json").write_text(
        json.dumps(evt_result, indent=2, allow_nan=False), encoding="utf-8"
    )
    if len(selected) > 250:
        forecasts = rolling_var_backtest(selected)
        forecasts.to_csv(output_dir / "var_backtest.csv")
        (output_dir / "var_coverage.json").write_text(
            json.dumps(kupiec_test(forecasts["exception"]), indent=2),
            encoding="utf-8",
        )
    shocks = dict(
        zip(DEFAULT_TICKERS, [-0.2, -0.25, -0.3, 0.02, 0.05, -0.3, -0.25])
    )
    hypothetical_stress(optimized.weights, shocks, initial_value).to_csv(
        output_dir / "hypothetical_stress.csv", header=["value"]
    )
    stress_rows = []
    if source != "synthetic":
        for name, (first, last) in HISTORICAL_SCENARIOS.items():
            try:
                row = historical_stress(
                    returns, optimized.weights, first, last
                )
                stress_rows.append({"scenario": name, "status": "ok", **row})
            except ValueError as exc:
                stress_rows.append({"scenario": name, "status": str(exc)})
    else:
        stress_rows.append(
            {"scenario": "historical", "status": "synthetic: skipped"}
        )
    pd.DataFrame(stress_rows).to_csv(
        output_dir / "historical_stress.csv", index=False
    )
    stressed = volatility_stress(test)
    stress_bt = backtest_portfolio(stressed, optimized.weights, initial_value)
    stress_bt["history"].to_csv(output_dir / "volatility_stress.csv")
    metadata = {
        "source": source,
        "tickers": list(DEFAULT_TICKERS),
        "training_start": str(train.index[0]),
        "training_end": str(train.index[-1]),
        "test_start": str(test.index[0]),
        "test_end": str(test.index[-1]),
        "training_observations": len(train),
        "test_observations": len(test),
        "risk_free_rate": risk_free_rate,
        "initial_value": initial_value,
        "transaction_cost_bps": transaction_cost_bps,
        "price_sha256": hashlib.sha256(
            (output_dir / "adjusted_prices.csv").read_bytes()
        ).hexdigest(),
        "historical_stress_interpretation": "counterfactual exposures",
        "volatility_stress_multiplier": 2.0,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return {
        "output_dir": output_dir,
        "summary": output_dir / "performance_summary.csv",
    }


def main() -> None:
    """Run online, local CSV, or seeded demo analysis from CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--demo", action="store_true", help="Use synthetic prices."
    )
    source.add_argument(
        "--prices-csv", type=Path, help="Date-indexed adjusted prices."
    )
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument(
        "--end", default=None, help="Exclusive download end date."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--initial-value", type=float, default=100000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING
    )
    try:
        prices = None
        label = "yahoo"
        if args.demo:
            prices, label = demo_prices(), "synthetic"
        elif args.prices_csv:
            prices = pd.read_csv(
                args.prices_csv, index_col=0, parse_dates=True
            )
            label = "local_csv"
        result = run_pipeline(
            args.output,
            args.start,
            args.end,
            args.risk_free_rate,
            args.initial_value,
            prices=prices,
            source=label,
            transaction_cost_bps=args.transaction_cost_bps,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        LOGGER.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        raise SystemExit(1) from exc
    print(f"Outputs saved to {result['output_dir'].resolve()}")


if __name__ == "__main__":
    main()
