"""Portfolio accounting, timing, VaR coverage, and full offline integration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import demo_prices, run_pipeline
from backtesting import backtest_portfolio, kupiec_test, rolling_var_backtest


def test_buy_and_hold() -> None:
    """Holdings drift without implicit daily rebalancing."""
    frame = pd.DataFrame(
        {"A": [0.1] * 3, "B": [0.0] * 3},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    result = backtest_portfolio(frame, {"A": 0.5, "B": 0.5}, 100)
    assert result["summary"]["ending_value"] == pytest.approx(50 * 1.1**3 + 50)


def test_costs_are_self_financing_and_net() -> None:
    """Trade costs, wealth, and compounded net returns reconcile exactly."""
    frame = pd.DataFrame(
        {"A": [0.2, 0, -0.1], "B": [0, 0, 0.1]},
        index=pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"]),
    )
    result = backtest_portfolio(
        frame, {"A": 0.5, "B": 0.5}, 100, "monthly", 100
    )
    history, trades = result["history"], result["rebalancing_log"]
    assert history.portfolio_value.iloc[0] == pytest.approx(100 / 1.01 * 1.1)
    assert (1 + history.portfolio_return).prod() * 100 == pytest.approx(
        history.portfolio_value.iloc[-1]
    )
    assert trades.cost.sum() == pytest.approx(history.transaction_cost.sum())
    for date, group in trades.groupby("date"):
        before = (
            100
            if date == history.index[0]
            else history.loc[:date].portfolio_value.iloc[-2]
        )
        pre = (
            np.zeros(2)
            if date == history.index[0]
            else np.array([60, 50]) / 1.01
        )
        post = pre + group.trade_notional.to_numpy()
        assert post.sum() + group.cost.sum() == pytest.approx(before)
        assert post[0] == pytest.approx(post[1])


@pytest.mark.parametrize(
    "weights", [{"A": 0.2, "B": 0.2}, {"A": 1.0}, {"A": 1, "B": 0, "C": 0}]
)
def test_invalid_weights(weights: dict[str, float]) -> None:
    """Missing/extra assets and nonunit budgets cannot pass validation."""
    frame = demo_prices(20).iloc[:, :2].pct_change().dropna()
    frame.columns = ["A", "B"]
    with pytest.raises(ValueError):
        backtest_portfolio(frame, weights)


def test_forecasts_do_not_look_ahead() -> None:
    """Changing an outcome cannot change that day's prior-window forecast."""
    returns = pd.Series(np.linspace(-0.02, 0.02, 60))
    original = rolling_var_backtest(returns, window=20)
    returns.iloc[30] = -0.5
    changed = rolling_var_backtest(returns, window=20)
    assert original.loc[30, "var"] == changed.loc[30, "var"]
    assert changed.loc[30, "exception"]
    for flags in [[False] * 100, [True] * 100, [True] * 5 + [False] * 95]:
        assert np.isfinite(kupiec_test(pd.Series(flags))["lr_statistic"])
    assert kupiec_test(pd.Series([True] * 5 + [False] * 95))[
        "p_value"
    ] == pytest.approx(1)


def test_full_offline_pipeline(tmp_path: Path) -> None:
    """All orchestration stages save usable outputs without network access."""
    run_pipeline(tmp_path, prices=demo_prices(1200), source="synthetic")
    summary = pd.read_csv(tmp_path / "performance_summary.csv", index_col=0)
    assert len(summary) == 3
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["training_end"] < metadata["test_start"]
    assert metadata["source"] == "synthetic"
    assert len(metadata["price_sha256"]) == 64
    assert (tmp_path / "var_backtest.csv").exists()
    assert (tmp_path / "volatility_stress.csv").exists()
