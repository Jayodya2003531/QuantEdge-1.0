# QuantEdge

QuantEdge is a Python quantitative research toolkit for adjusted ETF data,
long-only portfolio optimization, performance attribution at portfolio level,
one-day VaR/Expected Shortfall, EVT tail extrapolation, historical backtesting,
and scenario analysis. Default assets are SPY, EFA, EEM, AGG, GLD, XLE, and XLK.
Returns, weights, rates, and shocks use decimal fractions; dollar outputs use
the units of the supplied initial value.

## Architecture

| Module | Responsibility |
| --- | --- |
| `data.py` | Downloads adjusted closes, validates/aligns data, reports coverage, calculates simple/log returns and statistics |
| `portfolio.py` | Annual sample covariance, constrained minimum variance and maximum Sharpe optimization |
| `performance.py` | Wealth, CAGR, annual sample volatility, Sharpe, Sortino, maximum drawdown, Calmar |
| `risk_metrics.py` | Empirical and normal one-period loss VaR/ES |
| `evt.py` | GPD maximum likelihood fit and POT loss-tail extrapolation |
| `backtesting.py` | Drifting holdings, periodic trades, transaction costs, rolling VaR, Kupiec coverage |
| `stress_testing.py` | Historical buy-and-hold replay, asset shocks, volatility scaling |
| `app.py` | Online/CSV/demo orchestration and CSV/JSON exports |
| `tests/` | Offline formula, validation, accounting, timing, and integration tests |

## Installation

Use Python 3.11 or newer. Python 3.12 was used for verification.

```bash
cd quant-edge
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell alternative:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Constraints permit compatible releases rather than freezing one platform.
For deployment, resolve and lock versions in your target environment:

```bash
python -m pip freeze > environment-lock.txt
```

No API keys are needed. Network downloads require Yahoo Finance access and are
subject to provider availability, rate limits, revisions, and applicable data
terms. There is no fabricated-data fallback after a failed download.

## Usage

Start with a deterministic offline run:

```bash
python app.py --demo --output outputs/demo --verbose
```

Download actual adjusted ETF observations (end date is exclusive):

```bash
python app.py --start 2007-01-01 --end 2026-01-01 \
  --risk-free-rate 0.03 --initial-value 100000 \
  --transaction-cost-bps 5 --output outputs/market
```

Replay a saved snapshot without network access:

```bash
python app.py --prices-csv outputs/market/adjusted_prices.csv \
  --output outputs/replay --transaction-cost-bps 5 --risk-free-rate 0.03
```

CSV input has dates in its first column and exactly the seven ticker columns,
with strictly positive adjusted prices. Dates must be unique and parseable.
`--start` and `--end` apply to online downloads only; CSV/demo inputs use their
entire supplied date range. Use separate output directories for separate runs;
existing files with the same names are overwritten. Errors return exit code 1;
`--verbose` adds a traceback. A failed run may leave partial diagnostic files.

Programmatic optimization:

```python
from app import demo_prices
from data import calculate_returns
from portfolio import optimize_portfolio
from backtesting import backtest_portfolio
from performance import performance_summary

returns = calculate_returns(demo_prices()).simple_returns
training = returns.iloc[:2000]
holdout = returns.iloc[2000:]
solution = optimize_portfolio(training, objective="max_sharpe", max_weight=0.4)
result = backtest_portfolio(
    holdout, solution.weights, rebalance="quarterly", transaction_cost_bps=5
)
print(solution.weights)
print(performance_summary(result["history"]["portfolio_return"]))
```

The pipeline estimates minimum-variance targets on the first 70% of returns,
then evaluates fixed targets on the remaining 30%. Equal weight and SPY are
comparators. All three strategies use monthly rebalancing and identical cost
conventions. Targets are not re-estimated using holdout observations.

## Data handling

The downloader requests adjusted OHLC closes explicitly. In unadjusted download
mode it selects `Adj Close`, never the raw `Close`. Provider missingness is
retained for availability reporting. Sorting is explicit; duplicate dates or
asset labels, nonpositive prices, and infinities are rejected.

`calculate_returns` rejects gaps rather than silently filling them. The CLI
saves `data_availability.csv` before this check. To choose an alignment policy
programmatically, inspect raw availability, then call `clean_prices` with
`drop_incomplete_rows=True` or a bounded `forward_fill_limit`. There is no
backfill. Dropping an internal price row turns the next return into a
multi-session return; forward filling creates a stale-price zero return.
Neither should automatically be interpreted as a genuine daily observation.
The caller is responsible for a suitable trading calendar and annualization.

Simple returns are P[t]/P[t-1]-1 and log returns are log(P[t]/P[t-1]). Portfolio
accounting always uses simple returns. Statistics require complete, finite
observations; they never silently drop invalid return rows. Correlations for
constant assets are undefined and remain NaN.

## Models and conventions

### Optimization

Annual mean is 252 times the sample daily mean; annual covariance is 252 times
the sample covariance. SLSQP enforces a unit budget and 0 <= weight <= cap.
Minimum variance is convex with a positive semidefinite sample covariance.
Maximum Sharpe uses deterministic feasible starts and is not a guarantee of a
global solution. Zero-risk Sharpe problems raise an error. Solver success and
numerical feasibility are checked; constraints are not silently repaired by
renormalizing an infeasible solution. Sample moments are noisy; no shrinkage,
leverage, short selling, or parameter uncertainty model is assumed.

For optimization, `risk_free_rate` is an annual arithmetic rate subtracted
from the arithmetic expected return. Performance ratios instead convert the
annual effective risk-free rate to a daily effective rate. The CLI uses minimum
variance, which is independent of the risk-free input.

### Performance

CAGR = exp(252/n * sum(log(1+r))) - 1. Annual volatility uses sample standard
deviation (ddof=1) times sqrt(252). Sharpe uses mean daily excess return divided
by daily standard deviation, annualized by sqrt(252). Sortino uses excess over
an effective daily minimum acceptable return (MAR); downside deviation includes
all observations, assigning zero downside to observations above MAR.

Maximum drawdown is negative and includes starting wealth, so a first-day loss
counts. Calmar is CAGR/abs(maximum drawdown). Undefined zero-denominator ratios
are NaN, exported as blank CSV cells. Performance and portfolio simulation
require returns strictly greater than -100%, since complete ruin prevents
subsequent fractional-return accounting. Risk estimators allow a -100% loss.
Annualization is based on observation count, not elapsed calendar time.

### VaR and Expected Shortfall

Loss L = -return. VaR and ES are one-observation loss fractions, floored at zero;
they are not dollar amounts or annualized risk. Multiply by current wealth to
obtain dollar estimates. Normal approximations may extrapolate beyond a 100%
loss; their mathematical support is unbounded.

Historical VaR uses the inverse empirical CDF. Historical ES integrates its
upper-tail quantiles, assigning fractional weight to the boundary observation.
This handles ties and noninteger sample tail sizes consistently. Normal VaR is
-mu + sigma*z[c]; normal ES is -mu + sigma*phi(z[c])/(1-c), before zero flooring.

POT fits strict excesses above loss threshold u with GPD location fixed at zero.
With empirical exceedance fraction p, fitted shape xi and scale beta:

- VaR = u + beta/xi * ((p/(1-c))**xi - 1).
- For xi approximately zero, VaR = u + beta*log(p/(1-c)).
- ES = (VaR + beta - xi*u)/(1-xi), requiring xi < 1.

Insufficient/degenerate exceedances, invalid support, confidence outside the
fitted tail, and infinite ES raise errors. The CLI records unavailable EVT fits
with a reason instead of inventing estimates. Minimum exceedance counts are
only a numerical guard. Threshold stability, serial dependence, volatility
clustering, goodness of fit, and parameter uncertainty require further research.
The toolkit does not imply those assumptions were validated.

### Backtest execution and costs

Holdings drift after each daily return. Rebalancing occurs before the first
return in a new calendar month or quarter, based on previous closing holdings.
This is an idealized previous-close execution convention, with fractional
holdings and adjusted-return reinvestment, rather than an intraday fill model.
The initial allocation is also logged and charged transaction costs.

`transaction_cost_bps` applies to every dollar bought or sold. Turnover in the
per-asset trade log is absolute traded dollars divided by pre-trade wealth;
summing it gives two-sided turnover, not half-turnover. Post-trade value solves
V_after + cost_rate*sum(abs(w*V_after - holdings_before)) = V_before. Net portfolio
returns include costs, and their product reconciles with the wealth series.
There are no external cash flows, integer lots, taxes, bid/ask simulation,
market impact, or discretionary execution delays.

Rolling historical VaR uses only the preceding window and flags strict loss
exceedances. Kupiec tests unconditional exception frequency using an asymptotic
chi-square distribution, including zero/all-exception boundaries. This does not
test independence or guarantee coverage in small samples.

### Stress scenarios

Historical windows are inclusive: September 2008–March 2009 and
February 19–March 23, 2020. Missing coverage is reported explicitly, allowing
seven calendar days for boundary holidays. Historical replay uses buy-and-hold
and no costs; weights estimated later are counterfactual exposures, not an
ex-ante strategy result. Synthetic data never receive historical-crisis labels.

Hypothetical shocks apply simultaneous per-asset returns, treating omitted
assets as zero shock and rejecting unknown assets. Supply an interest-rate
scenario as asset shocks, e.g. AGG return approximately -duration*yield_change;
this first-order mapping omits convexity, spreads, and duration changes.

```python
from stress_testing import hypothetical_stress

report = hypothetical_stress(
    {"SPY": 0.6, "AGG": 0.4},
    {"SPY": -0.20, "AGG": -6.0 * 0.01},
    portfolio_value=100000,
)
print(report)
```

`volatility_stress` multiplies demeaned returns by a configurable nonnegative
factor, preserving means and correlations. The pipeline uses 2x and exports a
cost-free buy-and-hold replay. This is a sensitivity experiment; it does not
model correlation breakdown or forecast future volatility.

## Outputs and reproducibility

Outputs include raw availability, adjusted prices, simple/log returns,
annualized means/volatilities, correlations, target weights, strategy histories,
per-asset trade logs, performance and VaR/ES tables, EVT diagnostics, rolling
VaR exceptions/Kupiec coverage (when >250 holdout observations), hypothetical
and historical stress reports, and a volatility-stressed wealth path.
`run_metadata.json` records the source, training/test boundaries, cost settings,
and SHA-256 of the saved adjusted-price CSV. Preserve this snapshot and your
environment lock to reproduce a run; Yahoo observations can be revised.

For `python app.py --demo`, the seeded synthetic run produced the following
rounded outputs in the verified environment. These are examples, not actual
ETF investment results:

| Strategy | CAGR | Annual volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Minimum variance | 8.59% | 5.53% | 1.52 | -5.67% |
| Equal weight | 14.06% | 10.14% | 1.35 | -13.11% |
| SPY benchmark | 17.44% | 14.84% | 1.16 | -20.14% |

## Verification

From the project root:

```bash
python -m pytest -q
python -m compileall -q .
# Optional developer checks (ruff is not a runtime dependency):
python -m pip install 'ruff>=0.9,<1'
python -m ruff check --select E,F,I --line-length 79 .
python -m ruff format --check --line-length 79 .
```

Tests are network independent. They mock Yahoo schemas and exercise analytical
inverse-variance optima, bound feasibility, constant returns, invalid data,
empirical tail integration/ties, normal closed forms, GPD/exponential limits,
infinite ES, starting-wealth drawdowns, stress identities, self-financing cost
accounting, look-ahead prevention, Kupiec boundaries, and full CLI pipeline
orchestration. The live Yahoo endpoint has not been integration-tested here.

This is an implemented and tested research codebase, not a certified trading
system. Deployments need environment-specific data/execution validation and
operational monitoring. Results describe model assumptions and historical
samples; they are not forecasts or investment recommendations.
