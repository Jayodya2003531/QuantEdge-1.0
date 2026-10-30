"""
QuantEdge Pro — Institutional-Grade Portfolio Analytics Terminal
==================================================================
A Bloomberg-terminal-style Streamlit front end for the QuantEdge
quantitative backend (data.py, portfolio.py, performance.py,
risk_metrics.py, evt.py, backtesting.py, stress_testing.py).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# QuantEdge Core Modules
# ----------------------------------------------------------------------------
from data import (
    DEFAULT_TICKERS, download_adjusted_prices, clean_prices,
    calculate_returns, descriptive_statistics
)
from portfolio import optimize_portfolio
from backtesting import backtest_portfolio
from performance import performance_summary
from evt import fit_evt_var_es
from stress_testing import HISTORICAL_SCENARIOS, historical_stress

# risk_metrics is optional — not every build exposes every ratio, so we
# import defensively and fall back to local formulas if something is missing.
try:
    import risk_metrics as rm
except ImportError:
    rm = None

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="QuantEdge Pro | Portfolio Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS — INSTITUTIONAL DARK THEME
# ==============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0a0e14;
        --bg-secondary: #0f1520;
        --bg-card: #121824;
        --border-color: #1f2a3a;
        --accent-green: #00e6a8;
        --accent-red: #ff4d6d;
        --accent-blue: #4d9fff;
        --accent-gold: #f0b429;
        --text-primary: #e8edf4;
        --text-secondary: #8694a8;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: radial-gradient(circle at top left, #0d1420 0%, #080b10 100%);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    /* Titles */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 0%, #8694a8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.3px;
    }

    .terminal-subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--bg-card), #0d1320);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.1rem 0.8rem 1.1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #2c3a50;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem !important;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: var(--text-secondary) !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem !important;
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: var(--bg-secondary);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card) !important;
        color: var(--accent-green) !important;
        border: 1px solid var(--border-color);
    }

    /* Expanders */
    details {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton button {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton button:hover {
        border-color: var(--accent-green);
        color: var(--accent-green);
    }

    /* Section divider */
    .section-divider {
        border-top: 1px solid var(--border-color);
        margin: 1.4rem 0 1.2rem 0;
    }

    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .pill-green { background: rgba(0,230,168,0.12); color: var(--accent-green); border: 1px solid rgba(0,230,168,0.3); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_GREEN = "#00e6a8"
COLOR_RED = "#ff4d6d"
COLOR_BLUE = "#4d9fff"
COLOR_GOLD = "#f0b429"
PALETTE = ["#00e6a8", "#4d9fff", "#f0b429", "#ff4d6d", "#9d7bff", "#3ad6ff", "#ff9e4d"]


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply consistent institutional styling to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e8edf4", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(bgcolor="#121824", font_size=12, font_family="JetBrains Mono"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1a2332", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1a2332", zeroline=False)
    return fig


def safe_get(d, key, default=np.nan):
    try:
        return d[key]
    except Exception:
        return default


def compute_drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    running_max = portfolio_values.cummax()
    return (portfolio_values / running_max) - 1.0


def compute_sortino(returns: pd.Series, risk_free: float, periods_per_year: int = 252) -> float:
    excess = returns - (risk_free / periods_per_year)
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess.mean() * periods_per_year) / downside_std


def compute_calmar(cagr: float, max_dd: float) -> float:
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return cagr / abs(max_dd)


def compute_rolling_sharpe(returns: pd.Series, risk_free: float, window: int = 63,
                            periods_per_year: int = 252) -> pd.Series:
    excess = returns - (risk_free / periods_per_year)
    roll_mean = excess.rolling(window).mean() * periods_per_year
    roll_std = returns.rolling(window).std() * np.sqrt(periods_per_year)
    return roll_mean / roll_std


def fmt_pct(x, decimals=2):
    return f"{x * 100:.{decimals}f}%" if pd.notna(x) else "N/A"


def fmt_num(x, decimals=2):
    return f"{x:.{decimals}f}" if pd.notna(x) else "N/A"


# ==============================================================================
# CACHED PIPELINE FUNCTIONS
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_market_data(tickers, start, end):
    raw_prices = download_adjusted_prices(tickers=tickers, start=start, end=end)
    clean = clean_prices(raw_prices, drop_incomplete_rows=False)
    clean = clean.loc[:, list(DEFAULT_TICKERS)].dropna()
    return calculate_returns(clean)


@st.cache_data(show_spinner=False)
def run_optimization(returns, objective, max_weight):
    return optimize_portfolio(returns, objective=objective, max_weight=max_weight)


@st.cache_data(show_spinner=False)
def run_backtest_cached(returns, weights, initial_val, reb, cost):
    return backtest_portfolio(
        asset_returns=returns,
        target_weights=weights,
        initial_value=initial_val,
        rebalance=reb,
        transaction_cost_bps=cost,
    )


@st.cache_data(show_spinner=False)
def run_performance(port_returns, risk_free_rate):
    return performance_summary(port_returns, risk_free_rate=risk_free_rate)


@st.cache_data(show_spinner=False)
def run_evt(port_returns, confidence, threshold_quantile):
    return fit_evt_var_es(port_returns, confidence=confidence, threshold_quantile=threshold_quantile)


# ==============================================================================
# SIDEBAR — STRATEGY CONTROL PANEL
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Strategy Control Panel")
    st.markdown('<span class="status-pill pill-green">● ENGINE READY</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📅 Sample Window**")
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 1, 1))

    st.markdown("**💰 Capital & Costs**")
    initial_capital = st.number_input("Initial Capital ($)", value=100000.0, step=10000.0, format="%.2f")
    risk_free_rate = st.number_input("Risk-Free Rate (Annual %)", value=2.0, step=0.1) / 100
    transaction_cost = st.slider("Transaction Cost (bps)", 0.0, 50.0, 5.0)

    st.markdown("**🎯 Optimization Setup**")
    objective = st.selectbox("Objective", ["max_sharpe", "min_variance"])
    max_asset_weight = st.slider("Max Weight per Asset", 0.1, 1.0, 0.4)
    rebalance_freq = st.selectbox("Rebalancing", ["monthly", "quarterly", "none"])

    st.markdown("**⚠️ Tail Risk Settings**")
    evt_confidence = st.select_slider("EVT Confidence", options=[0.95, 0.975, 0.99], value=0.99)
    evt_threshold_q = st.slider("POT Threshold Quantile", 0.80, 0.97, 0.90)

    st.markdown("---")
    run_button = st.button("🚀 Run Pipeline", use_container_width=True)
    st.caption("QuantEdge Pro · Internal Research Build")

# ==============================================================================
# HEADER
# ==============================================================================
st.title("📈 QuantEdge Pro")
st.markdown(
    f'<div class="terminal-subtitle">Institutional Portfolio Analytics Terminal · '
    f'{", ".join(DEFAULT_TICKERS)} · Updated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
with st.spinner("Fetching market data and running quantitative pipeline..."):
    try:
        # 1. Fetch & prepare data
        try:
            ret_data = get_market_data(DEFAULT_TICKERS, str(start_date), str(end_date))
            simple_returns = ret_data.simple_returns
        except Exception as e:
            st.error(f"🔴 Data download failed: {e}")
            st.stop()

        # 2. Optimize portfolio
        try:
            opt_result = run_optimization(simple_returns, objective, max_asset_weight)
        except Exception as e:
            st.toast("⚠️ Optimizer did not converge cleanly — check constraints.", icon="⚠️")
            st.error(f"🔴 Portfolio optimization failed: {e}")
            st.stop()

        # 3. Backtest
        try:
            bt_result = run_backtest_cached(
                simple_returns, opt_result.weights, initial_capital, rebalance_freq, transaction_cost
            )
            port_history = bt_result["history"]
            port_returns = port_history["portfolio_return"]
        except Exception as e:
            st.error(f"🔴 Backtest execution failed: {e}")
            st.stop()

        # 4. Performance metrics
        perf_metrics = run_performance(port_returns, risk_free_rate)
        cagr = safe_get(perf_metrics, "annualized_return_cagr")
        ann_vol = safe_get(perf_metrics, "annualized_volatility")
        sharpe = safe_get(perf_metrics, "sharpe_ratio")
        max_dd = safe_get(perf_metrics, "maximum_drawdown")

        # Sortino / Calmar — prefer backend risk_metrics module, else compute locally
        sortino = np.nan
        calmar = np.nan
        if rm is not None:
            try:
                sortino = rm.sortino_ratio(port_returns, risk_free_rate=risk_free_rate)
            except Exception:
                sortino = safe_get(perf_metrics, "sortino_ratio")
        else:
            sortino = safe_get(perf_metrics, "sortino_ratio")
        if pd.isna(sortino):
            sortino = compute_sortino(port_returns, risk_free_rate)

        if rm is not None:
            try:
                calmar = rm.calmar_ratio(port_returns)
            except Exception:
                calmar = safe_get(perf_metrics, "calmar_ratio")
        else:
            calmar = safe_get(perf_metrics, "calmar_ratio")
        if pd.isna(calmar):
            calmar = compute_calmar(cagr, max_dd)

        toast_shown = st.session_state.get("_pipeline_toast_shown", False)
        if not toast_shown:
            st.toast("✅ Pipeline executed successfully.", icon="✅")
            st.session_state["_pipeline_toast_shown"] = True

        # ----------------------------------------------------------------
        # KPI HEADER
        # ----------------------------------------------------------------
        st.markdown("#### 🧮 Key Performance Indicators")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("CAGR", fmt_pct(cagr))
        k2.metric("Ann. Volatility", fmt_pct(ann_vol))
        k3.metric("Sharpe", fmt_num(sharpe))
        k4.metric("Sortino", fmt_num(sortino))
        k5.metric("Calmar", fmt_num(calmar))
        k6.metric("Max Drawdown", fmt_pct(max_dd))

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------
        # TABS
        # ----------------------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊  Performance Overview",
            "🥧  Portfolio Allocation",
            "⚠️  Tail Risk & EVT",
            "🌪️  Stress Testing",
        ])

        # ============================= TAB 1 =============================
        with tab1:
            colA, colB = st.columns([2, 1])

            with colA:
                st.markdown("##### Wealth Compounding Index")
                fig_wealth = go.Figure()
                fig_wealth.add_trace(go.Scatter(
                    x=port_history.index, y=port_history["portfolio_value"],
                    mode="lines", name="Portfolio Value",
                    line=dict(color=COLOR_GREEN, width=2),
                    fill="tozeroy", fillcolor="rgba(0,230,168,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
                ))
                fig_wealth = style_fig(fig_wealth, height=380)
                fig_wealth.update_yaxes(title_text="Portfolio Value ($)")
                st.plotly_chart(fig_wealth, use_container_width=True)

                st.markdown("##### Underwater Chart (Drawdown)")
                dd_series = compute_drawdown_series(port_history["portfolio_value"])
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd_series.index, y=dd_series.values * 100,
                    mode="lines", name="Drawdown",
                    line=dict(color=COLOR_RED, width=1.4),
                    fill="tozeroy", fillcolor="rgba(255,77,109,0.18)",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
                ))
                fig_dd = style_fig(fig_dd, height=260)
                fig_dd.update_yaxes(title_text="Drawdown (%)")
                st.plotly_chart(fig_dd, use_container_width=True)

            with colB:
                st.markdown("##### Rolling Risk Metrics")
                window_days = st.radio("Window", [21, 63, 126], index=1, horizontal=True,
                                        key="rolling_window")

                roll_vol = port_returns.rolling(window_days).std() * np.sqrt(252) * 100
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=roll_vol.index, y=roll_vol.values, mode="lines",
                    name=f"{window_days}D Rolling Vol",
                    line=dict(color=COLOR_BLUE, width=1.6),
                ))
                fig_vol = style_fig(fig_vol, height=260)
                fig_vol.update_yaxes(title_text="Ann. Vol (%)")
                st.plotly_chart(fig_vol, use_container_width=True)

                roll_sharpe = compute_rolling_sharpe(port_returns, risk_free_rate, window=window_days)
                fig_sharpe = go.Figure()
                fig_sharpe.add_trace(go.Scatter(
                    x=roll_sharpe.index, y=roll_sharpe.values, mode="lines",
                    name=f"{window_days}D Rolling Sharpe",
                    line=dict(color=COLOR_GOLD, width=1.6),
                ))
                fig_sharpe.add_hline(y=0, line_dash="dot", line_color="#3a4a60")
                fig_sharpe = style_fig(fig_sharpe, height=260)
                fig_sharpe.update_yaxes(title_text="Rolling Sharpe")
                st.plotly_chart(fig_sharpe, use_container_width=True)

            with st.expander("📋 Full Performance Metrics Table"):
                perf_df = pd.DataFrame(
                    {k: [v] for k, v in perf_metrics.items()}
                ).T.rename(columns={0: "Value"})
                st.dataframe(perf_df, use_container_width=True)

        # ============================= TAB 2 =============================
        with tab2:
            colC, colD = st.columns([1, 1.3])

            w_df = opt_result.weights.to_frame(name="Weight")
            w_df = w_df[w_df["Weight"] > 0.001].sort_values("Weight", ascending=False)

            with colC:
                st.markdown(f"##### Optimal Allocation ({objective.replace('_', ' ').title()})")
                fig_donut = go.Figure(data=[go.Pie(
                    labels=w_df.index, values=w_df["Weight"],
                    hole=0.62,
                    marker=dict(colors=PALETTE, line=dict(color="#0a0e14", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=12, color="#e8edf4"),
                    hovertemplate="%{label}<br>%{percent}<extra></extra>",
                )])
                fig_donut.update_layout(
                    annotations=[dict(text="Weights", x=0.5, y=0.5, font_size=14,
                                       font_color="#8694a8", showarrow=False)]
                )
                fig_donut = style_fig(fig_donut, height=380)
                st.plotly_chart(fig_donut, use_container_width=True)

            with colD:
                st.markdown("##### Weight Breakdown")
                fig_bar = px.bar(
                    w_df.reset_index(), x=w_df.reset_index().columns[0], y="Weight",
                    color=w_df.reset_index().columns[0],
                    color_discrete_sequence=PALETTE,
                    text_auto=".1%",
                )
                fig_bar.update_traces(textposition="outside", showlegend=False)
                fig_bar = style_fig(fig_bar, height=380)
                fig_bar.update_yaxes(title_text="Weight", tickformat=".0%")
                fig_bar.update_xaxes(title_text="")
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("##### Asset Descriptive Statistics")
            stats = descriptive_statistics(simple_returns)
            stats_df = pd.DataFrame({
                "Ann. Mean (%)": stats["annualized_mean"] * 100,
                "Ann. Vol (%)": stats["annualized_volatility"] * 100,
            })
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

        # ============================= TAB 3 =============================
        with tab3:
            st.markdown("##### Extreme Value Theory (EVT) — Peaks Over Threshold")
            try:
                evt_stats = run_evt(port_returns, evt_confidence, evt_threshold_q)
                ecol1, ecol2, ecol3 = st.columns(3)
                ecol1.metric(f"{int(evt_confidence*100)}% EVT VaR (Daily Loss)",
                             fmt_pct(evt_stats["evt_var"]))
                ecol2.metric(f"{int(evt_confidence*100)}% EVT Expected Shortfall",
                             fmt_pct(evt_stats["evt_es"]))
                ecol3.metric("GPD Shape Parameter (ξ)", fmt_num(evt_stats["shape"], 4))

                st.info(
                    "💡 **EVT Insights:** Peak-Over-Threshold (POT) fitted to a Generalized "
                    "Pareto Distribution estimates extreme tail risk beyond what normal "
                    "historical sampling captures — critical for stress-aware capital allocation."
                )

                st.markdown("###### Return Distribution vs. Tail Threshold")
                threshold_val = port_returns.quantile(1 - evt_threshold_q) if "threshold" not in evt_stats \
                    else evt_stats.get("threshold")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=port_returns * 100, nbinsx=80,
                    marker_color=COLOR_BLUE, opacity=0.65, name="Daily Returns",
                ))
                if threshold_val is not None:
                    fig_hist.add_vline(x=threshold_val * 100, line_dash="dash",
                                        line_color=COLOR_RED,
                                        annotation_text="POT Threshold",
                                        annotation_font_color=COLOR_RED)
                fig_hist = style_fig(fig_hist, height=340)
                fig_hist.update_xaxes(title_text="Daily Return (%)")
                fig_hist.update_yaxes(title_text="Frequency")
                st.plotly_chart(fig_hist, use_container_width=True)

            except ValueError as e:
                st.warning(f"⚠️ EVT fitting failed (likely insufficient tail data): {e}")
            except Exception as e:
                st.warning(f"⚠️ EVT module raised an unexpected error: {e}")

        # ============================= TAB 4 =============================
        with tab4:
            st.markdown("##### Historical Scenario Replay (Buy-and-Hold)")
            stress_results = []
            for name, (s_start, s_end) in HISTORICAL_SCENARIOS.items():
                try:
                    res = historical_stress(simple_returns, opt_result.weights, s_start, s_end, initial_capital)
                    stress_results.append({
                        "Scenario": name.replace("_", " ").title(),
                        "Start": res["start"],
                        "End": res["end"],
                        "Return (%)": res["period_return"] * 100,
                        "Max Drawdown (%)": res["maximum_drawdown"] * 100,
                    })
                except Exception:
                    continue

            if stress_results:
                stress_df = pd.DataFrame(stress_results)

                fig_stress = go.Figure()
                fig_stress.add_trace(go.Bar(
                    x=stress_df["Scenario"], y=stress_df["Return (%)"],
                    name="Period Return",
                    marker_color=[COLOR_GREEN if v >= 0 else COLOR_RED for v in stress_df["Return (%)"]],
                ))
                fig_stress = style_fig(fig_stress, height=340)
                fig_stress.update_yaxes(title_text="Return (%)")
                st.plotly_chart(fig_stress, use_container_width=True)

                st.dataframe(
                    stress_df.style.format({
                        "Return (%)": "{:.2f}",
                        "Max Drawdown (%)": "{:.2f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.warning("⚠️ Insufficient data range for default historical scenarios.")

    except Exception as e:
        st.error(f"🔴 Pipeline Error: {e}")
        st.stop()