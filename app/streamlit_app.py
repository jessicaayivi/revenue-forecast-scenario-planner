"""Revenue scenario planner. Run from the repo root: streamlit run app/streamlit_app.py

All logic lives in src/scenarios.py; this file only handles controls and display.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config as cfg, scenarios  # noqa: E402

P = cfg.PROCESSED_DIR
BLUE, ORANGE, MUTED, GRID, INK2 = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#52514e"
MKT_COLOR = {"SP": "#2a78d6", "Rest of Brazil": "#eb6834", "South": "#1baf7a", "RJ": "#eda100", "MG": "#e87ba4"}

st.set_page_config(page_title="Revenue Scenario Planner", layout="wide")


@st.cache_data
def load():
    base = scenarios.load_baseline()
    presets = scenarios.preset_scenarios(base)
    hist = pd.read_csv(P / "monthly_total.csv")
    hist = hist[hist["in_model_window"]]
    return (base, presets, scenarios.black_friday_uplift(), hist,
            scenarios.market_scorecard(), scenarios.category_scorecard())


base, presets, uplift, hist, mkt_sc, cat_sc = load()


def apply_preset():
    p = presets[st.session_state["preset"]]
    for m in cfg.MARKETS:
        st.session_state[f"gc_{m}"] = round(float(p.loc[m, "g_customers"]) * 100, 2)
        st.session_state[f"ga_{m}"] = round(float(p.loc[m, "g_aov"]) * 100, 2)


if "preset" not in st.session_state:
    st.session_state["preset"] = "Base"
    apply_preset()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Scenario")
    st.selectbox("Start from a preset", cfg.SCENARIOS, key="preset", on_change=apply_preset)
    st.caption(scenarios.DESCRIPTIONS[st.session_state["preset"]])
    bf = st.toggle(f"Black Friday uplift in November (x{uplift:.2f})", value=True,
                   help="From one November (2017). A planning assumption, not a learned effect.")
    st.subheader("Monthly growth by market (%)")
    for m in cfg.MARKETS:
        with st.expander(m, expanded=False):
            st.slider("Customers, % per month", -8.0, 8.0, step=0.01, format="%.2f", key=f"gc_{m}")
            st.slider("AOV, % per month", -4.0, 4.0, step=0.01, format="%.2f", key=f"ga_{m}")
    st.caption("Orders per customer is held at its Jun-Aug 2018 level (about 1.007): it barely moved in the data.")

sliders = pd.DataFrame({
    "g_customers": [st.session_state[f"gc_{m}"] / 100 for m in cfg.MARKETS],
    "g_aov": [st.session_state[f"ga_{m}"] / 100 for m in cfg.MARKETS],
}, index=cfg.MARKETS)
# Sliders show rates rounded to 0.01 pt; while they still match the preset, use its exact rates so the
# app shows the same numbers as the notebooks.
preset = presets[st.session_state["preset"]]
edited = not ((preset * 100).round(2) - sliders * 100).abs().lt(1e-9).all().all()
params = sliders if edited else preset
scenario_label = f"{st.session_state['preset']} (edited)" if edited else st.session_state["preset"]
proj = scenarios.project(params, black_friday=bf, base=base, uplift=uplift)
ref = scenarios.project(presets["Base"], black_friday=bf, base=base, uplift=uplift)
summ = scenarios.summarise(proj).set_index("market")
ref_summ = scenarios.summarise(ref).set_index("market")

# ------------------------------------------------------------------ header + KPIs
st.title("Revenue Scenario Planner")
st.caption("Demonstration project using public data (Olist). Not affiliated with any employer. "
           "Plan period: Sep 2018 - Feb 2019. Revenue = item price on delivered orders, R$.")

tot, rtot = summ.loc["Total"], ref_summ.loc["Total"]
st.markdown(f"**Scenario: {scenario_label}**")
vs_flat = tot["six_months"] / rtot["six_months"] - 1
last6 = hist.tail(6)["revenue"].sum()
k1, k2 = st.columns(2), st.columns(2)
k1[0].metric("6-month revenue", f"R$ {tot['six_months']/1e6:.2f}M",
             f"{vs_flat:+.1%} vs flat" if abs(vs_flat) >= 0.0005 else None)
k1[1].metric("vs Mar-Aug 2018 actual", f"{tot['six_months']/last6-1:+.1%}", help=f"R$ {last6/1e6:.2f}M actual")
k2[0].metric("Plan Q1 (Sep-Nov)", f"R$ {tot['plan_q1']/1e6:.2f}M")
k2[1].metric("Plan Q2 (Dec-Feb)", f"R$ {tot['plan_q2']/1e6:.2f}M")

tab_plan, tab_where, tab_about = st.tabs(["Plan", "Where to invest", "Method"])

with tab_plan:
    m_tot = proj.groupby("month", as_index=False)["revenue"].sum()
    r_tot = ref.groupby("month", as_index=False)["revenue"].sum()
    fig = go.Figure()
    fig.add_scatter(x=hist["month"], y=hist["revenue"], name="Actual", mode="lines+markers",
                    line=dict(color=MUTED, width=2), marker=dict(size=5),
                    hovertemplate="%{x}<br>R$ %{y:,.0f}<extra>Actual</extra>")
    fig.add_scatter(x=r_tot["month"], y=r_tot["revenue"], name="Flat baseline", mode="lines",
                    line=dict(color=MUTED, width=2, dash="dash"),
                    hovertemplate="%{x}<br>R$ %{y:,.0f}<extra>Flat baseline</extra>")
    fig.add_scatter(x=m_tot["month"], y=m_tot["revenue"], name="This scenario", mode="lines+markers",
                    line=dict(color=BLUE, width=2.5), marker=dict(size=8),
                    hovertemplate="%{x}<br>R$ %{y:,.0f}<extra>This scenario</extra>")
    months = list(hist["month"]) + list(m_tot["month"])
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=80, b=10), plot_bgcolor="white",
                      title=dict(text="Monthly revenue, total", y=0.97, x=0, xanchor="left"),
                      legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0),
                      yaxis=dict(tickprefix="R$ ", gridcolor=GRID, rangemode="tozero"),
                      xaxis=dict(type="category", tickvals=months[::3], tickangle=0),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({
        "Customers %/month": params["g_customers"] * 100,
        "AOV %/month": params["g_aov"] * 100,
        "Plan Q1 (R$k)": summ.loc[cfg.MARKETS, "plan_q1"] / 1e3,
        "Plan Q2 (R$k)": summ.loc[cfg.MARKETS, "plan_q2"] / 1e3,
        "6 months (R$k)": summ.loc[cfg.MARKETS, "six_months"] / 1e3,
        "vs flat (R$k)": (summ.loc[cfg.MARKETS, "six_months"] - ref_summ.loc[cfg.MARKETS, "six_months"]) / 1e3,
    })
    table.loc["Total"] = [None, None, tot["plan_q1"] / 1e3, tot["plan_q2"] / 1e3, tot["six_months"] / 1e3,
                          (tot["six_months"] - rtot["six_months"]) / 1e3]
    st.dataframe(table.style.format("{:,.1f}", na_rep=""), use_container_width=True)
    st.download_button("Download monthly projection (CSV)", proj.round(2).to_csv(index=False),
                       "scenario_projection.csv", "text/csv")

with tab_where:
    st.subheader("Markets")
    st.caption("Extra 6-month revenue from +1 point of monthly customer growth shows where the same "
               "lift is worth most. Momentum columns compare Mar-Aug 2018 with Sep 2017-Feb 2018.")
    ms = mkt_sc.set_index("market")
    fig2 = go.Figure(go.Bar(
        x=ms["extra_6m_revenue_per_1pt_customers"], y=ms.index, orientation="h",
        marker_color=[MKT_COLOR[m] for m in ms.index],
        hovertemplate="%{y}<br>+R$ %{x:,.0f} over 6 months<extra></extra>"))
    fig2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                       title="Extra 6-month revenue per +1 pt monthly customer growth",
                       xaxis=dict(tickprefix="R$ ", gridcolor=GRID), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)
    pct_cols = ["revenue_share", "revenue_growth_yoy", "revenue_growth_last6m", "customers_growth_last6m",
                "aov_growth_last6m", "customers_growth_last3m", "share_of_recent_growth"]
    show = ms.copy()
    show[pct_cols] = show[pct_cols] * 100
    show = show.rename(columns={
        "revenue_share": "Share %", "baseline_revenue_month": "Revenue/month R$", "aov": "AOV R$",
        "revenue_growth_yoy": "Revenue YoY %", "revenue_growth_last6m": "Revenue last 6m %",
        "customers_growth_last6m": "Customers last 6m %", "aov_growth_last6m": "AOV last 6m %",
        "customers_growth_last3m": "Customers Jun-Aug vs Mar-May %",
        "share_of_recent_growth": "Share of recent growth %",
        "extra_6m_revenue_per_1pt_customers": "R$ per +1 pt customers"})
    st.dataframe(show.style.format("{:,.1f}"), use_container_width=True)
    st.subheader("Categories")
    st.caption("YoY = Jan-Aug 2018 vs Jan-Aug 2017. Last 6m = Mar-Aug 2018 vs Sep 2017-Feb 2018.")
    cs = cat_sc.set_index("category")
    pc = ["revenue_share_last6m", "revenue_growth_yoy", "revenue_growth_last6m", "aov_growth_last6m"]
    cs[pc] = cs[pc] * 100
    cs = cs.rename(columns={
        "revenue_share_last6m": "Share last 6m %", "revenue_last6m": "Revenue last 6m R$",
        "aov_last6m": "AOV R$", "revenue_growth_yoy": "Revenue YoY %",
        "revenue_growth_last6m": "Revenue last 6m %", "aov_growth_last6m": "AOV last 6m %"})
    st.dataframe(cs.style.format("{:,.1f}"), use_container_width=True)

with tab_about:
    st.markdown(f"""
**Engine.** For each market and month t: customers x (1 + customer growth)^t x orders per customer x
AOV x (1 + AOV growth)^t, from Jun-Aug 2018 averages. With all growth at 0 it equals the Stage 3
baseline forecast (3-month average, chosen by backtest).

**Presets** come from the data:
- *Downside*: {scenarios.DESCRIPTIONS['Downside']} (Jun-Aug vs Mar-May 2018).
- *Base*: {scenarios.DESCRIPTIONS['Base']}.
- *Upside*: {scenarios.DESCRIPTIONS['Upside']} (Mar-Aug 2018 vs Sep 2017-Feb 2018).

**Limits.** 20 months of history, one November, no cost or marketing data: the planner shows revenue
sensitivity, not return on investment.
""")
