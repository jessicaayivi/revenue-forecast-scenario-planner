"""Revenue scenario planner. Run from the repo root: streamlit run app/streamlit_app.py

All logic lives in src/scenarios.py; this file only handles controls and display.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config as cfg, scenarios  # noqa: E402

P = cfg.PROCESSED_DIR
REPO_URL = "https://github.com/jessicaayivi/revenue-forecast-scenario-planner"
BLUE, ORANGE, MUTED, GRID, INK2, LIGHT = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#52514e", "#c3c2b7"
MKT_COLOR = {"SP": "#2a78d6", "Rest of Brazil": "#eb6834", "South": "#1baf7a", "RJ": "#eda100", "MG": "#e87ba4"}
MKT_STATES = {"SP": "São Paulo", "RJ": "Rio de Janeiro", "MG": "Minas Gerais",
              "South": "Paraná, Santa Catarina, Rio Grande do Sul", "Rest of Brazil": "the other 21 states"}

st.set_page_config(page_title="Revenue Scenario Planner", layout="wide")


@st.cache_data
def load():
    base = scenarios.load_baseline()
    presets = scenarios.preset_scenarios(base)
    hist = pd.read_csv(P / "monthly_total.csv")
    hist = hist[hist["in_model_window"]]
    states = pd.read_csv(P / "state_summary.csv")
    dq = json.loads((P / "data_quality.json").read_text())
    scores = pd.read_csv(P / "backtest_scores.csv")
    preset_totals = {n: scenarios.summarise(scenarios.project(presets[n], base=base))
                     .set_index("market").loc["Total", "six_months"] for n in cfg.SCENARIOS}
    check = pd.read_csv(P / "forecast_check.csv")
    check_paths = pd.read_csv(P / "forecast_check_paths.csv")
    fsum = json.loads((P / "forecast_summary.json").read_text())
    return (base, presets, scenarios.black_friday_uplift(), hist, scenarios.market_scorecard(),
            scenarios.category_scorecard(), states, dq, scores, preset_totals, check, check_paths, fsum)


(base, presets, uplift, hist, mkt_sc, cat_sc, states, dq, scores, preset_totals,
 check, check_paths, fsum) = load()
MODEL_NAMES = {"WindowAverage": "3-month average", "Naive": "Last month repeated", "AutoTheta": "AutoTheta",
               "AutoETS": "AutoETS (exponential smoothing)", "RWD": "Random walk with drift",
               "Drivers": "Driver-based (customers x orders/customer x AOV)",
               "Prophet": "Prophet (daily, seasonality, Black Friday)",
               "LagRegression": "Regression on revenue + customer momentum",
               "ProphetKPI": "Prophet + forecast KPIs (customers, orders/customer, AOV)",
               "ProphetLead": "Prophet + leading indicators (late deliveries, cancellations, new customers)"}
N_MODELS = len(MODEL_NAMES)
PLAN_MONTHS = scenarios.forecast_months()  # months still to forecast (the rest of the plan is actuals)


def _span(a, b):
    return f"{pd.Period(a, 'M').strftime('%b %Y')} – {pd.Period(b, 'M').strftime('%b %Y')}"


HIST_LABEL = cfg.HISTORY_LABEL
PLAN_LABEL = cfg.PLAN_LABEL            # e.g. Jul–Dec 2018
PLAN = cfg.PLAN_NAME                    # e.g. H2 2018
FC_LABEL = cfg.FORECAST_LABEL           # e.g. Sep–Dec 2018
ACT_LABEL = cfg.ACTUAL_LABEL            # e.g. Jul–Aug 2018
Q1, Q2 = cfg.PLAN_Q1_LABEL, cfg.PLAN_Q2_LABEL


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
    with st.expander("How to use these controls", expanded=False):
        st.markdown(f"""
These controls build the **{cfg.PLAN_NAME} plan** shown on every tab.

1. **Pick a preset**: a story about what happens next (Downside, Base or Upside).
2. **Black Friday**: switch the November boost on or off.
3. **Fine-tune a market** (optional): open it and move its sliders. The scenario then shows *(edited)*.

Every number on the page updates as you go. Choose a preset again to reset the sliders.
To test a single idea instead ("what if prices rise 5%?"), use the **What-if** tab.
""")

    st.radio("Show amounts in", ["R$", "€"], horizontal=True, key="currency",
             help=f"Brazilian real or euro. Euro amounts use one constant rate: R$ {cfg.BRL_PER_EUR} = €1 "
                  f"({cfg.FX_SOURCE}), so exchange-rate swings do not distort growth.")
    st.caption("Display only: it does not change the plan.")

    st.header("Scenario")
    st.caption(f"Jul–Aug 2018 are actual sales; a scenario only changes the forecast months ({cfg.FORECAST_LABEL}).")
    st.selectbox("Start from a preset", cfg.SCENARIOS, key="preset", on_change=apply_preset,
                 help="A preset fills in the market sliders below with growth rates measured in the data.")
    st.caption(scenarios.DESCRIPTIONS[st.session_state["preset"]])
    _fx = 1.0 if st.session_state["currency"] == "R$" else 1 / cfg.BRL_PER_EUR
    _sym = st.session_state["currency"]
    _desc = {"Downside": "customers keep falling as in the last quarter",
             "Base": "sales stay at the recent level",
             "Upside": "last 6 months' growth returns"}
    st.markdown("\n".join(
        f"- **{n}**: {_desc[n]} → {_sym} {preset_totals[n] * _fx / 1e6:.2f}M" for n in cfg.SCENARIOS
    ).replace("$", "\\$"))

    bf = st.toggle(f"Black Friday uplift in November (x{uplift:.2f})", value=True,
                   help="From one November (2017). A planning assumption, not a learned effect.")
    st.caption(f"On: November gets {uplift - 1:.0%} more revenue than a normal month, as in Nov 2017. "
               "Off: November is a normal month. It rests on a single November, so it is a switch.")

    st.subheader("Monthly growth by market (%)")
    st.caption("Optional fine-tuning. Each market has two sliders, set by the preset: how fast its number of "
               "**customers** and its **average order value (AOV)** grow or shrink **each month**. "
               "Example: SP customers at +2.00 means 2% more SP customers every month.")
    for m in cfg.MARKETS:
        with st.expander(m, expanded=False):
            st.slider("Customers, % per month", -8.0, 8.0, step=0.01, format="%.2f", key=f"gc_{m}",
                      help="Monthly growth in the number of customers in this market. 0 = stays at its "
                           f"{cfg.BASELINE_LABEL} level.")
            st.slider("AOV, % per month", -4.0, 4.0, step=0.01, format="%.2f", key=f"ga_{m}",
                      help="Monthly growth in the average order value (revenue per order) in this market.")
    st.caption(f"No slider for orders per customer: almost everyone orders once, so it is held at its "
               f"{cfg.BASELINE_LABEL} level (about 1.007).")

# ------------------------------------------------------------------ currency helpers
SYM = st.session_state["currency"]
FX = 1.0 if SYM == "R$" else 1 / cfg.BRL_PER_EUR


def mm(v, d=2):
    """Money in millions, in the chosen currency."""
    return f"{SYM} {v * FX / 1e6:.{d}f}M"


def kk(v, sign=False):
    """Money in thousands, in the chosen currency. Amounts that round to zero show as 0k (no +/-)."""
    if abs(v * FX) < 500:
        return f"{SYM} 0k"
    return f"{SYM} {v * FX / 1e3:{'+' if sign else ''},.0f}k"


def md(text, caption=False):
    """Markdown with "$" escaped: Streamlit renders a pair of "$" as maths."""
    (st.caption if caption else st.markdown)(text.replace("$", "\\$"))


def cur_axis(**kw):
    return dict(tickprefix=f"{SYM} ", gridcolor=GRID, **kw)


HOVER = f"%{{x}}<br>{SYM} %{{y:,.0f}}"

# ------------------------------------------------------------------ scenario
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
md(f"Brazilian e-commerce marketplace (Olist, public data). **History: {HIST_LABEL}** → "
   f"**Plan: {PLAN} ({PLAN_LABEL})** = {ACT_LABEL} actual + {FC_LABEL} forecast. Amounts in **{SYM}**"
   + ("" if SYM == "R$" else f" (R$ {cfg.BRL_PER_EUR} = €1, {cfg.FX_SOURCE})")
   + ". New here? Open **Start here**.")
st.caption("Demonstration project using public data (Olist). Not affiliated with any employer.")

tot, rtot = summ.loc["Total"], ref_summ.loc["Total"]
st.markdown(f"**Scenario: {scenario_label}**")
vs_flat = tot["six_months"] / rtot["six_months"] - 1
prev_half = hist[hist["month"].between(*cfg.PREV_HALF)]["revenue"].sum()
k1, k2 = st.columns(2), st.columns(2)
k1[0].metric(f"{PLAN} revenue", mm(tot["six_months"]),
             f"{vs_flat:+.1%} vs flat" if abs(vs_flat) >= 0.0005 else None,
             help=f"{PLAN_LABEL}: {mm(tot['actual'])} actual ({ACT_LABEL}) + {mm(tot['forecast'])} forecast ({FC_LABEL}).")
k1[1].metric(f"vs {cfg.PREV_HALF_NAME} actual", f"{tot['six_months']/prev_half-1:+.1%}",
             help=f"{cfg.PREV_HALF_NAME}: {mm(prev_half)} actual")
k2[0].metric(Q1, mm(tot["plan_q1"]), help=f"Includes {ACT_LABEL} actuals.")
k2[1].metric(Q2, mm(tot["plan_q2"]))

tab_start, tab_plan, tab_whatif, tab_where, tab_check, tab_hist, tab_data, tab_about = st.tabs(
    ["Start here", "Plan", "What-if", "Where to invest", "Forecast check", "History", "Data", "Method"])

# ------------------------------------------------------------------ start here
with tab_start:
    st.subheader("What this app does")
    md("It answers a planning question for an online marketplace in Brazil: **how much revenue should we plan "
       f"for {PLAN}, and which markets or product categories deserve extra growth effort?** "
       f"It is built on {cfg.HISTORY_MONTHS} months of real orders and turns them into a forecast, three scenarios and a "
       "what-if tool you can steer yourself.")

    st.markdown("**The time frame**")
    ps = pd.Period(cfg.PLAN_START, "M")
    end = pd.Period(cfg.MODEL_END, "M")
    tl = pd.DataFrame([
        dict(Phase="Too few orders (excluded)", Start="2016-09-01", Finish=pd.Period(cfg.MODEL_START, "M").start_time,
             Label="2016: test period", Row="Data"),
        dict(Phase="History used by the model", Start=pd.Period(cfg.MODEL_START, "M").start_time,
             Finish=(end + 1).start_time, Label=f"History: {HIST_LABEL} ({cfg.HISTORY_MONTHS} months)", Row="Data"),
        dict(Phase="Plan: actual", Start=ps.start_time, Finish=(end + 1).start_time, Label="", Row=PLAN),
        dict(Phase="Plan: forecast", Start=(end + 1).start_time, Finish=(pd.Period(cfg.PLAN_END, "M") + 1).start_time,
             Label="", Row=PLAN),
    ])
    ftl = px.timeline(tl, x_start="Start", x_end="Finish", y="Row", color="Phase", text="Label",
                      color_discrete_map={"Too few orders (excluded)": "#e1e0d9",
                                          "History used by the model": "#86b6ef",
                                          "Plan: actual": "#86b6ef", "Plan: forecast": BLUE})
    ftl.update_traces(textposition="inside", insidetextanchor="middle", textfont_color="#0b0b0b",
                      textangle=0, constraintext="none")
    # the plan bars are too short for text: label them from the left
    ftl.add_annotation(x=ps.start_time, y=PLAN, xanchor="right", showarrow=False, font=dict(size=11, color=INK2),
                       text=f"Plan {PLAN}: <b>{ACT_LABEL} actual</b> (light) + <b>{FC_LABEL} forecast</b> (dark)  ")
    ftl.update_layout(height=190, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", showlegend=False,
                      yaxis=dict(title=None, categoryorder="array", categoryarray=[PLAN, "Data"]),
                      xaxis=dict(dtick="M3", tickformat="%b %Y", gridcolor=GRID))
    st.plotly_chart(ftl, use_container_width=True)
    md(f"The data runs from Sep 2016 to Oct 2018. 2016 has only {dq['delivered_orders_before_model_window']} "
       f"delivered orders and Sep-Oct 2018 none yet delivered, so the model learns from {HIST_LABEL}. The plan is "
       f"the calendar half-year {PLAN}: {ACT_LABEL} is already known, {FC_LABEL} is forecast (the usual "
       "\"actuals to date + forecast\" outlook). Dates are the real ones from the dataset.", caption=True)

    st.markdown("**Where the customers are: five markets**")
    cmap, ctab = st.columns([3, 2])
    s = states.copy()
    s["size"] = 7 + 38 * np.sqrt(s["revenue"] / s["revenue"].max())
    fmap = go.Figure()
    for mk in cfg.MARKETS:
        d = s[s["market"] == mk]
        fmap.add_scattergeo(
            lat=d["lat"], lon=d["lon"], name=mk, mode="markers+text",
            text=np.where(d["revenue_share"] >= 0.02, d["state"], ""), textposition="middle right",
            textfont=dict(size=10, color=INK2),
            marker=dict(size=d["size"], color=MKT_COLOR[mk], opacity=0.85, line=dict(color="white", width=1)),
            customdata=np.stack([d["state_name"], d["revenue"] * FX / 1e6, d["revenue_share"] * 100,
                                 d["revenue_growth_last6m"] * 100], axis=-1),
            hovertemplate=(f"<b>%{{customdata[0]}}</b> ({mk})<br>Revenue {HIST_LABEL}: {SYM} "
                           "%{customdata[1]:.2f}M<br>Share: %{customdata[2]:.1f}%<br>"
                           "Last 6 months vs prior 6: %{customdata[3]:+.0f}%<extra></extra>"))
    fmap.update_geos(showcountries=True, countrycolor=LIGHT, countrywidth=1,
                     showland=True, landcolor="#f6f5f1", showocean=True, oceancolor="#eef4fb",
                     showlakes=False, showframe=False, showcoastlines=True, coastlinecolor=LIGHT,
                     projection_type="mercator", resolution=50,
                     lonaxis_range=[-75, -33], lataxis_range=[-35, 7])  # Brazil's bounding box
    fmap.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0),
                       legend=dict(orientation="h", y=-0.02, x=0, title_text="Market  "))
    with cmap:
        st.plotly_chart(fmap, use_container_width=True)
        st.caption("One bubble per state, placed at its capital; size = revenue Jan 2017 - Aug 2018.")
    with ctab:
        share = s.groupby("market")["revenue_share"].sum()
        mt = pd.DataFrame({"Share %": (share.loc[cfg.MARKETS] * 100).round(1),
                           "States": [MKT_STATES[m] for m in cfg.MARKETS]}, index=cfg.MARKETS)
        st.dataframe(mt.style.format({"Share %": "{:.1f}"}), use_container_width=True)
        md(f"The three largest states stand alone; the South is grouped; the remaining 21 states are pooled. "
           f"The top 5 states make up {dq['top5_state_revenue_share']:.0%} of revenue.", caption=True)

    st.markdown("**How to use it**")
    md(f"""
1. **Pick a scenario** in the sidebar: *Downside*, *Base* or *Upside*. Each comes from growth rates measured in
   the data. The headline numbers and every tab update.
2. **Plan** tab: monthly revenue for {PLAN} next to the history, and the plan by quarter and market.
3. **What-if** tab: test a hypothesis, e.g. "+10% customers in SP from October" or "raise prices 5%, and lose
   some customers". It shows the revenue impact and where the same change is worth most.
4. **Where to invest** tab: market and category scorecards: size, momentum and order value.
5. **Forecast check** tab: {N_MODELS} forecasting models compared, and what would have happened if we had planned in
   June 2018.
6. **History** tab: the monthly history of every metric (revenue, customers, AOV, late deliveries...) by market
   and category, with tables to download. **Data** tab: where the data comes from, how it was cleaned, and a
   glossary of every term.
7. Fine-tune: open a market in the sidebar to set its monthly customer and AOV growth; switch Black Friday on
   or off; show amounts in R$ or €.
""")

    st.markdown("**What the data says**")
    mk = mkt_sc.set_index("market")
    md(f"""
- Plan **{mm(preset_totals['Base'])}** for {PLAN} ({ACT_LABEL} actual + {FC_LABEL} flat at the {cfg.BASELINE_LABEL}
  run rate, plus Black Friday); downside {mm(preset_totals['Downside'])}, upside {mm(preset_totals['Upside'])}.
- Growth came from **new customers**, not repeat buyers: only {dq['repeat_rate_model_window']:.0%} of customers
  ordered more than once. And it had stalled: monthly customers fell in {cfg.BASELINE_LABEL} in every market.
- A simple flat forecast beat {N_MODELS - 1} alternatives, including Prophet and KPI-based models (see **Forecast check**).
- **SP** is {mk.loc['SP', 'revenue_share']:.0%} of revenue and the biggest lever; **South** and **Rest of Brazil**
  grew through order value; **RJ** shrank.
- Full reasoning: [decision memo]({REPO_URL}/blob/main/docs/decision_memo.md) · code: [GitHub]({REPO_URL}).
""")

# ------------------------------------------------------------------ what-if
with tab_whatif:
    st.subheader("Test a hypothesis")
    st.caption(f"See the effect on {PLAN} revenue (changes apply to the forecast months, {FC_LABEL}; {ACT_LABEL} "
               f"are actuals), on top of the **{scenario_label}** scenario selected "
               "in the sidebar. Revenue = customers x orders per customer x items per order x price per item "
               "(the last two make up AOV).")
    mode = st.radio("What do you want to test?", ["Change one driver", "Price vs volume"], horizontal=True,
                    key="wi_mode")
    c1, c2 = st.columns(2)
    if mode == "Change one driver":
        lever = c1.selectbox("Driver to change", list(scenarios.LEVERS), format_func=scenarios.LEVERS.get,
                             key="wi_lever")
        change = c2.slider("Change (%)", -30, 50, 10, step=1, key="wi_change",
                           help="A one-off step up (or down) in the level of the driver, held from the start month.")
    else:
        price = c1.slider("Price change (%)", -20, 20, 5, step=1, key="wi_price",
                          help="Change in price per item; basket size is unchanged.")
        elasticity = c2.slider("Customer response (elasticity)", 0.0, 3.0, 1.0, step=0.1, key="wi_elasticity",
                               help="How many % of customers you lose for each 1% price rise (and gain for each "
                                    "1% cut). 0 = nobody reacts, 1 = one-for-one, 2 = twice as strong. "
                                    "The data has no price tests, so this is your assumption.")
    c3, c4 = st.columns(2)
    wi_markets = c3.multiselect("Markets", cfg.MARKETS, default=cfg.MARKETS, key="wi_markets")
    start = c4.selectbox("Starting from", PLAN_MONTHS, key="wi_start",
                         format_func=lambda m: pd.Period(m, "M").strftime("%b %Y"))
    if mode == "Price vs volume":
        with st.expander("How to read this, and how it is calculated", expanded=False):
            md("""
**The question:** if we raise prices, do we make more money, or lose so many customers that revenue falls?

- **Price change:** every item gets this much more (or less) expensive. Basket size stays the same.
- **Customer response (elasticity):** your assumption about how strongly customers react. It means *"for
  every 1% price rise, this many % of customers leave"*: **0** = nobody reacts, **1** = one-for-one,
  **2** = twice as fast as prices rise.
- **Markets / Starting from:** where and from which month the new price applies.

**The rule:** a price rise grows revenue only if the elasticity is **below 1**, meaning customers react less
than one-for-one. At exactly 1 the extra price and the lost customers cancel out (break-even). A price cut
works the other way: it grows revenue only if the elasticity is **above 1**.""")

            # worked calculation with the values currently selected
            p, e = price / 100, elasticity
            pf, cf = 1 + p, (1 + p) ** (-e)
            rf = pf * cf
            hit = proj["market"].isin(wi_markets) & (proj["month"] >= start) if wi_markets else proj["month"] < ""
            affected = proj.loc[hit, "revenue"].sum()
            where_ = "all markets" if len(wi_markets) == len(cfg.MARKETS) else (", ".join(wi_markets) or "no market")
            span = _span(start, PLAN_MONTHS[-1])
            md(f"""
**How it is calculated, with your settings** ({price:+d}% price, elasticity {e:.1f}, {where_}, {span}):

Revenue = customers x orders per customer x items per order x **price per item**. Only price and customers
change; the other two stay as they are.

1. **Price factor** = 1 + {price:+d}% = **{pf:.3f}**
2. **Customer factor** = (price factor) to the power of minus the elasticity = {pf:.3f}^(-{e:.1f}) =
   **{cf:.3f}**, so customers move **{cf - 1:+.1%}**
3. **Revenue factor** = price factor x customer factor = {pf:.3f} x {cf:.3f} = **{rf:.3f}**, so revenue
   moves **{rf - 1:+.1%}**. (Equivalently {pf:.3f}^(1 - {e:.1f}): it equals 1 exactly when the elasticity
   is 1, which is why 1 is the break-even point.)
4. **Revenue affected:** the {scenario_label} scenario has **{mm(affected)}** in {where_} over {span}. Other
   markets and months are unchanged.
5. **Impact** = {mm(affected)} x ({rf:.3f} - 1) = **{f"{SYM} 0k (no change)" if abs(affected * (rf - 1)) < 500
   else kk(affected * (rf - 1), sign=True)}** for {PLAN}.

Step 2 uses a *constant-elasticity* demand curve, a standard way to model price response: every 1% move in
price moves customers by the same %, however large the price change.""")
            ex = pd.DataFrame({"Elasticity": [0.5, 1.0, 1.5, 2.0]})
            ex["Customers"] = (1.05 ** -ex["Elasticity"] - 1).map("{:+.1%}".format)
            ex["Revenue"] = (1.05 ** (1 - ex["Elasticity"]) - 1).map("{:+.1%}".format)
            md("For comparison, a **+5% price rise** under different customer responses:")
            st.dataframe(ex.set_index("Elasticity"), use_container_width=False)
            md("""
**Two caveats.** The elasticity is a guess you enter: the data contains no price experiments, so use the
slider to ask *"how sensitive would customers need to be for this to fail?"* And this is **revenue, not
profit**: a price rise also improves the margin on every sale, but the data has no costs.""")
    if mode == "Change one driver" and lever == "orders_per_customer":
        st.info(f"In the data orders per customer barely moved (about {base['orders_per_customer'].mean():.3f}; "
                "only 3% of customers ordered more than once). Treat a big change here as a strong assumption.")
    if mode == "Change one driver" and lever in ("items_per_order", "price_per_item", "aov"):
        bt = base.loc[cfg.MARKETS]
        w8 = bt["customers"] * bt["orders_per_customer"]
        md(f"{cfg.BASELINE_LABEL}: {(bt['items_per_order'] * w8).sum() / w8.sum():.2f} items per order, "
           f"{SYM} {bt['revenue'].sum() / (bt['items_per_order'] * w8).sum() * FX:.0f} per item. "
           f"Basket size ranged {bt['items_per_order'].min():.2f}-{bt['items_per_order'].max():.2f} "
           f"across markets; price per item {SYM} {bt['price_per_item'].min() * FX:.0f}-"
           f"{bt['price_per_item'].max() * FX:.0f} (highest in {bt['price_per_item'].idxmax()}).", caption=True)

    if not wi_markets:
        st.warning("Pick at least one market.")
    else:
        where = "all markets" if len(wi_markets) == len(cfg.MARKETS) else ", ".join(wi_markets)
        when = pd.Period(start, "M").strftime("%b %Y")
        if mode == "Change one driver":
            hyp = scenarios.apply_hypothesis(proj, lever, change / 100, wi_markets, start)
            verb = "Raising" if change >= 0 else "Cutting"
            what = f"{verb} {scenarios.LEVERS[lever].lower()} by {abs(change)}% in {where} from {when}"
        else:
            hyp = scenarios.price_volume(proj, price / 100, elasticity, wi_markets, start)
            cust = (1 + price / 100) ** (-elasticity) - 1
            what = (f"A {price:+d}% price change in {where} from {when}, with customers moving "
                    f"{cust:+.1%} (elasticity {elasticity:.1f}),")
        imp = scenarios.hypothesis_impact(proj, hyp).set_index("market")
        t = imp.loc["Total"]
        md(f"**{what}** changes {PLAN} revenue by **{kk(t['change'], sign=True)} ({t['change_pct']:+.1%})**: "
           f"{mm(t['before'])} → {mm(t['after'])}.")
        w = st.columns(3)
        flat = abs(t["change"]) < 500  # under R$ 0.5k rounds to zero: show it as no change
        w[0].metric(f"Revenue impact, {PLAN}", f"{SYM} 0k" if flat else kk(t["change"], sign=True),
                    None if flat else f"{t['change_pct']:+.1%}")
        w[1].metric(f"Impact in {Q1}", kk(t["plan_q1_change"], sign=True))
        w[2].metric(f"Impact in {Q2}", kk(t["plan_q2_change"], sign=True))

        before = proj.groupby("month", as_index=False)["revenue"].sum()
        after = hyp.groupby("month", as_index=False)["revenue"].sum()
        f3 = go.Figure()
        f3.add_bar(x=before["month"], y=before["revenue"] * FX, name=f"{scenario_label}", marker_color=LIGHT,
                   hovertemplate=HOVER + "<extra>Before</extra>")
        f3.add_bar(x=after["month"], y=after["revenue"] * FX, name="With hypothesis", marker_color=BLUE,
                   hovertemplate=HOVER + "<extra>With hypothesis</extra>")
        f3.update_layout(height=340, barmode="group", bargap=0.3, bargroupgap=0.08, plot_bgcolor="white",
                         margin=dict(l=10, r=10, t=70, b=10),
                         title=dict(text=f"Monthly revenue, before and after ({FC_LABEL})", y=0.97, x=0,
                                    xanchor="left"),
                         legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0),
                         yaxis=cur_axis(rangemode="tozero"), xaxis=dict(type="category"), hovermode="x unified")
        st.plotly_chart(f3, use_container_width=True)

        by_mkt = imp.drop(index="Total")
        tbl = pd.DataFrame({
            f"Before ({SYM}k)": by_mkt["before"] * FX / 1e3, f"After ({SYM}k)": by_mkt["after"] * FX / 1e3,
            f"Change ({SYM}k)": by_mkt["change"] * FX / 1e3, "Change %": by_mkt["change_pct"] * 100,
        }).loc[cfg.MARKETS]
        tbl.loc["Total"] = [t["before"] * FX / 1e3, t["after"] * FX / 1e3, t["change"] * FX / 1e3,
                            t["change_pct"] * 100]
        tbl = tbl.round(1) + 0.0  # drop rounding noise such as "-0.0"
        st.dataframe(tbl.style.format("{:,.1f}"), use_container_width=True)

    if wi_markets and mode == "Price vs volume":
        st.markdown("**When does the price change pay off?**")
        be = scenarios.break_even_customer_change(price / 100)
        if price == 0:
            st.caption("With no price change there is nothing to trade off.")
        else:
            gain = "grows" if price > 0 else "falls"
            md(f"Break-even: a {price:+d}% price change leaves revenue flat if customers move **{be:+.1%}**, "
               f"an elasticity of **1.0**. Below 1.0 revenue {gain}; above 1.0 the opposite. "
               "This is revenue only: a price rise also lifts margin, but costs are not in the data.")
            curve = scenarios.price_volume_curve(proj, price / 100, wi_markets, start)
            f5 = go.Figure()
            f5.add_scatter(x=curve["elasticity"], y=curve["revenue_change"] * FX, mode="lines",
                           line=dict(color=BLUE, width=2.5), name="Revenue impact",
                           hovertemplate=f"Elasticity %{{x:.1f}}<br>{SYM} %{{y:+,.0f}}<extra></extra>")
            f5.add_scatter(x=[elasticity], y=[t["change"] * FX], mode="markers", name="Your assumption",
                           marker=dict(color=BLUE, size=12, line=dict(color="white", width=2)),
                           hovertemplate=f"Your assumption<br>Elasticity %{{x:.1f}}<br>{SYM} %{{y:+,.0f}}"
                                         "<extra></extra>")
            f5.add_vline(x=1.0, line=dict(color=MUTED, dash="dash", width=1.5),
                         annotation_text="break-even", annotation_position="top")
            f5.update_layout(height=320, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="white",
                             title=dict(text=f"{PLAN} revenue impact by customer response", y=0.97, x=0,
                                        xanchor="left"),
                             showlegend=False,
                             xaxis=dict(title="Customer response (elasticity)", gridcolor=GRID, range=[0, 3]),
                             yaxis=cur_axis(zeroline=True, zerolinecolor=LIGHT))
            st.plotly_chart(f5, use_container_width=True)

    if wi_markets and mode == "Change one driver":
        st.markdown("**Where would the same change be worth most?**")
        st.caption(f"The same {change:+d}% {scenarios.LEVERS[lever].lower()} change from "
                   f"{pd.Period(start, 'M').strftime('%b %Y')}, applied to one market at a time. Bigger markets "
                   "gain more from the same % change; costs are not in the data.")
        each = scenarios.hypothesis_by_market(proj, lever, change / 100, start).set_index("market")
        each = each.sort_values("extra_revenue", ascending=change < 0)
        f4 = go.Figure(go.Bar(
            x=each["extra_revenue"] * FX, y=each.index, orientation="h",
            marker_color=[MKT_COLOR[m] for m in each.index],
            text=[kk(v, sign=True) for v in each["extra_revenue"]], textposition="outside",
            hovertemplate=f"%{{y}}<br>{SYM} %{{x:,.0f}} in {PLAN}<extra></extra>"))
        lo, hi = min(0.0, each["extra_revenue"].min() * FX), max(0.0, each["extra_revenue"].max() * FX)
        pad = (hi - lo) * 0.22 or 1.0  # room for the value labels at the bar ends
        f4.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="white",
                         xaxis=cur_axis(zeroline=True, zerolinecolor=LIGHT,
                                        range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)]),
                         yaxis=dict(autorange="reversed"))
        st.plotly_chart(f4, use_container_width=True)

# ------------------------------------------------------------------ plan
with tab_plan:
    m_tot = proj.groupby("month", as_index=False)["revenue"].sum()
    r_tot = ref.groupby("month", as_index=False)["revenue"].sum()
    fig = go.Figure()
    fig.add_scatter(x=hist["month"], y=hist["revenue"] * FX, name="Actual (history)", mode="lines+markers",
                    line=dict(color=MUTED, width=2), marker=dict(size=5),
                    hovertemplate=HOVER + "<extra>Actual</extra>")
    fig.add_scatter(x=r_tot["month"], y=r_tot["revenue"] * FX, name="Flat baseline", mode="lines",
                    line=dict(color=MUTED, width=2, dash="dash"),
                    hovertemplate=HOVER + "<extra>Flat baseline</extra>")
    fig.add_scatter(x=m_tot["month"], y=m_tot["revenue"] * FX, name="This scenario", mode="lines+markers",
                    line=dict(color=BLUE, width=2.5), marker=dict(size=8),
                    hovertemplate=HOVER + "<extra>This scenario</extra>")
    months = list(hist["month"]) + list(m_tot["month"])
    p0 = months.index(cfg.PLAN_START)
    fig.add_vrect(x0=p0 - 0.5, x1=len(months) - 0.5, fillcolor="#eef4fb", line_width=0, layer="below")
    fig.add_annotation(x=(p0 - 1) / 2, y=1, yref="paper", text=f"History ({HIST_LABEL})",
                       showarrow=False, font=dict(color=INK2, size=12), yanchor="bottom")
    fig.add_annotation(x=p0 + (len(months) - p0 - 1) / 2, y=1, yref="paper",
                       text=f"Plan {PLAN}: {ACT_LABEL} actual + {FC_LABEL} forecast",
                       showarrow=False, font=dict(color=BLUE, size=12), yanchor="bottom")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=100, b=10), plot_bgcolor="white",
                      title=dict(text="Monthly revenue, total", y=0.97, x=0, xanchor="left"),
                      legend=dict(orientation="h", y=1.1, yanchor="bottom", x=0),
                      yaxis=cur_axis(rangemode="tozero"),
                      xaxis=dict(type="category", tickvals=months[::3], tickangle=0),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({
        "Customers %/month": params["g_customers"] * 100,
        "AOV %/month": params["g_aov"] * 100,
        f"{ACT_LABEL} actual ({SYM}k)": summ.loc[cfg.MARKETS, "actual"] * FX / 1e3,
        f"{Q1} ({SYM}k)": summ.loc[cfg.MARKETS, "plan_q1"] * FX / 1e3,
        f"{Q2} ({SYM}k)": summ.loc[cfg.MARKETS, "plan_q2"] * FX / 1e3,
        f"{PLAN} ({SYM}k)": summ.loc[cfg.MARKETS, "six_months"] * FX / 1e3,
        f"vs flat ({SYM}k)": (summ.loc[cfg.MARKETS, "six_months"] - ref_summ.loc[cfg.MARKETS, "six_months"])
        * FX / 1e3,
    })
    table.loc["Total"] = [float("nan"), float("nan"), tot["actual"] * FX / 1e3, tot["plan_q1"] * FX / 1e3,
                          tot["plan_q2"] * FX / 1e3,
                          tot["six_months"] * FX / 1e3, (tot["six_months"] - rtot["six_months"]) * FX / 1e3]
    # growth rates as text, so the Total row shows blank rather than Streamlit's "None"
    rates = ["Customers %/month", "AOV %/month"]
    table[rates] = table[rates].apply(lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"))
    st.dataframe(table.style.format("{:,.1f}", subset=[c for c in table.columns if c not in rates]),
                 use_container_width=True)
    out = proj.assign(revenue_eur=proj["revenue"] / cfg.BRL_PER_EUR).rename(columns={"revenue": "revenue_brl"})
    st.download_button(f"Download {FC_LABEL} projection (CSV, R$ and €)", out.round(2).to_csv(index=False),
                       "scenario_projection.csv", "text/csv")

# ------------------------------------------------------------------ where to invest
with tab_where:
    st.subheader("Markets")
    st.caption(f"Extra {FC_LABEL} revenue from +1 point of monthly customer growth shows where the same "
               f"lift is worth most. Momentum columns compare {cfg.LAST6_LABEL}.")
    ms = mkt_sc.set_index("market")
    fig2 = go.Figure(go.Bar(
        x=ms["extra_6m_revenue_per_1pt_customers"] * FX, y=ms.index, orientation="h",
        marker_color=[MKT_COLOR[m] for m in ms.index],
        hovertemplate=f"%{{y}}<br>+{SYM} %{{x:,.0f}} in {FC_LABEL}<extra></extra>"))
    fig2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                       title=f"Extra {FC_LABEL} revenue per +1 pt monthly customer growth",
                       xaxis=cur_axis(), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)
    pct_cols = ["revenue_share", "revenue_growth_yoy", "revenue_growth_last6m", "customers_growth_last6m",
                "aov_growth_last6m", "customers_growth_last3m", "share_of_recent_growth"]
    money_cols = ["baseline_revenue_month", "aov", "extra_6m_revenue_per_1pt_customers"]
    show = ms.copy()
    show[pct_cols] = show[pct_cols] * 100
    show[money_cols] = show[money_cols] * FX
    show = show.rename(columns={
        "revenue_share": "Share %", "baseline_revenue_month": f"Revenue/month {SYM}", "aov": f"AOV {SYM}",
        "revenue_growth_yoy": "Revenue YoY %", "revenue_growth_last6m": "Revenue last 6m %",
        "customers_growth_last6m": "Customers last 6m %", "aov_growth_last6m": "AOV last 6m %",
        "customers_growth_last3m": f"Customers {cfg.BASELINE_LABEL} vs {cfg.PRIOR3_LABEL} %",
        "share_of_recent_growth": "Share of recent growth %",
        "extra_6m_revenue_per_1pt_customers": f"{SYM} per +1 pt customers"})
    st.dataframe(show.style.format("{:,.1f}"), use_container_width=True)
    st.subheader("Categories")
    st.caption(f"YoY = {cfg.YOY_LABEL}. Last 6m = {cfg.LAST6_LABEL}.")
    cs = cat_sc.set_index("category")
    pc = ["revenue_share_last6m", "revenue_growth_yoy", "revenue_growth_last6m", "aov_growth_last6m"]
    cs[pc] = cs[pc] * 100
    cs[["revenue_last6m", "aov_last6m"]] = cs[["revenue_last6m", "aov_last6m"]] * FX
    cs = cs.rename(columns={
        "revenue_share_last6m": "Share last 6m %", "revenue_last6m": f"Revenue last 6m {SYM}",
        "aov_last6m": f"AOV {SYM}", "revenue_growth_yoy": "Revenue YoY %",
        "revenue_growth_last6m": "Revenue last 6m %", "aov_growth_last6m": "AOV last 6m %"})
    st.dataframe(cs.style.format("{:,.1f}"), use_container_width=True)

# ------------------------------------------------------------------ method & data
with tab_check:
    st.subheader("Would this forecast have worked?")
    chk = check.set_index("model")
    picked = fsum["check"]["selected_by_backtest"]
    bt_avg = scores.groupby("model")["wape"].mean()
    june = cfg.month_label(cfg.CHECK_END)
    md(f"""
{N_MODELS} forecasting models were compared. Each model only ever sees data up to the month it forecasts from,
so nothing is learned from the answer. This tab answers **two different questions**:

1. **Which model makes the plan?** Backtest on all the data (up to {cfg.month_label(cfg.MODEL_END)}): each model
   forecasts {cfg.FORECAST_H} months ahead from 4 starting points, error averaged over the total and the 5 markets.
   The winner, the **3-month average**, is the model behind the {PLAN} plan.
2. **Would that way of choosing have worked in the past?** Go back to the end of **{june}**, apply the same rule
   with only the data known then, and compare the chosen model's forecast with what really happened in
   **{cfg.CHECK_LABEL}**. This tests the *method*; the model it picks is **not** used for the plan.
""")
    c_now = f"Backtest to {cfg.month_label(cfg.MODEL_END)} %"
    c_then = f"Backtest to {june} %"
    c_real = f"Error on real {cfg.span_label(str(pd.Period(cfg.CHECK_END, 'M') + 1), cfg.MODEL_END)} %"
    short = {"WindowAverage": "3-month average", "Naive": "Last month repeated", "RWD": "Random walk + drift",
             "AutoETS": "AutoETS", "AutoTheta": "AutoTheta", "Drivers": "Driver-based",
             "Prophet": "Prophet (daily)", "LagRegression": "Customer-momentum regression",
             "ProphetKPI": "Prophet + forecast KPIs", "ProphetLead": "Prophet + leading indicators"}
    tbl = pd.DataFrame({
        "Model": [short[m] for m in chk.index],
        "Role": ["✓ Plan model" if m == fsum["selected_model"] else
                 ("June test pick" if m == picked else "") for m in chk.index],
        c_now: [bt_avg[m] * 100 for m in chk.index],
        c_then: chk["backtest_avg_wape"] * 100,
        c_real: chk["check_avg_wape"] * 100,
        "Direction": ["too high" if v > 0.005 else ("too low" if v < -0.005 else "on target")
                      for v in chk["check_total_error"]],
    }, index=chk.index).sort_values(c_now)
    st.dataframe(tbl.set_index("Model").style.format({c_now: "{:.1f}", c_then: "{:.1f}", c_real: "{:.1f}"}),
                 use_container_width=True)
    md(f"**✓ Plan model** = the model used for the {PLAN} plan. **June test pick** = the model the same rule would have "
       f"chosen in {june}: shown to test the rule, not used.", caption=True)
    md(f"All errors are **absolute** (WAPE): the average size of the gap between forecast and actual, whether too high "
       f"or too low, over the total and the 5 markets. Lower is better. Columns: backtest on all the data (question 1: "
       f"this ranks the models for the plan), backtest with data to {june} (what the rule saw in June), and the error on "
       f"the real {cfg.CHECK_LABEL} (question 2). *Direction* says whether the total was forecast too high or too low. "
       f"The real {cfg.CHECK_LABEL} months are a test only: choosing the plan model by them would be hindsight.",
       caption=True)

    # forecasts made in June vs what happened
    h18 = hist[hist["month"] >= "2018-01"]
    fpc = go.Figure()
    fpc.add_scatter(x=h18["month"], y=h18["revenue"] * FX, name="Actual", mode="lines+markers",
                    line=dict(color="#0b0b0b", width=2.5), marker=dict(size=6), hovertemplate=HOVER + "<extra>Actual</extra>")
    style = {picked: (ORANGE, 2.5, f"{short[picked]} (test only: June's pick)"),
             "WindowAverage": (BLUE, 2.5, "3-month average (used for the plan)"),
             "Prophet": ("#1baf7a", 2, "Prophet (daily)"),
             "Naive": ("#e87ba4", 2, "Last month repeated")}
    first_other = True
    for m in [c for c in check_paths.columns if c not in ("month", "actual")]:
        x = [cfg.CHECK_END] + list(check_paths["month"])
        y = [hist.set_index("month").loc[cfg.CHECK_END, "revenue"]] + list(check_paths[m])
        if m in style:
            col, wd, name = style[m]
            fpc.add_scatter(x=x, y=[v * FX for v in y], name=name, mode="lines", line=dict(color=col, width=wd),
                            hovertemplate=HOVER + f"<extra>{MODEL_NAMES[m]}</extra>")
        else:  # the other models, one grey legend entry
            fpc.add_scatter(x=x, y=[v * FX for v in y], name="Other models", mode="lines", legendgroup="other",
                            showlegend=first_other, line=dict(color="#c3c2b7", width=1.2, dash="dot"),
                            hovertemplate=HOVER + f"<extra>{MODEL_NAMES[m]}</extra>")
            first_other = False
    fpc.add_vrect(x0=cfg.CHECK_END, x1=check_paths["month"].iloc[-1], fillcolor="#eef4fb", line_width=0,
                  layer="below")
    fpc.add_annotation(x=check_paths["month"].iloc[1], y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"forecast from {june}", font=dict(size=11, color=INK2))
    months_axis = list(h18["month"]) + [m for m in check_paths["month"] if m not in set(h18["month"])]
    fpc.update_layout(height=440, margin=dict(l=10, r=10, t=70, b=10), plot_bgcolor="white",
                      title=dict(text=f"Forecasts made at the end of {june} vs what happened", y=0.97, x=0,
                                 xanchor="left"),
                      legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=11)),
                      yaxis=cur_axis(rangemode="tozero"),
                      xaxis=dict(type="category", tickvals=months_axis[::2], tickangle=0,
                                 ticktext=[pd.Period(m, "M").strftime("%b %y") for m in months_axis[::2]]),
                      hovermode="x unified")
    st.plotly_chart(fpc, use_container_width=True)

    # Prophet's error from each starting point of the main backtest: close in June, unstable before
    bp = pd.read_csv(P / "backtest_predictions.csv")
    pt = bp[bp["unique_id"] == "Total"]
    per_cut = {c: (g["Prophet"] - g["y"]).abs().sum() / g["y"].sum() for c, g in pt.groupby("cutoff")}
    span_txt = ", ".join(f"{pd.Period(c, 'M').strftime('%b %Y')} {v:.0%}" for c, v in per_cut.items())

    md(f"""
**What we learned**
- **In {june} the rule would have chosen {MODEL_NAMES[picked]}** (test only, not used), and it forecast
  {cfg.CHECK_LABEL} **{chk.loc[picked, 'check_total_error']:+.0%} too high**. Revenue had grown for 18 months, then
  stopped in July: a turning point that no model built on past revenue can see coming.
- **The flat 3-month average** (the model behind this plan) was {chk.loc['WindowAverage', 'check_total_error']:+.0%} out,
  and it wins the backtest on all the data ({bt_avg['WindowAverage']:.1%} error).
- **Prophet (daily) was close on {cfg.CHECK_LABEL}** ({chk.loc['Prophet', 'check_total_error']:+.1%} on the total), but
  it could not have been trusted: from the backtest's starting points its total error was {span_txt}; across the 5
  markets it was {chk.loc['Prophet', 'check_avg_wape']:.0%} off on {cfg.CHECK_LABEL}, worse than repeating last month
  ({chk.loc['Naive', 'check_avg_wape']:.0%}); and two months is a small test (last month repeated was even closer,
  {chk.loc['Naive', 'check_total_error']:+.1%}, largely by luck). Choosing it now, after seeing the answer, would be
  hindsight. It is promising: it learns Black Friday by itself and steadies with more history, so it should be tested
  again once 2+ years of data exist.
- **Richer models did not help:** the driver-based model ({bt_avg['Drivers']:.1%}), a regression on customer
  momentum ({bt_avg['LagRegression']:.1%}), **Prophet with forecast KPIs** as regressors ({bt_avg['ProphetKPI']:.1%}: it
  mostly rebuilds customers x orders x AOV, so it is only as good as those KPI forecasts) and **Prophet with leading
  indicators** lagged 2 months ({bt_avg['ProphetLead']:.1%}).
- **Leading indicators were the closest call:** late deliveries jumped to 14-19% in Feb-Mar 2018, and as of June the
  leading-indicator model had the best backtest of all. It still missed the July turning point.
- **The warning sign was in the KPIs:** customer numbers were already falling in 4 of 5 markets by {june}. That is
  why the plan comes with customer-based triggers (see the decision memo).
""")


# ------------------------------------------------------------------ history
@st.cache_data
def history_tables():
    """Monthly history for every metric, per market and the Total, over the modelling window."""
    tot = pd.read_csv(P / "monthly_total.csv").assign(market="Total")
    mk = pd.read_csv(P / "monthly_market.csv")
    h = pd.concat([mk, tot], ignore_index=True)
    h = h[h["in_model_window"]].copy()
    h["items_per_order"] = h["items"] / h["orders"]
    h["price_per_item"] = h["revenue"] / h["items"]
    ind = pd.read_csv(P / "monthly_indicators.csv")
    h = h.merge(ind, on=["month", "market"], how="left")
    cat = pd.read_csv(P / "monthly_category.csv")
    cat = cat[cat["in_model_window"]]
    return h, cat


# metric -> (label, kind): kind "money" follows the currency switch, "pct" is shown as %, "num" as is
METRICS = {
    "revenue": ("Revenue (item price, delivered orders)", "money"),
    "gmv": ("GMV (revenue + freight)", "money"),
    "orders": ("Orders", "num"),
    "customers": ("Customers (unique buyers)", "num"),
    "orders_per_customer": ("Orders per customer", "num"),
    "aov": ("Average order value (AOV)", "money"),
    "items_per_order": ("Items per order (basket size)", "num"),
    "price_per_item": ("Price per item", "money"),
    "late_share": ("Late deliveries (share of delivered orders)", "pct"),
    "cancel_rate": ("Cancellations (share of all orders)", "pct"),
    "new_share": ("New customers (share of buyers)", "pct"),
}

with tab_hist:
    st.subheader("History: the data behind every graph")
    md(f"Monthly history for {HIST_LABEL}, by market. Pick a metric and the markets to compare; the chart, the "
       "table and the download all follow your choice. Money amounts follow the R$ / € switch.")
    hdat, cdat = history_tables()
    h1, h2 = st.columns([2, 3])
    metric = h1.selectbox("Metric", list(METRICS), format_func=lambda k: METRICS[k][0], key="hist_metric")
    hm = h2.multiselect("Markets", ["Total"] + cfg.MARKETS, default=["Total"], key="hist_markets")
    label, kind = METRICS[metric]
    if not hm:
        st.warning("Pick at least one market.")
    else:
        piv = hdat.pivot(index="month", columns="market", values=metric)[hm]
        scale = FX if kind == "money" else (100 if kind == "pct" else 1)
        shown = piv * scale
        fh = go.Figure()
        for mk_ in hm:
            fh.add_scatter(x=shown.index, y=shown[mk_], name=mk_, mode="lines+markers",
                           line=dict(color="#0b0b0b" if mk_ == "Total" else MKT_COLOR[mk_], width=2.2),
                           marker=dict(size=5),
                           hovertemplate=(f"%{{x}}<br>{mk_}: " + (f"{SYM} %{{y:,.0f}}" if kind == "money" else
                                                                 "%{y:.1f}%" if kind == "pct" else "%{y:,.3f}")
                                          + "<extra></extra>"))
        fh.add_vline(x="2017-11", line=dict(color=LIGHT, dash="dot", width=1))
        fh.add_annotation(x="2017-11", y=1, yref="paper", text="Black Friday", showarrow=False,
                          font=dict(size=10, color=MUTED), yanchor="bottom")
        yax = cur_axis(rangemode="tozero") if kind == "money" else dict(
            gridcolor=GRID, ticksuffix="%" if kind == "pct" else "", rangemode="tozero")
        fh.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="white",
                         title=dict(text=f"{label}, monthly", y=0.97, x=0, xanchor="left"),
                         legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5),
                         yaxis=yax, xaxis=dict(type="category", tickvals=list(shown.index)[::3]),
                         hovermode="x unified")
        st.plotly_chart(fh, use_container_width=True)
        unit = f" ({SYM})" if kind == "money" else (" (%)" if kind == "pct" else "")
        fmt = "{:,.0f}" if kind == "money" or metric in ("orders", "customers") else (
            "{:.1f}" if kind == "pct" else "{:.3f}")
        st.dataframe(shown.rename(columns=lambda c: c + unit).style.format(fmt), use_container_width=True, height=300)
        st.download_button(f"Download this table (CSV)", shown.round(4).to_csv(), f"history_{metric}.csv", "text/csv",
                           key="dl_hist")

    st.markdown("**Categories**")
    top = (cdat.groupby("category")["revenue"].sum().sort_values(ascending=False).index.tolist())
    hc = st.multiselect("Categories", top, default=top[1:4], key="hist_categories",
                        help=f"Top {cfg.TOP_N_CATEGORIES} categories by revenue; everything else is 'Other'.")
    if hc:
        cp = cdat.pivot(index="month", columns="category", values="revenue")[hc] * FX
        fc_ = go.Figure()
        for i, c in enumerate(hc):
            fc_.add_scatter(x=cp.index, y=cp[c], name=c.replace("_", " "), mode="lines",
                            line=dict(width=2, color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                      "#008300", "#6250d6", "#e34948"][i % 8]),
                            hovertemplate=f"%{{x}}<br>{c}: {SYM} %{{y:,.0f}}<extra></extra>")
        fc_.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white",
                          title=dict(text="Monthly revenue by category", y=0.97, x=0, xanchor="left"),
                          legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.4),
                          yaxis=cur_axis(rangemode="tozero"),
                          xaxis=dict(type="category", tickvals=list(cp.index)[::3]), hovermode="x unified")
        st.plotly_chart(fc_, use_container_width=True)
        st.download_button("Download category history (CSV)", cp.round(2).to_csv(), "history_categories.csv",
                           "text/csv", key="dl_cat")

# ------------------------------------------------------------------ data
DATA_TABLES = {
    "monthly_total.csv": "Monthly revenue, GMV, orders, customers, items and AOV for the whole marketplace",
    "monthly_market.csv": "The same, per market (SP, RJ, MG, South, Rest of Brazil)",
    "monthly_category.csv": f"The same, per product category (top {cfg.TOP_N_CATEGORIES} + Other)",
    "monthly_market_category.csv": "The same, per market and category",
    "monthly_indicators.csv": "Late deliveries, cancellations and new-customer share, per market and month",
    "daily_market.csv": "Daily revenue per market (used by the daily Prophet model)",
    "state_summary.csv": "Revenue, orders, customers and growth per state (used by the map)",
    "order_status_by_month.csv": "Orders by status (delivered, canceled, shipped...) per month",
    "order_outliers.csv": "Very large orders (top 0.1%), checked and kept",
    "driver_tree_market.csv": "Customers, orders per customer, AOV, basket size and price per item, per market",
    "driver_decomposition.csv": "How much each driver contributed to revenue growth (LMDI split)",
    "driver_baseline.csv": "Starting values for the scenarios (last 3 months of data)",
    "backtest_scores.csv": "Error of every forecasting model, per series",
    "forecast.csv": "The forecast for the plan months, with 80% range and Black Friday",
    "forecast_check.csv": "The June 2018 real-world check of every model",
    "scenario_summary.csv": "Plan totals per scenario, market and quarter",
    "market_scorecard.csv": "Size, momentum and order value per market",
    "category_scorecard.csv": "Size and momentum per category",
}

with tab_data:
    st.subheader("The data")
    md(f"""
**Source:** the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0): real, anonymised orders from
Olist, a Brazilian marketplace where many independent sellers list products. It covers **Sep 2016 to Oct 2018**;
this project uses **{HIST_LABEL}** ({cfg.HISTORY_MONTHS} months, the months with stable volume).
""")
    st.markdown("**Raw files**")
    raw = pd.DataFrame([
        ("Orders", "olist_orders_dataset.csv", dq["raw_rows"]["orders"],
         "One row per order: status, purchase / delivery / promised dates", "Revenue month, delivered filter, late deliveries, cancellations"),
        ("Order items", "olist_order_items_dataset.csv", dq["raw_rows"]["items"],
         "One row per item: product, seller, price, freight", "Revenue (price), GMV (price + freight), basket size"),
        ("Customers", "olist_customers_dataset.csv", dq["raw_rows"]["customers"],
         "Customer id, unique person id, city, state", "Markets and the map; unique customers"),
        ("Products", "olist_products_dataset.csv", dq["raw_rows"]["products"],
         "Product category (Portuguese), size, weight", "Categories"),
        ("Category names", "product_category_name_translation.csv", dq["raw_rows"]["translation"],
         "Portuguese → English category names", "Readable categories"),
        ("Payments", "olist_order_payments_dataset.csv", dq["raw_rows"]["payments"],
         "Payment type, instalments, amount", "Only a cross-check of revenue"),
    ], columns=["Table", "File", "Rows", "Contains", "Used for"]).set_index("Table")
    st.dataframe(raw.style.format({"Rows": "{:,}"}), use_container_width=True)
    st.caption("Not used: geolocation (zip-code coordinates), sellers and customer reviews. They are part of the "
               "dataset but not needed for revenue planning; reviews could be added later as a leading indicator.")

    st.markdown("**How the data was cleaned**")
    status = pd.Series(dq["order_status_counts"])
    md(f"""
- **Only delivered orders count as revenue.** {status['delivered']:,} delivered; {dq['orders_excluded_not_delivered']:,}
  others excluded ({', '.join(f'{k} {v:,}' for k, v in status.drop('delivered').items())}).
- **Revenue = item price**; freight is a delivery charge, so it is reported separately as GMV.
- **No duplicates:** {dq['duplicate_order_ids']} duplicate orders, {dq['duplicate_item_keys']} duplicate items.
- **Missing categories:** {dq['products_missing_category']} products have none; they are grouped as "unknown" inside "Other".
- **Very large orders** (top 0.1%) were checked: genuine big baskets, so they are kept.
- **Sparse months dropped:** 2016 ({dq['delivered_orders_before_model_window']} delivered orders) and Sep-Oct 2018 (none delivered yet).
- **A customer** is Olist's `customer_unique_id`: a person buying twice counts once.
- **Every step is tested:** e.g. revenue in the monthly tables adds back exactly to the raw items.
""")

    st.markdown("**Tables the app uses** (built by `python -m src.run_all`)")
    pick = st.selectbox("Preview a table", list(DATA_TABLES), key="data_table",
                        format_func=lambda f: f"{f}: {DATA_TABLES[f]}")
    tdf = pd.read_csv(P / pick)
    st.caption(f"{len(tdf):,} rows × {tdf.shape[1]} columns. Amounts in R$.")
    st.dataframe(tdf.head(200), use_container_width=True, height=280)
    st.download_button(f"Download {pick}", (P / pick).read_bytes(), pick, "text/csv", key="dl_data")

    st.markdown("**Glossary**")
    gl = pd.DataFrame([
        ("Revenue", "Sum of item prices on delivered orders, by purchase month (freight excluded)."),
        ("GMV", "Gross merchandise value: revenue + freight."),
        ("Customers", "Unique buyers in the month (customer_unique_id)."),
        ("Orders per customer", "Orders ÷ customers in the month (about 1.007: almost everyone orders once)."),
        ("AOV", "Average order value: revenue ÷ orders. AOV = items per order × price per item."),
        ("Items per order", "Basket size: items ÷ orders."),
        ("Price per item", "Revenue ÷ items."),
        ("Market", "Customer location: SP, RJ, MG on their own, South (PR, SC, RS), Rest of Brazil (21 states)."),
        ("Late delivery", "Delivered after the date promised at purchase."),
        ("New customer", "A buyer whose first ever delivered order is in that month."),
        ("H2 2018 / Q3 / Q4", f"Jul–Dec 2018; Q3 = Jul–Sep, Q4 = Oct–Dec. {ACT_LABEL} are actuals, {FC_LABEL} forecast."),
        ("Backtest", "Pretend to be at an earlier date, forecast, and compare with what really happened."),
        ("WAPE", "Weighted absolute percentage error: total |forecast − actual| ÷ total actual."),
        ("LMDI", "A method that splits revenue growth exactly between its drivers."),
        ("Elasticity", "% of customers lost for each 1% price rise (an assumption in the What-if tab)."),
        ("Black Friday uplift", f"November vs the average of October and December 2017: ×{uplift:.2f}."),
        ("Constant currency", f"Euro amounts use one rate (R$ {cfg.BRL_PER_EUR} = €1) so growth is not distorted."),
    ], columns=["Term", "Meaning"]).set_index("Term")
    st.dataframe(gl, use_container_width=True, height=420)

with tab_about:
    wape = scores.groupby("model")["wape"].mean().sort_values()
    names = MODEL_NAMES
    md(f"""
#### The data
- **Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
  (CC BY-NC-SA 4.0): orders, order items, customers and products from a Brazilian marketplace.
- **Revenue** = item price on **delivered** orders, by purchase month. Freight is excluded (a delivery charge,
  not product revenue); {dq['orders_excluded_not_delivered']:,} non-delivered orders (3%) are excluded.
- **Customer** = Olist's `customer_unique_id`, so a person ordering twice counts once.
- **Window:** {HIST_LABEL} ({dq['model_window_months']} months, {dq['model_window_orders']:,} orders,
  {mm(dq['model_window_revenue'])}). 2016 and Sep-Oct 2018 are too sparse to use. The plan is {PLAN}:
  {ACT_LABEL} actuals + a {FC_LABEL} forecast.
- **Currency:** amounts are in Brazilian reais. The € view divides by {cfg.BRL_PER_EUR} ({cfg.FX_SOURCE}),
  one constant rate so growth rates are not affected by exchange-rate moves.

#### How the numbers are built
1. **Clean and check** (`src/prepare.py`): join the tables, keep delivered orders, test for duplicates and
   reconcile revenue back to the raw items.
2. **Driver tree** (`src/drivers.py`): revenue = customers x orders per customer x items per order x price per
   item, by market and month. Growth is split exactly between the drivers (LMDI method).
3. **Forecast** (`src/forecast.py`): {N_MODELS} models backtested from 4 starting points, {cfg.FORECAST_H} months ahead each:
   five that look at revenue alone, a driver-based model, a regression on customer momentum, Prophet on daily data,
   Prophet with forecast KPIs and Prophet with lagged leading indicators. See the **Forecast check** tab.
   Average error (WAPE) across the total and 5 markets: """ + ", ".join(
        f"{names[m]} {v:.1%}" for m, v in wape.items()) + f""". The **3-month average** wins, so the
   baseline is flat; trend models over-forecast because they carry 2017's growth into 2018.
4. **Scenarios** (`src/scenarios.py`): from the {cfg.BASELINE_LABEL} starting point, customers and AOV grow at monthly
   rates measured in the data. *Downside*: {scenarios.DESCRIPTIONS['Downside'].lower()} ({cfg.BASELINE_LABEL} vs {cfg.PRIOR3_LABEL}
   2018). *Base*: flat, equal to the forecast. *Upside*: {scenarios.DESCRIPTIONS['Upside'].lower()}
   ({cfg.LAST6_LABEL}). Black Friday adds x{uplift:.2f} in November (one observed November).
5. **What-if:** a one-off step in any driver from a chosen month. *Price vs volume* lets customers respond with a
   constant elasticity: customers x (1 + price change)^(-elasticity), so revenue changes by
   (1 + price change)^(1 - elasticity) - 1 and breaks even at an elasticity of 1.
6. **Decision memo** (`src/memo.py`): generated from the outputs, with checks that each claim still holds.

Every step is covered by automated tests: the maths (drivers multiply back to revenue, markets add up to the
total), the backtest (no look-ahead), the app and the memo. The app and the notebooks use the same engine, so
the numbers always match.

#### Limits
{cfg.HISTORY_MONTHS} months of history and a single November, so yearly seasonality cannot be learned reliably
(Prophet, which tries, was unstable). No cost, margin or marketing data:
the planner shows what a change is worth in revenue, not what it costs. Markets are customer locations, not
sales teams. Price elasticity is an assumption; the data has no price tests.

[Decision memo]({REPO_URL}/blob/main/docs/decision_memo.md) · [Code and notebooks on GitHub]({REPO_URL})
""")
