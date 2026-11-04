"""Stage 5: decision memo, generated from the pipeline outputs.

Every number in docs/decision_memo.md comes from data/processed/. The wording of the
recommendations is fixed here; if the data changed enough to contradict a recommendation,
`check_claims` fails (and so does the test suite), so the memo can't quietly go stale.

Run: python -m src.memo   (needs src.run_all outputs)
"""
from __future__ import annotations

import json

import pandas as pd

from src import config as cfg

P = cfg.PROCESSED_DIR
NAMES = {"WindowAverage": "the 3-month average", "Naive": "last month repeated", "RWD": "random walk with drift",
         "AutoETS": "AutoETS", "AutoTheta": "AutoTheta", "Drivers": "the driver-based model",
         "Prophet": "Prophet (daily, seasonal)", "LagRegression": "the customer-momentum regression",
         "ProphetKPI": "Prophet with forecast KPIs",
         "ProphetLead": "Prophet with leading indicators (late deliveries, cancellations, new customers)"}
OUT = cfg.ROOT / "docs" / "decision_memo.md"


def _m(v: float) -> str:
    return f"R$ {v/1e6:.2f}M"


def _k(v: float) -> str:
    return f"R$ {v/1e3:,.0f}k"


def _p(v: float, sign: bool = True) -> str:
    return f"{v:+.0%}" if sign else f"{v:.0%}"


def load() -> dict:
    proj = pd.read_csv(P / "scenario_projection.csv")
    summ = pd.read_csv(P / "scenario_summary.csv")
    hist = pd.read_csv(P / "monthly_total.csv")
    dec = pd.read_csv(P / "driver_decomposition.csv")
    return {
        "dq": json.loads((P / "data_quality.json").read_text()),
        "fs": json.loads((P / "forecast_summary.json").read_text()),
        "tot": summ[summ["market"] == "Total"].set_index("scenario"),
        "mkt6": summ[summ["market"] != "Total"].pivot(index="market", columns="scenario", values="six_months"),
        "msc": pd.read_csv(P / "market_scorecard.csv").set_index("market"),
        "csc": pd.read_csv(P / "category_scorecard.csv").set_index("category"),
        "base": pd.read_csv(P / "driver_baseline.csv").set_index("market"),
        "yoy": dec[dec["comparison"] == cfg.COMPARISONS[0][0]].set_index("market"),
        "l6": dec[dec["comparison"] == cfg.COMPARISONS[1][0]].set_index("market"),
        "cust": proj.groupby(["scenario", "month"])["customers"].sum().unstack(0),
        "prev_half": hist[hist["month"].between(*cfg.PREV_HALF)]["revenue"].sum(),
        "last_year_half": hist[hist["month"].between(*cfg.LAST_YEAR_HALF)]["revenue"].sum(),
        "scores": pd.read_csv(P / "backtest_scores.csv"),
        "check": pd.read_csv(P / "forecast_check.csv").set_index("model"),
    }


def check_claims(d: dict) -> None:
    """Fail loudly if the data no longer supports the memo's fixed wording."""
    msc, csc, tot = d["msc"], d["csc"], d["tot"]["six_months"]
    cats = csc.drop(index=cfg.OTHER_CATEGORY)
    avg = d["scores"].groupby("model")["wape"].mean()
    claims = {
        "scenarios are ordered": tot["Downside"] < tot["Base"] < tot["Upside"],
        "SP is the largest customer lever": msc["extra_6m_revenue_per_1pt_customers"].idxmax() == "SP",
        "SP led recent growth": msc["share_of_recent_growth"].idxmax() == "SP",
        "South and Rest of Brazil grew AOV most": set(msc["aov_growth_last6m"].nlargest(2).index) == {"South", "Rest of Brazil"},
        "Rest of Brazil has the highest AOV": msc["aov"].idxmax() == "Rest of Brazil",
        "RJ shrank over the last 6 months": msc.loc["RJ", "revenue_growth_last6m"] < 0,
        "customers drove YoY growth": d["yoy"].loc["Total", "contrib_customers"] / d["yoy"].loc["Total", "revenue_change"] > 0.9,
        "orders per customer is flat": abs(d["l6"].loc["Total", "orders_per_customer_growth"]) < 0.02,
        "monthly customers fell in the last quarter, in every market": (d["base"]["customers_growth_vs_prior"] < 0).all(),
        "AOV drove most of South and Rest of Brazil growth": all(
            d["l6"].loc[m, "contrib_aov"] > 0.5 * d["l6"].loc[m, "revenue_change"] for m in ("South", "Rest of Brazil")),
        "SP has by far the strongest customer growth": (
            msc.loc["SP", "customers_growth_last6m"] > 2 * msc["customers_growth_last6m"].drop("SP").max()),
        "health_beauty is a top-2 category and growing": (
            "health_beauty" in cats["revenue_last6m"].nlargest(2).index and cats.loc["health_beauty", "revenue_growth_last6m"] > 0),
        "housewares grew fastest": cats["revenue_growth_last6m"].idxmax() == "housewares",
        "cool_stuff and computers_accessories declined": (cats.loc[["cool_stuff", "computers_accessories"], "revenue_growth_last6m"] < 0).all(),
        "flat model selected": d["fs"]["selected_model"] == "WindowAverage",
        "no richer model beat the flat one in the backtest": (
            avg["WindowAverage"] <= avg[["Drivers", "Prophet", "LagRegression", "ProphetKPI", "ProphetLead"]].min()),
        "the June pick was a richer model, not the flat one": d["fs"]["check"]["selected_by_backtest"] != "WindowAverage",
        "the model picked as of June over-forecast the check months": d["fs"]["check"]["selected_check_total_error"] > 0.10,
        "Prophet was unstable in the backtest": d["scores"].groupby("model")["wape"].mean()["Prophet"] > 0.5,
    }
    failed = [k for k, ok in claims.items() if not ok]
    if failed:
        raise AssertionError(f"Memo claims no longer supported by the data: {failed}")


def triggers(d: dict) -> dict:
    """Average monthly customers in the first two forecast months that mark the switch to another scenario."""
    c = d["cust"].iloc[:2].mean()
    return {"base": c["Base"], "down": (c["Base"] + c["Downside"]) / 2, "up": (c["Base"] + c["Upside"]) / 2}


def render(d: dict) -> str:
    check_claims(d)
    dq, fs, tot, msc, csc, mkt6 = d["dq"], d["fs"], d["tot"], d["msc"], d["csc"], d["mkt6"]
    base, yoy, l6, chk = d["base"], d["yoy"], d["l6"], d["check"]
    b, dn, up = tot.loc["Base"], tot.loc["Downside"], tot.loc["Upside"]
    tr = triggers(d)
    sp_lever, rj_lever = msc.loc["SP", "extra_6m_revenue_per_1pt_customers"], msc.loc["RJ", "extra_6m_revenue_per_1pt_customers"]
    cats = csc.drop(index=cfg.OTHER_CATEGORY)
    wa = d["scores"].groupby("model")["wape"].mean()
    tb = d["scores"][d["scores"]["unique_id"] == "Total"].set_index("model")["bias"]
    trend_bias = tb[["RWD", "AutoETS", "AutoTheta"]]
    bf_rev = b["forecast"] - fs["forecast_total"]
    trig_label = cfg.span_label(d["cust"].index[0], d["cust"].index[1])
    picked = fs["check"]["selected_by_backtest"]
    q1, q2, h = cfg.PLAN_Q1_LABEL, cfg.PLAN_Q2_LABEL, cfg.PLAN_NAME
    prev, ly = cfg.PREV_HALF_NAME, cfg.LAST_YEAR_HALF_NAME

    def cat(k):
        return f"{k.replace('_', ' ')} ({_p(cats.loc[k, 'revenue_growth_last6m'])})"

    def vs(x):
        return f"{x / d['prev_half'] - 1:+.1%} vs {prev}, {_p(x / d['last_year_half'] - 1)} vs {ly}"

    return f"""# Decision memo: revenue plan for {h} ({cfg.PLAN_LABEL})

*Demonstration project using public data (Olist). Not affiliated with any employer.
Generated by `python -m src.memo` from the pipeline outputs; every figure below comes from `data/processed/`.*

**Question:** what revenue should we plan for {h}, and which markets or categories should get extra growth
investment?

**How the plan is built:** {h} = actual revenue for {cfg.ACTUAL_LABEL} (already in the data) + a forecast for
{cfg.FORECAST_LABEL}. The forecast learns from {cfg.HISTORY_LABEL} ({cfg.HISTORY_MONTHS} months).

## Recommendation

1. **Plan on {_m(b['six_months'])} for {h}**: {_m(b['plan_q1'])} in {q1} and {_m(b['plan_q2'])} in {q2}.
   {_m(b['actual'])} of it is already booked ({cfg.ACTUAL_LABEL}); the remaining {cfg.FORECAST_LABEL} is flat at the
   {cfg.BASELINE_LABEL} run rate ({_k(base.loc['Total', 'revenue'])} a month) plus a Black Friday uplift in November.
   That is {vs(b['six_months'])}.
2. **Hold a downside case of {_m(dn['six_months'])}** ({_m(b['six_months'] - dn['six_months'])} below plan). It
   assumes only that the customer decline of the last quarter carries on. It is not a stress test.
3. **Put extra customer-acquisition effort into SP first.** As the largest market it is the largest lever: one
   extra point of monthly customer growth there is worth {_k(sp_lever)} over {cfg.FORECAST_LABEL}, {sp_lever/rj_lever:.1f}× the
   same lift in RJ. It also has the momentum: SP delivered {_p(msc.loc['SP', 'share_of_recent_growth'], False)} of the last 6 months' growth.
4. **In South and Rest of Brazil, protect order value.** Their recent growth came mostly from AOV
   (South {_p(msc.loc['South', 'aov_growth_last6m'])}, Rest of Brazil {_p(msc.loc['Rest of Brazil', 'aov_growth_last6m'])});
   Rest of Brazil has the highest AOV at R$ {msc.loc['Rest of Brazil', 'aov']:.0f}.
5. **Don't add spend in RJ until the decline is understood** (revenue {_p(msc.loc['RJ', 'revenue_growth_last6m'])}
   over the last 6 months). MG is flat ({_p(msc.loc['MG', 'revenue_growth_last6m'])}).
6. **Categories: back {cat('health_beauty')}, {cat('housewares')} and {cat('auto')}**; review
   {cat('computers_accessories')} and {cat('cool_stuff')}.

## The plan in numbers

| Scenario | Assumes | {q1} | {q2} | {h} | vs {prev} |
|---|---|---|---|---|---|
| Downside | Last quarter's customer decline continues | {_m(dn['plan_q1'])} | {_m(dn['plan_q2'])} | {_m(dn['six_months'])} | {dn['six_months']/d['prev_half']-1:+.1%} |
| **Base (plan)** | Drivers stay at {cfg.BASELINE_LABEL} levels | **{_m(b['plan_q1'])}** | **{_m(b['plan_q2'])}** | **{_m(b['six_months'])}** | {b['six_months']/d['prev_half']-1:+.1%} |
| Upside | Last-6-month momentum continues | {_m(up['plan_q1'])} | {_m(up['plan_q2'])} | {_m(up['six_months'])} | {up['six_months']/d['prev_half']-1:+.1%} |

All three include the {cfg.ACTUAL_LABEL} actuals ({_m(b['actual'])}) and the Black Friday uplift (×{fs['black_friday_uplift']:.2f}
on November, worth {_m(bf_rev)} in the base case). Without the uplift the base is {_m(b['six_months'] - bf_rev)}.
For reference, {prev} was {_m(d['prev_half'])} and {ly} {_m(d['last_year_half'])}.

| Market | Share of revenue | Base, {h} | Downside | Upside | Revenue, last 6m vs prior 6m |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {m} | {_p(msc.loc[m, 'revenue_share'], False)} | {_k(mkt6.loc[m, 'Base'])} | {_k(mkt6.loc[m, 'Downside'])} "
        f"| {_k(mkt6.loc[m, 'Upside'])} | {_p(msc.loc[m, 'revenue_growth_last6m'])} |" for m in cfg.MARKETS) + f"""

![Scenarios](images/04_scenarios.png)

## Why

- **Growth came from new customers, not from customers coming back.** Customers explain
  {yoy.loc['Total', 'contrib_customers']/yoy.loc['Total', 'revenue_change']:.0%} of the {_p(yoy.loc['Total', 'revenue_growth'])}
  revenue rise ({cfg.YOY_LABEL}). Only {_p(dq['repeat_rate_model_window'], False)} of customers ordered more than once;
  orders per customer sits at {base.loc['Total', 'orders_per_customer']:.3f}. Retention is not a revenue lever on this horizon.
- **The growth has stalled.** Monthly customers in {cfg.BASELINE_LABEL} were {_p(base.loc['Total', 'customers_growth_vs_prior'])}
  vs {cfg.PRIOR3_LABEL}, and down in every market. A plan that extends 2017's growth would be wrong: in the backtest, the
  trend models over-forecast by {_p(trend_bias.min())} to {_p(trend_bias.max())}.
- **A flat forecast is the most accurate of the {len(wa)} we tested**, including the richer models. The 3-month average had
  the lowest average error (WAPE {wa['WindowAverage']:.1%} across the total and 5 markets). All of these did worse: a
  regression on customer momentum ({wa['LagRegression']:.1%}), a driver-based model ({wa['Drivers']:.1%}), Prophet using
  forecast KPIs as regressors ({wa['ProphetKPI']:.1%}), Prophet using lagged leading indicators such as late deliveries
  ({wa['ProphetLead']:.1%}) and Prophet with seasonality learned from daily data ({wa['Prophet']:.0%}, unstable with under
  two years of history).
- **Planning from {cfg.month_label(cfg.CHECK_END)}, the backtest would have picked {NAMES[picked]}**, and it over-forecast
  {cfg.CHECK_LABEL} by {_p(chk.loc[picked, 'check_total_error'])}: growth stopped in July, a turning point no model saw
  coming. The flat 3-month average was {_p(chk.loc['WindowAverage', 'check_total_error'])} out. This is why the plan is flat and
  why it comes with explicit triggers.
- **SP is where size and momentum meet.** {_p(msc.loc['SP', 'revenue_share'], False)} of revenue and
  {_p(msc.loc['SP', 'customers_growth_last6m'])} customers over the last 6 months, the only market with strong customer growth.

![Where to invest](images/04_market_scorecard.png)

## What would change the plan

Check average monthly customers for {trig_label} against the base of {tr['base']:,.0f}. Each threshold sits
halfway between the base path and the other scenario's path:

- **Below {tr['down']:,.0f}**: the decline is continuing. Re-plan on the downside ({_m(dn['six_months'])}).
- **Above {tr['up']:,.0f}**: momentum is back. Move towards the upside ({_m(up['six_months'])}).
- **November**: the uplift rests on one Black Friday (2017). Confirm promotion plans with the business before
  committing the {_m(bf_rev)}.

## Limits

- {cfg.HISTORY_MONTHS} months of usable history ({cfg.HISTORY_LABEL}) and a single November, so no yearly seasonality
  can be learned reliably; Black Friday is handled as an explicit uplift.
- No cost, margin or marketing data: "where to invest" is **revenue sensitivity, not return on investment**.
- Revenue is item price on delivered orders ({dq['orders_excluded_not_delivered']:,} non-delivered orders excluded; freight excluded).
- Olist is a marketplace: these markets are customer locations, not sales teams.

## How this was built

Data cleaning and checks (`src/prepare.py`), driver tree (`src/drivers.py`), backtested forecast and the
forecast check (`src/forecast.py`) and scenarios (`src/scenarios.py`), with notebooks `01`–`04` and an
interactive planner (`streamlit run app/streamlit_app.py`). Reproduce everything with `python -m src.run_all`.
"""


def main() -> str:
    text = render(load())
    OUT.write_text(text, encoding="utf-8")
    return str(OUT)


if __name__ == "__main__":
    print(main())
