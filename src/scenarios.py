"""Stage 4: scenario engine and investment scorecards.

For each market m and forecast month t = 1..6:

    customers_t = customers_0 x (1 + g_customers)^t
    aov_t       = aov_0       x (1 + g_aov)^t
    revenue_t   = customers_t x orders_per_customer_0 x aov_t   (x Black Friday uplift in November)

AOV is items per order x price per item. Basket size is held at its baseline, so AOV growth is carried
by price per item. The what-if functions below change any one driver, or price and customers together
(price-volume trade-off).

Months t = 1..n are the forecast months of the plan half-year (cfg.FORECAST_LABEL); the plan's earlier
months are actuals and are added unchanged in `summarise`. Starting values are the Stage 2 baseline
(averages of the last 3 months of data). With both growth rates at zero the projection equals the
Stage 3 baseline forecast (3-month average) exactly.

Preset scenarios, each derived from observed growth and converted to a monthly rate:

    Downside  the latest quarter continues: customers, last 3 months vs the 3 before (3 months apart),
              AOV held flat
    Base      flat: no change in any driver
    Upside    the last-6-month momentum continues: customers and AOV, last 6 months vs the 6 before
              (6 months apart)

Run: python -m src.scenarios   (needs src.drivers and src.forecast outputs)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as cfg

P = cfg.PROCESSED_DIR

DESCRIPTIONS = {
    "Downside": "Latest quarter's customer decline continues; AOV flat",
    "Base": f"Flat: drivers stay at {cfg.BASELINE_LABEL} levels",
    "Upside": "Last-6-month momentum continues (customers and AOV)",
}


def monthly_rate(period_growth: float | pd.Series, months_apart: int):
    """Convert growth between two periods `months_apart` months apart into a compound monthly rate."""
    return (1 + period_growth) ** (1 / months_apart) - 1


def load_baseline() -> pd.DataFrame:
    b = pd.read_csv(P / "driver_baseline.csv").set_index("market")
    return b.loc[cfg.MARKETS, ["customers", "orders_per_customer", "aov", "items_per_order", "price_per_item",
                               "revenue", "customers_growth_vs_prior"]]


def black_friday_uplift() -> float:
    return float(json.loads((P / "forecast_summary.json").read_text())["black_friday_uplift"])


def forecast_months() -> list[str]:
    end = pd.Period(cfg.MODEL_END, "M")
    return [str(end + i) for i in range(1, cfg.FORECAST_H + 1)]


def preset_scenarios(base: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Monthly growth rates per market for each preset. Columns: g_customers, g_aov."""
    base = load_baseline() if base is None else base
    dec = pd.read_csv(P / "driver_decomposition.csv")
    last6 = dec[dec["comparison"] == cfg.COMPARISONS[1][0]].set_index("market").loc[cfg.MARKETS]
    zero = pd.Series(0.0, index=cfg.MARKETS)
    return {
        "Downside": pd.DataFrame({"g_customers": monthly_rate(base["customers_growth_vs_prior"], 3),
                                  "g_aov": zero}),
        "Base": pd.DataFrame({"g_customers": zero, "g_aov": zero}),
        "Upside": pd.DataFrame({"g_customers": monthly_rate(last6["customers_growth"], 6),
                                "g_aov": monthly_rate(last6["aov_growth"], 6)}),
    }


def project(params: pd.DataFrame, black_friday: bool = True, base: pd.DataFrame | None = None,
            uplift: float | None = None) -> pd.DataFrame:
    """Monthly projection per market. `params`: index = market, columns g_customers, g_aov."""
    base = load_baseline() if base is None else base
    uplift = black_friday_uplift() if (black_friday and uplift is None) else (uplift or 1.0)
    months = forecast_months()
    rows = []
    for m in cfg.MARKETS:
        gc, ga = float(params.loc[m, "g_customers"]), float(params.loc[m, "g_aov"])
        for t, month in enumerate(months, start=1):
            cust = base.loc[m, "customers"] * (1 + gc) ** t
            aov = base.loc[m, "aov"] * (1 + ga) ** t
            rev = cust * base.loc[m, "orders_per_customer"] * aov
            bf = uplift if (black_friday and int(month[-2:]) == cfg.BLACK_FRIDAY_MONTH) else 1.0
            ipo = base.loc[m, "items_per_order"]  # basket size held; AOV growth shows up as price per item
            rows.append({"market": m, "month": month, "step": t, "customers": cust,
                         "orders_per_customer": base.loc[m, "orders_per_customer"], "aov": aov,
                         "items_per_order": ipo, "price_per_item": aov / ipo,
                         "black_friday_factor": bf, "revenue": rev * bf})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what-if hypotheses
# AOV = items per order x price per item, so a change to either moves AOV; a change to AOV itself is
# treated as a price change (basket size held).
LEVERS = {
    "customers": "Customers",
    "orders_per_customer": "Orders per customer",
    "items_per_order": "Items per order (basket size)",
    "price_per_item": "Price per item",
    "aov": "Average order value (AOV)",
}


def apply_hypothesis(proj: pd.DataFrame, lever: str, change: float, markets: list[str],
                     start_month: str) -> pd.DataFrame:
    """Raise (or cut) one driver by `change` (0.10 = +10%) in `markets`, from `start_month` onwards.

    The change is a one-off step in the level of the driver, on top of whatever path the scenario
    already has. Revenue is rebuilt as customers x orders per customer x AOV x Black Friday factor.
    """
    if lever not in LEVERS:
        raise ValueError(f"Unknown lever {lever!r}; choose from {list(LEVERS)}")
    h = proj.copy()
    hit = h["market"].isin(markets) & (h["month"] >= start_month)
    col = "price_per_item" if lever == "aov" else lever
    h.loc[hit, col] = h.loc[hit, col] * (1 + change)
    h["aov"] = h["items_per_order"] * h["price_per_item"]
    h["revenue"] = h["customers"] * h["orders_per_customer"] * h["aov"] * h["black_friday_factor"]
    return h


def price_volume(proj: pd.DataFrame, price_change: float, elasticity: float, markets: list[str],
                 start_month: str) -> pd.DataFrame:
    """Change price per item and let customers respond with a constant price elasticity.

    customers x (1 + price_change) ** (-elasticity). Elasticity 1 means a 1% price rise loses about
    1% of customers. Revenue changes by (1 + p) ** (1 - elasticity) - 1, so it breaks even at
    elasticity 1 whatever the price change. The elasticity is an input: the data has no price tests.
    """
    if elasticity < 0:
        raise ValueError("elasticity must be >= 0 (customers fall when prices rise)")
    h = apply_hypothesis(proj, "price_per_item", price_change, markets, start_month)
    return apply_hypothesis(h, "customers", (1 + price_change) ** (-elasticity) - 1, markets, start_month)


def break_even_customer_change(price_change: float) -> float:
    """Customer change that leaves revenue flat after a price change (e.g. +5% price -> -4.8% customers)."""
    return 1 / (1 + price_change) - 1


def price_volume_curve(proj: pd.DataFrame, price_change: float, markets: list[str], start_month: str,
                       elasticities=None) -> pd.DataFrame:
    """Revenue impact over the forecast months across a range of customer responses (elasticities)."""
    elasticities = np.round(np.arange(0, 3.01, 0.1), 2) if elasticities is None else elasticities
    before = proj["revenue"].sum()
    rows = []
    for e in elasticities:
        after = price_volume(proj, price_change, float(e), markets, start_month)["revenue"].sum()
        rows.append({"elasticity": float(e), "revenue_change": after - before,
                     "revenue_change_pct": after / before - 1})
    return pd.DataFrame(rows)


def hypothesis_impact(proj: pd.DataFrame, hyp: pd.DataFrame) -> pd.DataFrame:
    """Revenue before/after per market plus Total: Plan Q1, Plan Q2 and 6 months."""
    a, b = summarise(proj).set_index("market"), summarise(hyp).set_index("market")
    out = pd.DataFrame({"before": a["six_months"], "after": b["six_months"],
                        "plan_q1_change": b["plan_q1"] - a["plan_q1"],
                        "plan_q2_change": b["plan_q2"] - a["plan_q2"]})
    out["change"] = out["after"] - out["before"]
    out["change_pct"] = out["change"] / out["before"]
    return out.reset_index()


def hypothesis_by_market(proj: pd.DataFrame, lever: str, change: float, start_month: str) -> pd.DataFrame:
    """The same hypothesis applied to one market at a time: where is it worth most?"""
    rows = []
    for m in cfg.MARKETS:
        imp = hypothesis_impact(proj, apply_hypothesis(proj, lever, change, [m], start_month)).set_index("market")
        rows.append({"market": m, "extra_revenue": imp.loc["Total", "change"],
                     "share_of_total_revenue": imp.loc["Total", "change"] / imp.loc["Total", "before"]})
    return pd.DataFrame(rows)


_ACTUALS: pd.DataFrame | None = None


def plan_actuals() -> pd.DataFrame:
    """Actual revenue per market for the plan months that have already happened (e.g. Jul-Aug 2018)."""
    global _ACTUALS
    if _ACTUALS is None:
        m = pd.read_csv(P / "monthly_market.csv")
        _ACTUALS = m[m["month"].isin(cfg.ACTUAL_MONTHS)][["market", "month", "revenue"]].reset_index(drop=True)
    return _ACTUALS


def summarise(proj: pd.DataFrame) -> pd.DataFrame:
    """Plan half-year totals per market, plus a Total row.

    `actual` = plan months already in the data; `forecast` = the projected months; `plan_q1` and
    `plan_q2` = the two quarters of the half-year (actuals + forecast); `six_months` = the half-year total.
    """
    both = pd.concat([plan_actuals().assign(kind="actual"), proj[["market", "month", "revenue"]].assign(kind="forecast")])
    q1_end = str(pd.Period(cfg.PLAN_START, "M") + 2)
    both["q"] = np.where(both["month"] <= q1_end, "plan_q1", "plan_q2")
    s = both.pivot_table(index="market", columns="q", values="revenue", aggfunc="sum")
    k = both.pivot_table(index="market", columns="kind", values="revenue", aggfunc="sum")
    s = s.join(k).reindex(columns=["actual", "forecast", "plan_q1", "plan_q2"]).fillna(0.0)
    s["six_months"] = s["plan_q1"] + s["plan_q2"]
    s = s.loc[cfg.MARKETS]
    s.loc["Total"] = s.sum()
    return s.reset_index()


def sensitivity(base: pd.DataFrame | None = None, step: float = cfg.SENSITIVITY_STEP) -> pd.DataFrame:
    """Extra 6-month revenue (vs flat, no Black Friday) from +step monthly growth in one driver of one market."""
    base = load_baseline() if base is None else base
    zero = preset_scenarios(base)["Base"]
    flat = project(zero, black_friday=False, base=base).groupby("market")["revenue"].sum()
    rows = []
    for m in cfg.MARKETS:
        row = {"market": m}
        for drv in ("g_customers", "g_aov"):
            p = zero.copy()
            p.loc[m, drv] = step
            rev = project(p, black_friday=False, base=base).groupby("market")["revenue"].sum()
            row[f"extra_revenue_{drv[2:]}"] = rev[m] - flat[m]
        rows.append(row)
    return pd.DataFrame(rows)


def market_scorecard() -> pd.DataFrame:
    base = load_baseline()
    dec = pd.read_csv(P / "driver_decomposition.csv")
    yoy = dec[dec["comparison"] == cfg.COMPARISONS[0][0]].set_index("market")
    l6 = dec[dec["comparison"] == cfg.COMPARISONS[1][0]].set_index("market")
    sens = sensitivity(base).set_index("market")
    share = pd.read_csv(P / "driver_baseline.csv").set_index("market")["revenue_share"]
    sc = pd.DataFrame({
        "revenue_share": share.loc[cfg.MARKETS],
        "baseline_revenue_month": base["revenue"],
        "aov": base["aov"],
        "revenue_growth_yoy": yoy.loc[cfg.MARKETS, "revenue_growth"],
        "revenue_growth_last6m": l6.loc[cfg.MARKETS, "revenue_growth"],
        "customers_growth_last6m": l6.loc[cfg.MARKETS, "customers_growth"],
        "aov_growth_last6m": l6.loc[cfg.MARKETS, "aov_growth"],
        "customers_growth_last3m": base["customers_growth_vs_prior"],
        "share_of_recent_growth": l6.loc[cfg.MARKETS, "share_of_total_change"],
        "extra_6m_revenue_per_1pt_customers": sens["extra_revenue_customers"],
    })
    return sc.rename_axis("market").reset_index()


def category_scorecard() -> pd.DataFrame:
    c = pd.read_csv(P / "monthly_category.csv")
    c = c[c["in_model_window"]]

    def window(a, b):
        return c[(c["month"] >= a) & (c["month"] <= b)].groupby("category")[["revenue", "orders"]].sum()

    (_, yb0, yb1, yc0, yc1), (_, lb0, lb1, lc0, lc1) = cfg.COMPARISONS
    yb, yc, lb, lc = window(yb0, yb1), window(yc0, yc1), window(lb0, lb1), window(lc0, lc1)
    sc = pd.DataFrame({
        "revenue_share_last6m": lc["revenue"] / lc["revenue"].sum(),
        "revenue_last6m": lc["revenue"],
        "aov_last6m": lc["revenue"] / lc["orders"],
        "revenue_growth_yoy": yc["revenue"] / yb["revenue"] - 1,
        "revenue_growth_last6m": lc["revenue"] / lb["revenue"] - 1,
        "aov_growth_last6m": (lc["revenue"] / lc["orders"]) / (lb["revenue"] / lb["orders"]) - 1,
    }).sort_values("revenue_last6m", ascending=False)
    return sc.rename_axis("category").reset_index()


def main() -> dict:
    base = load_baseline()
    presets = preset_scenarios(base)
    projs, sums, params = [], [], []
    for name in cfg.SCENARIOS:
        p = project(presets[name], black_friday=True, base=base).assign(scenario=name)
        projs.append(p)
        sums.append(summarise(p).assign(scenario=name))
        params.append(presets[name].rename_axis("market").reset_index().assign(
            scenario=name, description=DESCRIPTIONS[name]))
    pd.concat(projs).round(4).to_csv(P / "scenario_projection.csv", index=False)
    summary = pd.concat(sums)
    summary.round(2).to_csv(P / "scenario_summary.csv", index=False)
    pd.concat(params).round(6).to_csv(P / "scenario_params.csv", index=False)
    market_scorecard().round(4).to_csv(P / "market_scorecard.csv", index=False)
    category_scorecard().round(4).to_csv(P / "category_scorecard.csv", index=False)
    tot = summary[summary["market"] == "Total"].set_index("scenario")
    return {s: {"six_months": round(float(r["six_months"]), 0), "plan_q1": round(float(r["plan_q1"]), 0),
                "plan_q2": round(float(r["plan_q2"]), 0)} for s, r in tot.iterrows()}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
