"""Stage 4: scenario engine and investment scorecards.

For each market m and forecast month t = 1..6:

    customers_t = customers_0 x (1 + g_customers)^t
    aov_t       = aov_0       x (1 + g_aov)^t
    revenue_t   = customers_t x orders_per_customer_0 x aov_t   (x Black Friday uplift in November)

Starting values are the Stage 2 baseline (Jun-Aug 2018 averages). With both growth rates at zero the
projection equals the Stage 3 baseline forecast (3-month average) exactly.

Preset scenarios, each derived from observed growth and converted to a monthly rate:

    Downside  the latest quarter continues: customers Jun-Aug 2018 vs Mar-May 2018 (3 months apart),
              AOV held flat
    Base      flat: no change in any driver
    Upside    the last-6-month momentum continues: customers and AOV, Mar-Aug 2018 vs Sep 2017-Feb 2018
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
    "Base": "Flat: drivers stay at Jun-Aug 2018 levels",
    "Upside": "Last-6-month momentum continues (customers and AOV)",
}


def monthly_rate(period_growth: float | pd.Series, months_apart: int):
    """Convert growth between two periods `months_apart` months apart into a compound monthly rate."""
    return (1 + period_growth) ** (1 / months_apart) - 1


def load_baseline() -> pd.DataFrame:
    b = pd.read_csv(P / "driver_baseline.csv").set_index("market")
    return b.loc[cfg.MARKETS, ["customers", "orders_per_customer", "aov", "revenue",
                               "customers_growth_vs_prior"]]


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
            rows.append({"market": m, "month": month, "step": t, "customers": cust,
                         "orders_per_customer": base.loc[m, "orders_per_customer"], "aov": aov,
                         "revenue": rev * bf})
    return pd.DataFrame(rows)


def summarise(proj: pd.DataFrame) -> pd.DataFrame:
    """6-month and plan-quarter totals per market, plus a Total row."""
    q = np.where(proj["step"] <= 3, "plan_q1", "plan_q2")
    s = proj.assign(q=q).pivot_table(index="market", columns="q", values="revenue", aggfunc="sum")
    s["six_months"] = s.sum(axis=1)
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
