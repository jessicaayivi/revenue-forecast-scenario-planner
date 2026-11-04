"""Stage 2: revenue driver tree by market.

    revenue = customers x orders_per_customer x AOV

Monthly, per market. For a period of several months, customers are summed month by month
("customer-months"), so the identity holds exactly for the period total too.

The change in revenue between two periods is split into one contribution per driver with the
LMDI method (logarithmic mean Divisia index). It is exact: driver contributions add up to the
market's revenue change, and market changes add up to the total change. No residual term.

Run: python -m src.drivers   (needs src.prepare outputs)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as cfg

P = cfg.PROCESSED_DIR


def load_market() -> pd.DataFrame:
    m = pd.read_csv(P / "monthly_market.csv")
    m = m[m["in_model_window"]].copy()
    return m[["month", "market", "revenue", "orders", "customers", "items"]]


def driver_tree(m: pd.DataFrame) -> pd.DataFrame:
    """Monthly driver table per market, plus a 'Total' row per month and each market's revenue share.

    AOV is further split into items per order (basket size) x price per item.
    """
    tot = m.groupby("month", as_index=False)[["revenue", "orders", "customers", "items"]].sum()
    tot["market"] = "Total"
    d = pd.concat([m, tot], ignore_index=True)
    d["orders_per_customer"] = d["orders"] / d["customers"]
    d["aov"] = d["revenue"] / d["orders"]
    d["items_per_order"] = d["items"] / d["orders"]
    d["price_per_item"] = d["revenue"] / d["items"]
    month_total = d.loc[d["market"] == "Total"].set_index("month")["revenue"]
    d["revenue_share"] = d["revenue"] / d["month"].map(month_total)
    order = {k: i for i, k in enumerate(cfg.MARKETS + ["Total"])}
    d = d.sort_values(["month", "market"], key=lambda s: s.map(order) if s.name == "market" else s)
    return d.reset_index(drop=True)


def period_drivers(m: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    x = m[(m["month"] >= start) & (m["month"] <= end)]
    g = x.groupby("market")[["revenue", "orders", "customers"]].sum()
    g["orders_per_customer"] = g["orders"] / g["customers"]
    g["aov"] = g["revenue"] / g["orders"]
    g["months"] = x["month"].nunique()
    return g


def _logmean(a: pd.Series, b: pd.Series) -> pd.Series:
    """Logarithmic mean L(a, b) = (a - b) / (ln a - ln b), with L(a, a) = a."""
    a, b = a.astype(float), b.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = (a - b) / (np.log(a) - np.log(b))
    return lm.where(~np.isclose(a, b), a)


def decompose(m: pd.DataFrame, label: str, b0: str, b1: str, c0: str, c1: str) -> pd.DataFrame:
    """Split each market's revenue change between two periods into driver contributions (LMDI)."""
    base, cur = period_drivers(m, b0, b1), period_drivers(m, c0, c1)
    if base["months"].iloc[0] != cur["months"].iloc[0]:
        raise ValueError(f"{label}: periods have different lengths")
    w = _logmean(cur["revenue"], base["revenue"])
    out = pd.DataFrame({
        "comparison": label,
        "base_period": f"{b0} to {b1}",
        "current_period": f"{c0} to {c1}",
        "base_revenue": base["revenue"],
        "current_revenue": cur["revenue"],
        "revenue_change": cur["revenue"] - base["revenue"],
        "revenue_growth": cur["revenue"] / base["revenue"] - 1,
    })
    for drv in cfg.DRIVERS:
        out[f"{drv}_growth"] = cur[drv] / base[drv] - 1
        out[f"contrib_{drv}"] = w * np.log(cur[drv] / base[drv])
    out = out.reindex(cfg.MARKETS).rename_axis("market").reset_index()
    total = out.select_dtypes("number").sum()
    total_row = {"market": "Total", "comparison": label, "base_period": out["base_period"].iloc[0],
                 "current_period": out["current_period"].iloc[0], **total.to_dict()}
    # growth rates do not add up across markets: recompute them for the total
    tb, tc = base[["revenue", "orders", "customers"]].sum(), cur[["revenue", "orders", "customers"]].sum()
    total_row["revenue_growth"] = tc["revenue"] / tb["revenue"] - 1
    total_row["customers_growth"] = tc["customers"] / tb["customers"] - 1
    total_row["orders_per_customer_growth"] = (tc["orders"] / tc["customers"]) / (tb["orders"] / tb["customers"]) - 1
    total_row["aov_growth"] = (tc["revenue"] / tc["orders"]) / (tb["revenue"] / tb["orders"]) - 1
    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
    out["share_of_total_change"] = out["revenue_change"] / total_row["revenue_change"]
    return out


def baseline(tree: pd.DataFrame, n: int = cfg.BASELINE_MONTHS) -> pd.DataFrame:
    """Starting point for scenarios: each driver averaged over the last n months, per market.

    Customers is the plain monthly average. Orders per customer and AOV are ratios of the n-month
    sums, so baseline revenue = customers x orders_per_customer x AOV exactly.
    Also reports trailing growth of monthly customers (last n months vs the n months before).
    """
    months = sorted(tree["month"].unique())
    last, prev = months[-n:], months[-2 * n:-n]
    rows = []
    for mk, g in tree.groupby("market", sort=False):
        a, b = g[g["month"].isin(last)], g[g["month"].isin(prev)]
        rows.append({
            "market": mk,
            "baseline_months": f"{last[0]} to {last[-1]}",
            "customers": a["customers"].mean(),
            "orders_per_customer": a["orders"].sum() / a["customers"].sum(),
            "aov": a["revenue"].sum() / a["orders"].sum(),
            "items_per_order": a["items"].sum() / a["orders"].sum(),
            "price_per_item": a["revenue"].sum() / a["items"].sum(),
            "revenue": a["revenue"].mean(),
            "customers_growth_vs_prior": a["customers"].sum() / b["customers"].sum() - 1,
            "revenue_share": a["revenue"].sum() / tree.loc[(tree["market"] == "Total") &
                                                           tree["month"].isin(last), "revenue"].sum(),
        })
    return pd.DataFrame(rows)


def main() -> dict:
    m = load_market()
    tree = driver_tree(m)
    dec = pd.concat([decompose(m, *c) for c in cfg.COMPARISONS], ignore_index=True)
    base = baseline(tree)

    tree.round({"revenue": 2, "orders_per_customer": 4, "aov": 2, "items_per_order": 4,
                "price_per_item": 2, "revenue_share": 4}).to_csv(
        P / "driver_tree_market.csv", index=False)
    dec.round(4).to_csv(P / "driver_decomposition.csv", index=False)
    base.round(4).to_csv(P / "driver_baseline.csv", index=False)

    tot = dec[dec["market"] == "Total"].set_index("comparison")
    summary = {
        c: {"revenue_growth": round(float(r["revenue_growth"]), 4),
            **{f"contrib_{d}": round(float(r[f"contrib_{d}"]), 0) for d in cfg.DRIVERS}}
        for c, r in tot.iterrows()
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
