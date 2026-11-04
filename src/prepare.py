"""Stage 1: clean and join the raw Olist tables into tidy monthly tables.

Revenue definition: item price only (freight excluded); GMV = price + freight, kept as a reference.
Only `delivered` orders count. Month = order purchase month.
Customer = `customer_unique_id` (Olist's `customer_id` is one per order, so it can't measure repeats).

Run: python -m src.prepare
"""
from __future__ import annotations

import json

import pandas as pd

from src import config as cfg


# ---------------------------------------------------------------- load
def load_raw(raw_dir=cfg.RAW_DIR) -> dict[str, pd.DataFrame]:
    missing = [f for f in cfg.RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files in {raw_dir}: {missing}")
    t = {k: pd.read_csv(raw_dir / f) for k, f in cfg.RAW_FILES.items()}
    t["orders"]["order_purchase_timestamp"] = pd.to_datetime(t["orders"]["order_purchase_timestamp"])
    return t


def in_model_window(month: pd.Series) -> pd.Series:
    return (month >= pd.Period(cfg.MODEL_START, "M")) & (month <= pd.Period(cfg.MODEL_END, "M"))


# ---------------------------------------------------------------- build
def build_line_items(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per delivered order item, with month, market and English category."""
    orders = t["orders"]
    delivered = orders[orders["order_status"] == cfg.KEEP_STATUS].copy()
    delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M")

    products = t["products"].merge(t["translation"], on="product_category_name", how="left")
    # Two Portuguese names have no translation: keep the Portuguese name. Missing names -> "unknown".
    products["category_raw"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna(cfg.UNKNOWN_CATEGORY)
    )

    li = (
        t["items"][["order_id", "product_id", "price", "freight_value"]]
        .merge(delivered[["order_id", "customer_id", "month"]], on="order_id", how="inner")
        .merge(t["customers"][["customer_id", "customer_unique_id", "customer_state"]],
               on="customer_id", how="left")
        .merge(products[["product_id", "category_raw"]], on="product_id", how="left")
    )
    li["market"] = li["customer_state"].map(cfg.state_to_market)
    li["gmv"] = li["price"] + li["freight_value"]
    return li


def top_categories(li: pd.DataFrame, n: int = cfg.TOP_N_CATEGORIES) -> list[str]:
    """Top-n categories by revenue inside the modelling window ("unknown" is never a top category)."""
    w = li[in_model_window(li["month"])]
    rev = w[w["category_raw"] != cfg.UNKNOWN_CATEGORY].groupby("category_raw")["price"].sum()
    return rev.sort_values(ascending=False).head(n).index.tolist()


def add_category_group(li: pd.DataFrame, top: list[str]) -> pd.DataFrame:
    li = li.copy()
    li["category"] = li["category_raw"].where(li["category_raw"].isin(top), cfg.OTHER_CATEGORY)
    return li


def aggregate(li: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Monthly revenue, GMV, orders, unique customers and derived drivers for any grouping.

    Note: an order (or customer) that spans several categories counts once in each, so category
    order/customer counts do not add up to the total. Revenue always adds up.
    """
    keys = ["month"] + by
    g = li.groupby(keys, observed=True).agg(
        revenue=("price", "sum"),
        gmv=("gmv", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_unique_id", "nunique"),
        items=("order_id", "size"),
    ).reset_index()
    g["aov"] = g["revenue"] / g["orders"]
    g["orders_per_customer"] = g["orders"] / g["customers"]
    g["revenue_per_customer"] = g["revenue"] / g["customers"]
    g["in_model_window"] = in_model_window(g["month"])
    g["month"] = g["month"].astype(str)
    return g.round({"revenue": 2, "gmv": 2, "aov": 2, "orders_per_customer": 4,
                    "revenue_per_customer": 2})


def status_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Order counts by purchase month and status, with a flag for months too thin to model."""
    o = orders.copy()
    o["month"] = o["order_purchase_timestamp"].dt.to_period("M").astype(str)
    tab = pd.crosstab(o["month"], o["order_status"])
    tab["total"] = tab.sum(axis=1)
    tab["delivered_share"] = (tab["delivered"] / tab["total"]).round(4)
    tab["sparse_flag"] = tab["delivered"] < 500
    return tab.reset_index()


def order_outliers(li: pd.DataFrame, q: float = 0.999) -> pd.DataFrame:
    """Orders above the 99.9th percentile of order revenue. Flagged for review, not removed."""
    o = li.groupby(["order_id", "month", "market"]).agg(
        revenue=("price", "sum"), items=("price", "size"), max_item_price=("price", "max"),
        category=("category", "first"),
    ).reset_index()
    cut = o["revenue"].quantile(q)
    out = o[o["revenue"] > cut].sort_values("revenue", ascending=False).copy()
    out["threshold"] = round(cut, 2)
    out["month"] = out["month"].astype(str)
    return out


def quality_report(t: dict[str, pd.DataFrame], li: pd.DataFrame, top: list[str]) -> dict:
    orders, items, pay = t["orders"], t["items"], t["payments"]
    delivered = orders[orders["order_status"] == cfg.KEEP_STATUS]

    item_tot = items.groupby("order_id")[["price", "freight_value"]].sum()
    pay_tot = pay.groupby("order_id")["payment_value"].sum()
    rec = item_tot.join(pay_tot, how="inner")
    gap = (rec["payment_value"] - rec["price"] - rec["freight_value"]).abs()

    w = li[in_model_window(li["month"])]
    per_cust = w.groupby("customer_unique_id")["order_id"].nunique()
    state_rev = w.groupby("customer_state")["price"].sum().sort_values(ascending=False)
    state_share = state_rev / state_rev.sum()
    market_share = (w.groupby("market")["price"].sum() / w["price"].sum()).sort_values(ascending=False)
    cat_share = (w.groupby("category")["price"].sum() / w["price"].sum()).sort_values(ascending=False)

    return {
        "raw_rows": {k: int(len(v)) for k, v in t.items()},
        "duplicate_order_ids": int(orders["order_id"].duplicated().sum()),
        "duplicate_item_keys": int(items.duplicated(["order_id", "order_item_id"]).sum()),
        "order_status_counts": {k: int(v) for k, v in orders["order_status"].value_counts().items()},
        "orders_excluded_not_delivered": int(len(orders) - len(delivered)),
        "delivered_orders_without_delivery_date": int(
            pd.to_datetime(delivered["order_delivered_customer_date"]).isna().sum()),
        "delivered_orders_without_items": int((~delivered["order_id"].isin(items["order_id"])).sum()),
        "products_missing_category": int(t["products"]["product_category_name"].isna().sum()),
        "payments_vs_items_gap_over_1brl_share": round(float((gap > 1).mean()), 4),
        "delivered_orders_after_model_window": int(
            li.loc[li["month"] > pd.Period(cfg.MODEL_END, "M"), "order_id"].nunique()),
        "delivered_orders_before_model_window": int(
            li.loc[li["month"] < pd.Period(cfg.MODEL_START, "M"), "order_id"].nunique()),
        "model_window": [cfg.MODEL_START, cfg.MODEL_END],
        "model_window_months": int(w["month"].nunique()),
        "model_window_revenue": round(float(w["price"].sum()), 2),
        "model_window_gmv": round(float(w["gmv"].sum()), 2),
        "model_window_orders": int(w["order_id"].nunique()),
        "model_window_customers": int(w["customer_unique_id"].nunique()),
        "repeat_rate_model_window": round(float((per_cust > 1).mean()), 4),
        "orders_per_customer_model_window": round(float(per_cust.mean()), 4),
        "top5_states": state_share.head(5).index.tolist(),
        "top5_state_revenue_share": round(float(state_share.head(5).sum()), 4),
        "market_revenue_share": {k: round(float(v), 4) for k, v in market_share.items()},
        "top_categories": top,
        "category_revenue_share": {k: round(float(v), 4) for k, v in cat_share.items()},
    }


# ---------------------------------------------------------------- run
def main() -> dict:
    t = load_raw()
    li = build_line_items(t)
    top = top_categories(li)
    li = add_category_group(li, top)

    out = cfg.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    aggregate(li, []).to_csv(out / "monthly_total.csv", index=False)
    aggregate(li, ["market"]).to_csv(out / "monthly_market.csv", index=False)
    aggregate(li, ["category"]).to_csv(out / "monthly_category.csv", index=False)
    aggregate(li, ["market", "category"]).to_csv(out / "monthly_market_category.csv", index=False)
    status_summary(t["orders"]).to_csv(out / "order_status_by_month.csv", index=False)
    order_outliers(li).to_csv(out / "order_outliers.csv", index=False)

    report = quality_report(t, li, top)
    (out / "data_quality.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
