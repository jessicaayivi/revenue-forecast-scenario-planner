"""Basic data tests: run `pytest` from the repo root after `python -m src.prepare`."""
import json

import pandas as pd
import pytest

from src import config as cfg, prepare

P = cfg.PROCESSED_DIR


@pytest.fixture(scope="module")
def raw():
    return prepare.load_raw()


@pytest.fixture(scope="module")
def line_items(raw):
    li = prepare.build_line_items(raw)
    return prepare.add_category_group(li, prepare.top_categories(li))


def test_no_duplicate_orders(raw):
    assert raw["orders"]["order_id"].is_unique
    assert not raw["items"].duplicated(["order_id", "order_item_id"]).any()


def test_only_delivered_orders(raw, line_items):
    status = raw["orders"].set_index("order_id")["order_status"]
    assert (status.loc[line_items["order_id"].unique()] == cfg.KEEP_STATUS).all()


def test_every_line_has_market_and_category(line_items):
    assert line_items["market"].isin(cfg.MARKETS).all()
    assert line_items["category"].notna().all()
    assert line_items["category"].nunique() == cfg.TOP_N_CATEGORIES + 1


def test_revenue_reconciles_to_raw_items(raw, line_items):
    delivered = raw["orders"].loc[raw["orders"]["order_status"] == cfg.KEEP_STATUS, "order_id"]
    raw_rev = raw["items"].loc[raw["items"]["order_id"].isin(delivered), "price"].sum()
    assert line_items["price"].sum() == pytest.approx(raw_rev)
    total = pd.read_csv(P / "monthly_total.csv")
    assert total["revenue"].sum() == pytest.approx(raw_rev, abs=1.0)


@pytest.mark.parametrize("name,key", [("monthly_market.csv", "market"),
                                      ("monthly_category.csv", "category"),
                                      ("monthly_market_category.csv", None)])
def test_breakdowns_add_up_to_total(name, key):
    total = pd.read_csv(P / "monthly_total.csv").set_index("month")["revenue"]
    part = pd.read_csv(P / name).groupby("month")["revenue"].sum()
    pd.testing.assert_series_equal(part.sort_index(), total.sort_index(), check_names=False, atol=1.0)


def test_market_orders_add_up_to_total():
    # each order has one customer state, so market order counts must add up exactly
    total = pd.read_csv(P / "monthly_total.csv").set_index("month")["orders"]
    part = pd.read_csv(P / "monthly_market.csv").groupby("month")["orders"].sum()
    assert (part.sort_index() == total.sort_index()).all()


def test_model_window_is_complete():
    total = pd.read_csv(P / "monthly_total.csv")
    w = total[total["in_model_window"]]
    expected = pd.period_range(cfg.MODEL_START, cfg.MODEL_END, freq="M").astype(str).tolist()
    assert w["month"].tolist() == expected


def test_state_summary_matches_markets():
    s = pd.read_csv(P / "state_summary.csv")
    assert len(s) == len(cfg.STATES) and set(s["state"]) == set(cfg.STATES)
    assert s["revenue_share"].sum() == pytest.approx(1.0, abs=1e-3)
    # states roll up to the market tables used by the scenarios
    base = pd.read_csv(P / "driver_baseline.csv").set_index("market")["revenue"]
    roll = s.groupby("market")["revenue_month_baseline"].sum()
    assert roll.loc[cfg.MARKETS].to_numpy() == pytest.approx(base.loc[cfg.MARKETS].to_numpy(), abs=0.01)
    assert (s["market"] == s["state"].map(cfg.state_to_market)).all()
    # every state sits inside Brazil's bounding box
    assert s["lat"].between(-34, 6).all() and s["lon"].between(-74, -34).all()


def test_daily_series_adds_up_to_monthly():
    d = pd.read_csv(P / "daily_market.csv")
    m = pd.read_csv(P / "monthly_market.csv")
    m = m[m["in_model_window"]].set_index(["month", "market"])["revenue"]
    roll = d.assign(month=d["date"].str[:7]).groupby(["month", "market"])["revenue"].sum()
    assert roll.loc[m.index].to_numpy() == pytest.approx(m.to_numpy(), abs=0.05)
    days = pd.to_datetime(d["date"].unique())
    assert len(days) == (days.max() - days.min()).days + 1  # every day present


def test_quality_report_matches_tables():
    dq = json.loads((P / "data_quality.json").read_text())
    total = pd.read_csv(P / "monthly_total.csv")
    w = total[total["in_model_window"]]
    assert dq["model_window_revenue"] == pytest.approx(w["revenue"].sum(), abs=1.0)
    assert dq["model_window_orders"] == w["orders"].sum()
