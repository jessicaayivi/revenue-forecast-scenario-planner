"""Driver tree tests: run `pytest` after `python -m src.prepare` and `python -m src.drivers`."""
import pandas as pd
import pytest

from src import config as cfg, drivers

P = cfg.PROCESSED_DIR
CONTRIBS = [f"contrib_{d}" for d in cfg.DRIVERS]


@pytest.fixture(scope="module")
def tree():
    return drivers.driver_tree(drivers.load_market())


@pytest.fixture(scope="module")
def dec():
    m = drivers.load_market()
    return pd.concat([drivers.decompose(m, *c) for c in cfg.COMPARISONS], ignore_index=True)


def test_drivers_multiply_back_to_revenue(tree):
    rebuilt = tree["customers"] * tree["orders_per_customer"] * tree["aov"]
    assert rebuilt.to_numpy() == pytest.approx(tree["revenue"].to_numpy(), rel=1e-9)


def test_markets_add_up_to_total(tree):
    mk = tree[tree["market"] != "Total"].groupby("month")[["revenue", "orders"]].sum()
    tot = tree[tree["market"] == "Total"].set_index("month")[["revenue", "orders"]]
    pd.testing.assert_frame_equal(mk.sort_index(), tot.sort_index(), check_dtype=False)
    shares = tree[tree["market"] != "Total"].groupby("month")["revenue_share"].sum()
    assert shares.to_numpy() == pytest.approx(1.0)


def test_contributions_add_up_to_revenue_change(dec):
    # LMDI is exact: no residual, per market and for the total
    assert dec[CONTRIBS].sum(axis=1).to_numpy() == pytest.approx(dec["revenue_change"].to_numpy(), abs=0.01)


def test_market_changes_add_up_to_total(dec):
    for _, g in dec.groupby("comparison"):
        total = g[g["market"] == "Total"].iloc[0]
        markets = g[g["market"] != "Total"]
        assert markets["revenue_change"].sum() == pytest.approx(total["revenue_change"], abs=0.01)
        assert markets["share_of_total_change"].sum() == pytest.approx(1.0)


def test_decomposition_matches_monthly_tables(dec):
    total = pd.read_csv(P / "monthly_total.csv").set_index("month")["revenue"]
    for label, b0, b1, c0, c1 in cfg.COMPARISONS:
        row = dec[(dec["comparison"] == label) & (dec["market"] == "Total")].iloc[0]
        assert row["base_revenue"] == pytest.approx(total.loc[b0:b1].sum(), abs=1.0)
        assert row["current_revenue"] == pytest.approx(total.loc[c0:c1].sum(), abs=1.0)


def test_logmean_handles_equal_values():
    s = pd.Series([5.0, 2.0])
    lm = drivers._logmean(s, pd.Series([5.0, 8.0]))
    assert lm.iloc[0] == pytest.approx(5.0)
    assert 2.0 < lm.iloc[1] < 8.0


def test_baseline_identity_and_markets(tree):
    b = drivers.baseline(tree)
    assert b["market"].tolist() == cfg.MARKETS + ["Total"]
    rebuilt = b["customers"] * b["orders_per_customer"] * b["aov"]
    assert rebuilt.to_numpy() == pytest.approx(b["revenue"].to_numpy(), rel=1e-9)
    assert b.loc[b["market"] != "Total", "revenue_share"].sum() == pytest.approx(1.0)
