"""Scenario engine tests: run `pytest` after `python -m src.run_all`."""
import json

import pandas as pd
import pytest

from src import config as cfg, scenarios

P = cfg.PROCESSED_DIR


@pytest.fixture(scope="module")
def presets():
    return scenarios.preset_scenarios()


def test_base_scenario_equals_stage3_forecast(presets):
    fc = pd.read_csv(P / "forecast.csv")
    fc = fc[fc["unique_id"].isin(cfg.MARKETS)].set_index(["unique_id", "month"])
    for bf, col in ((False, "forecast"), (True, "forecast_bf")):
        p = scenarios.project(presets["Base"], black_friday=bf).set_index(["market", "month"])
        assert p["revenue"].to_numpy() == pytest.approx(fc.loc[p.index, col].to_numpy(), rel=1e-4)


def test_projection_identity(presets):
    p = scenarios.project(presets["Upside"], black_friday=False)
    rebuilt = p["customers"] * p["orders_per_customer"] * p["aov"]
    assert rebuilt.to_numpy() == pytest.approx(p["revenue"].to_numpy(), rel=1e-12)


def test_growth_compounds_monthly():
    base = scenarios.load_baseline()
    params = pd.DataFrame({"g_customers": 0.02, "g_aov": -0.01}, index=cfg.MARKETS)
    p = scenarios.project(params, black_friday=False, base=base).set_index(["market", "step"])
    sp6 = p.loc[("SP", 6)]
    assert sp6["customers"] == pytest.approx(base.loc["SP", "customers"] * 1.02 ** 6)
    assert sp6["aov"] == pytest.approx(base.loc["SP", "aov"] * 0.99 ** 6)


def test_black_friday_only_in_november(presets):
    on = scenarios.project(presets["Upside"], black_friday=True)
    off = scenarios.project(presets["Upside"], black_friday=False)
    ratio = on["revenue"] / off["revenue"]
    nov = on["month"].str.endswith(f"-{cfg.BLACK_FRIDAY_MONTH:02d}")
    assert ratio[~nov].to_numpy() == pytest.approx(1.0)
    assert ratio[nov].to_numpy() == pytest.approx(scenarios.black_friday_uplift())


def test_monthly_rate_round_trips():
    assert (1 + scenarios.monthly_rate(0.125, 6)) ** 6 == pytest.approx(1.125)
    assert scenarios.monthly_rate(0.0, 3) == 0.0


def test_presets_are_ordered(presets):
    tot = {n: scenarios.project(presets[n])["revenue"].sum() for n in cfg.SCENARIOS}
    assert tot["Downside"] < tot["Base"] < tot["Upside"]


def test_presets_come_from_observed_growth(presets):
    dec = pd.read_csv(P / "driver_decomposition.csv")
    l6 = dec[dec["comparison"] == cfg.COMPARISONS[1][0]].set_index("market")
    up = presets["Upside"]
    assert (1 + up.loc["SP", "g_customers"]) ** 6 - 1 == pytest.approx(l6.loc["SP", "customers_growth"])
    base = scenarios.load_baseline()
    dn = presets["Downside"]
    assert (1 + dn.loc["RJ", "g_customers"]) ** 3 - 1 == pytest.approx(base.loc["RJ", "customers_growth_vs_prior"])


def test_summary_totals_add_up():
    s = pd.read_csv(P / "scenario_summary.csv")
    for _, g in s.groupby("scenario"):
        mk, tot = g[g["market"] != "Total"], g[g["market"] == "Total"].iloc[0]
        assert mk["six_months"].sum() == pytest.approx(tot["six_months"], abs=0.1)
        assert tot["plan_q1"] + tot["plan_q2"] == pytest.approx(tot["six_months"], abs=0.1)


def test_sensitivity_scales_with_market_size():
    sens = scenarios.sensitivity().set_index("market")["extra_revenue_customers"]
    rev = scenarios.load_baseline()["revenue"]
    assert (sens > 0).all()
    assert sens.idxmax() == rev.idxmax()


def test_category_scorecard_shares_add_up():
    c = scenarios.category_scorecard()
    assert len(c) == cfg.TOP_N_CATEGORIES + 1
    assert c["revenue_share_last6m"].sum() == pytest.approx(1.0)
