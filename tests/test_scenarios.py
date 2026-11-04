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
    n = cfg.FORECAST_H
    sp = p.loc[("SP", n)]
    assert sp["customers"] == pytest.approx(base.loc["SP", "customers"] * 1.02 ** n)
    assert sp["aov"] == pytest.approx(base.loc["SP", "aov"] * 0.99 ** n)


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


def test_hypothesis_only_touches_chosen_markets_and_months(presets):
    proj = scenarios.project(presets["Base"])
    hyp = scenarios.apply_hypothesis(proj, "customers", 0.10, ["SP"], "2018-10")
    hit = (hyp["market"] == "SP") & (hyp["month"] >= "2018-10")
    ratio = hyp["revenue"] / proj["revenue"]
    assert ratio[hit].to_numpy() == pytest.approx(1.10)
    assert ratio[~hit].to_numpy() == pytest.approx(1.0)
    assert (hyp.loc[~hit, "customers"] == proj.loc[~hit, "customers"]).all()


@pytest.mark.parametrize("lever", list(scenarios.LEVERS))
def test_hypothesis_keeps_identity_and_black_friday(presets, lever):
    proj = scenarios.project(presets["Upside"], black_friday=True)
    hyp = scenarios.apply_hypothesis(proj, lever, -0.2, cfg.MARKETS, "2018-09")
    rebuilt = hyp["customers"] * hyp["orders_per_customer"] * hyp["aov"] * hyp["black_friday_factor"]
    assert rebuilt.to_numpy() == pytest.approx(hyp["revenue"].to_numpy())
    assert hyp["revenue"].sum() == pytest.approx(proj["revenue"].sum() * 0.8)


def test_hypothesis_impact_adds_up(presets):
    proj = scenarios.project(presets["Base"])
    imp = scenarios.hypothesis_impact(
        proj, scenarios.apply_hypothesis(proj, "aov", 0.05, ["South", "RJ"], "2018-12")).set_index("market")
    assert imp.drop(index="Total")["change"].sum() == pytest.approx(imp.loc["Total", "change"])
    assert imp.loc["Total", "plan_q1_change"] == pytest.approx(0.0)  # Dec is in Plan Q2
    assert imp.loc[["SP", "MG", "Rest of Brazil"], "change"].abs().max() == pytest.approx(0.0)


def test_hypothesis_by_market_ranks_by_size(presets):
    proj = scenarios.project(presets["Base"])
    each = scenarios.hypothesis_by_market(proj, "customers", 0.10, "2018-09").set_index("market")
    rev = proj.groupby("market")["revenue"].sum()
    assert each["extra_revenue"].to_numpy() == pytest.approx((rev.loc[each.index] * 0.10).to_numpy())


def test_aov_splits_into_basket_and_price(presets):
    base = scenarios.load_baseline()
    assert (base["items_per_order"] * base["price_per_item"]).to_numpy() == pytest.approx(base["aov"].to_numpy(), rel=1e-4)
    p = scenarios.project(presets["Upside"])
    assert (p["items_per_order"] * p["price_per_item"]).to_numpy() == pytest.approx(p["aov"].to_numpy())
    assert (p.groupby("market")["items_per_order"].nunique() == 1).all()  # basket size held


@pytest.mark.parametrize("lever", ["items_per_order", "price_per_item", "aov"])
def test_basket_and_price_levers_move_aov(presets, lever):
    proj = scenarios.project(presets["Base"])
    hyp = scenarios.apply_hypothesis(proj, lever, 0.10, ["MG"], "2018-09")
    mg = hyp["market"] == "MG"
    assert (hyp.loc[mg, "aov"] / proj.loc[mg, "aov"]).to_numpy() == pytest.approx(1.10)
    assert (hyp.loc[mg, "revenue"] / proj.loc[mg, "revenue"]).to_numpy() == pytest.approx(1.10)
    other = "price_per_item" if lever == "items_per_order" else "items_per_order"
    assert (hyp[other] == proj[other]).all()


@pytest.mark.parametrize("price,e,expected", [(0.05, 0.0, 0.05), (0.05, 1.0, 0.0), (0.10, 2.0, 1.1 ** -1 - 1),
                                              (-0.10, 1.5, 0.9 ** -0.5 - 1)])
def test_price_volume_revenue(presets, price, e, expected):
    proj = scenarios.project(presets["Base"])
    hyp = scenarios.price_volume(proj, price, e, cfg.MARKETS, "2018-09")
    assert hyp["revenue"].sum() / proj["revenue"].sum() - 1 == pytest.approx(expected)
    assert (hyp["customers"] / proj["customers"]).to_numpy() == pytest.approx((1 + price) ** -e)


def test_break_even_and_curve(presets):
    assert scenarios.break_even_customer_change(0.05) == pytest.approx(1 / 1.05 - 1)
    proj = scenarios.project(presets["Base"])
    c = scenarios.price_volume_curve(proj, 0.05, ["SP"], "2018-09").set_index("elasticity")
    assert c.loc[1.0, "revenue_change"] == pytest.approx(0.0, abs=1e-6)
    assert c.loc[0.5, "revenue_change"] > 0 > c.loc[1.5, "revenue_change"]
    assert c["revenue_change"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        scenarios.price_volume(proj, 0.05, -1.0, ["SP"], "2018-09")


def test_unknown_lever_rejected(presets):
    with pytest.raises(ValueError):
        scenarios.apply_hypothesis(scenarios.project(presets["Base"]), "price", 0.1, ["SP"], "2018-09")


def test_plan_is_half_year_of_actuals_plus_forecast(presets):
    assert cfg.PLAN_NAME == "H2 2018" and cfg.ACTUAL_MONTHS == ["2018-07", "2018-08"]
    assert scenarios.forecast_months() == ["2018-09", "2018-10", "2018-11", "2018-12"]
    s = scenarios.summarise(scenarios.project(presets["Base"])).set_index("market")
    mk = pd.read_csv(P / "monthly_market.csv")
    act = mk[mk["month"].isin(cfg.ACTUAL_MONTHS)].groupby("market")["revenue"].sum()
    assert s.loc[cfg.MARKETS, "actual"].to_numpy() == pytest.approx(act.loc[cfg.MARKETS].to_numpy())
    assert (s["actual"] + s["forecast"]).to_numpy() == pytest.approx(s["six_months"].to_numpy())
    assert (s["plan_q1"] + s["plan_q2"]).to_numpy() == pytest.approx(s["six_months"].to_numpy())
    # Q3 = Jul + Aug actual + Sep forecast
    p = scenarios.project(presets["Base"])
    sep = p[p["month"] == "2018-09"].groupby("market")["revenue"].sum()
    assert s.loc[cfg.MARKETS, "plan_q1"].to_numpy() == pytest.approx((act + sep).loc[cfg.MARKETS].to_numpy())


def test_hypotheses_never_touch_actual_months(presets):
    proj = scenarios.project(presets["Base"])
    hyp = scenarios.apply_hypothesis(proj, "customers", 0.5, cfg.MARKETS, scenarios.forecast_months()[0])
    a, b = scenarios.summarise(proj), scenarios.summarise(hyp)
    assert a["actual"].to_numpy() == pytest.approx(b["actual"].to_numpy())


def test_category_scorecard_shares_add_up():
    c = scenarios.category_scorecard()
    assert len(c) == cfg.TOP_N_CATEGORIES + 1
    assert c["revenue_share_last6m"].sum() == pytest.approx(1.0)
