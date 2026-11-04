"""Smoke test: the Streamlit app runs, and presets drive the headline number."""
import pytest
from streamlit.testing.v1 import AppTest

from src import config as cfg, scenarios

APP = str(cfg.ROOT / "app" / "streamlit_app.py")


HEADLINE = f"{cfg.PLAN_NAME} revenue"
IMPACT = f"Revenue impact, {cfg.PLAN_NAME}"


def six_month_value(at):
    m = next(x for x in at.metric if x.label == HEADLINE)
    return float(m.value.replace("R$ ", "").replace("M", ""))


def plan_total(name):
    """Plan half-year total (actual months + forecast) for a preset, in R$M."""
    s = scenarios.summarise(scenarios.project(scenarios.preset_scenarios()[name])).set_index("market")
    return s.loc["Total", "six_months"] / 1e6


def test_app_runs_and_presets_change_total():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    expected = {}
    for name in cfg.SCENARIOS:
        expected[name] = plan_total(name)
        at.selectbox(key="preset").set_value(name).run()
        assert not at.exception
        assert six_month_value(at) == pytest.approx(expected[name], abs=0.006)
    assert expected["Downside"] < expected["Base"] < expected["Upside"]


def impact_value(at):
    m = next(x for x in at.metric if x.label == IMPACT)
    return float(m.value.replace("R$ ", "").replace("k", "").replace(",", ""))


def test_what_if_tab_matches_engine():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="wi_lever").set_value("customers").run()
    at.slider(key="wi_change").set_value(10).run()
    at.multiselect(key="wi_markets").set_value(["SP"]).run()
    at.selectbox(key="wi_start").set_value("2018-10").run()
    assert not at.exception
    proj = scenarios.project(scenarios.preset_scenarios()["Base"])
    hyp = scenarios.apply_hypothesis(proj, "customers", 0.10, ["SP"], "2018-10")
    expected = (hyp["revenue"].sum() - proj["revenue"].sum()) / 1e3
    assert impact_value(at) == pytest.approx(expected, abs=0.6)
    assert impact_value(at) > 0


def test_what_if_handles_no_market_and_negative_change():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.multiselect(key="wi_markets").set_value([]).run()
    assert not at.exception and any("at least one market" in w.value for w in at.warning)
    at.multiselect(key="wi_markets").set_value(["RJ", "MG"]).run()
    at.selectbox(key="wi_lever").set_value("orders_per_customer").run()
    at.slider(key="wi_change").set_value(-20).run()
    assert not at.exception and impact_value(at) < 0


def test_price_volume_mode():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="wi_mode").set_value("Price vs volume").run()
    assert not at.exception
    at.slider(key="wi_price").set_value(10).run()
    at.slider(key="wi_elasticity").set_value(0.5).run()
    assert not at.exception
    proj = scenarios.project(scenarios.preset_scenarios()["Base"])
    expected = proj["revenue"].sum() * ((1.1) ** 0.5 - 1) / 1e3
    assert impact_value(at) == pytest.approx(expected, abs=0.6)
    assert any("Break-even" in m.value for m in at.markdown)
    assert any(e.label.startswith("How to read this") for e in at.expander)
    assert any("below 1" in m.value for m in at.markdown)
    # the worked calculation uses the selected values and lands on the same impact as the engine
    calc = next(m.value for m in at.markdown if "Price factor" in m.value)
    assert "1.100" in calc and f"{1.1 ** -0.5:.3f}" in calc and f"{1.1 ** 0.5:.3f}" in calc
    hyp = scenarios.price_volume(proj, 0.10, 0.5, cfg.MARKETS, "2018-09")
    impact_k = (hyp["revenue"].sum() - proj["revenue"].sum()) / 1e3
    assert f"R\\$ {impact_k:+,.0f}k" in calc
    at.slider(key="wi_elasticity").set_value(1.0).run()
    assert abs(impact_value(at)) < 0.6


def test_basket_size_lever():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key="wi_lever").set_value("items_per_order").run()
    at.slider(key="wi_change").set_value(5).run()
    assert not at.exception
    assert any("items per order" in c.value for c in at.caption)
    total = scenarios.project(scenarios.preset_scenarios()["Base"])["revenue"].sum()
    assert impact_value(at) == pytest.approx(total * 0.05 / 1e3, abs=0.6)


def test_euro_view_converts_at_constant_rate():
    at = AppTest.from_file(APP, default_timeout=60).run()
    brl = six_month_value(at)
    at.radio(key="currency").set_value("€").run()
    assert not at.exception
    m = next(x for x in at.metric if x.label == HEADLINE)
    assert m.value.startswith("€ ")
    eur = float(m.value.replace("€ ", "").replace("M", ""))
    assert eur == pytest.approx(brl / cfg.BRL_PER_EUR, abs=0.006)
    # the what-if numbers switch too
    w = next(x for x in at.metric if x.label == IMPACT)
    assert w.value.startswith("€ ")


def test_start_here_explains_time_frame_and_markets():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs][0] == "Start here"
    text = " ".join(m.value for m in at.markdown)
    assert f"History: {cfg.HISTORY_LABEL}" in text and f"Plan: {cfg.PLAN_NAME} ({cfg.PLAN_LABEL})" in text
    assert "Forecast check" in [t.label for t in at.tabs]
    assert "How to use it" in text


def test_slider_moves_total():
    at = AppTest.from_file(APP, default_timeout=60).run()
    before = six_month_value(at)
    at.slider(key="gc_SP").set_value(3.0).run()
    assert not at.exception
    assert six_month_value(at) > before
