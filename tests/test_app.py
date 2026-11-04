"""Smoke test: the Streamlit app runs, and presets drive the headline number."""
import pytest
from streamlit.testing.v1 import AppTest

from src import config as cfg, scenarios

APP = str(cfg.ROOT / "app" / "streamlit_app.py")


def six_month_value(at):
    m = next(x for x in at.metric if x.label == "6-month revenue")
    return float(m.value.replace("R$ ", "").replace("M", ""))


def test_app_runs_and_presets_change_total():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    expected = {}
    for name in cfg.SCENARIOS:
        expected[name] = scenarios.project(scenarios.preset_scenarios()[name])["revenue"].sum() / 1e6
        at.selectbox(key="preset").set_value(name).run()
        assert not at.exception
        assert six_month_value(at) == pytest.approx(expected[name], abs=0.006)
    assert expected["Downside"] < expected["Base"] < expected["Upside"]


def test_slider_moves_total():
    at = AppTest.from_file(APP, default_timeout=60).run()
    before = six_month_value(at)
    at.slider(key="gc_SP").set_value(3.0).run()
    assert not at.exception
    assert six_month_value(at) > before
