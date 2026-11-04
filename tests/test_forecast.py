"""Forecast tests: run `pytest` after `python -m src.forecast` (src.run_all does it)."""
import json

import pandas as pd
import pytest

from src import config as cfg, forecast

P = cfg.PROCESSED_DIR
SERIES = cfg.MARKETS + ["Total"]


@pytest.fixture(scope="module")
def fc():
    return pd.read_csv(P / "forecast.csv")


@pytest.fixture(scope="module")
def cv():
    return pd.read_csv(P / "backtest_predictions.csv")


@pytest.fixture(scope="module")
def summary():
    return json.loads((P / "forecast_summary.json").read_text())


def test_forecast_covers_next_two_quarters(fc):
    after = pd.Period(cfg.MODEL_END, "M")
    expected = [str(after + i) for i in range(1, cfg.FORECAST_H + 1)]
    for uid in SERIES:
        assert fc.loc[fc["unique_id"] == uid, "month"].tolist() == expected


def test_markets_add_up_to_total(fc):
    mk = fc[fc["unique_id"].isin(cfg.MARKETS)].groupby("month")["forecast"].sum()
    tot = fc[fc["unique_id"] == "Total"].set_index("month")["forecast"]
    assert mk.sort_index().to_numpy() == pytest.approx(tot.sort_index().to_numpy(), rel=1e-6)


def test_range_contains_point_forecast(fc):
    assert (fc["lo80"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["hi80"]).all()
    assert (fc["lo80"] > 0).all()


def test_black_friday_only_changes_november(fc, summary):
    nov = fc["month"].str.endswith(f"-{cfg.BLACK_FRIDAY_MONTH:02d}")
    assert (fc.loc[~nov, "forecast_bf"] == fc.loc[~nov, "forecast"]).all()
    ratio = fc.loc[nov, "forecast_bf"] / fc.loc[nov, "forecast"]
    assert ratio.to_numpy() == pytest.approx(summary["black_friday_uplift"], rel=1e-3)


def test_black_friday_uplift_formula():
    t = pd.read_csv(P / "monthly_total.csv").set_index("month")["revenue"]
    expected = t["2017-11"] / ((t["2017-10"] + t["2017-12"]) / 2)
    assert forecast.black_friday_uplift() == pytest.approx(expected)


def test_backtest_has_no_leakage(cv):
    cut = pd.PeriodIndex(cv["cutoff"], freq="M")
    month = pd.PeriodIndex(cv["month"], freq="M")
    steps = (month - cut).map(lambda d: d.n)
    assert (steps >= 1).all() and (steps <= cfg.FORECAST_H).all()
    assert (steps == cv["step"]).all()
    assert cv["cutoff"].nunique() == cfg.BACKTEST_WINDOWS
    assert cv["month"].max() == cfg.MODEL_END


def test_backtest_actuals_match_history(cv):
    tot = pd.read_csv(P / "monthly_total.csv").set_index("month")["revenue"]
    t = cv[cv["unique_id"] == "Total"]
    assert t["y"].to_numpy() == pytest.approx(t["month"].map(tot).to_numpy(), abs=0.01)


def test_selected_model_has_lowest_average_wape(summary):
    s = pd.read_csv(P / "backtest_scores.csv")
    avg = s.groupby("model")["wape"].mean()
    assert summary["selected_model"] == avg.idxmin()
    assert s.loc[s["selected"], "model"].unique().tolist() == [summary["selected_model"]]


def test_summary_matches_forecast_file(fc, summary):
    t = fc[fc["unique_id"] == "Total"]
    assert summary["forecast_total"] == pytest.approx(t["forecast"].sum(), abs=1)
    assert summary["forecast_total_bf"] == pytest.approx(t["forecast_bf"].sum(), abs=1)
