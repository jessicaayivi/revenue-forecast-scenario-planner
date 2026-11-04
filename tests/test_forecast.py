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


def test_all_eight_models_scored(cv):
    assert set(forecast.MODELS) <= set(cv.columns)
    assert len(forecast.MODELS) == 8
    assert cv[forecast.MODELS].notna().all().all()


def test_new_models_use_no_future_data():
    # forecasting from a cutoff must not change when later months are added to the data
    c = "2018-03"
    full = forecast.lag_regression_forecasts([c], 3, cfg.MODEL_END)
    cut = forecast.lag_regression_forecasts([c], 3, c)
    assert full["LagRegression"].to_numpy() == pytest.approx(cut["LagRegression"].to_numpy())
    a = forecast.forecast_all(end=c, h=2).set_index(["unique_id", "ds"])["Drivers"]
    cv = forecast.backtest(end=cfg.MODEL_END, h=2, n_windows=6)
    b = cv[cv["cutoff"] == pd.Timestamp(c + "-01")].set_index(["unique_id", "ds"])["Drivers"]
    assert b.loc[a.index].to_numpy() == pytest.approx(a.to_numpy(), rel=1e-6)


def test_forecast_check_is_out_of_sample(summary):
    chk = pd.read_csv(P / "forecast_check.csv").set_index("model")
    s = summary["check"]
    assert set(chk.index) == set(forecast.MODELS)
    assert all(m > cfg.CHECK_END and m <= cfg.MODEL_END for m in s["check_months"])
    # the model "picked in June" is the one with the lowest June-time backtest error
    assert s["selected_by_backtest"] == chk["backtest_avg_wape"].idxmin()
    assert chk["selected_in_check"].sum() == 1
    tot = pd.read_csv(P / "monthly_total.csv").set_index("month")["revenue"]
    assert s["check_actual_total"] == pytest.approx(tot.loc[s["check_months"]].sum(), abs=1)


def test_prophet_gets_no_extra_black_friday():
    assert "Prophet" in forecast.HAS_BLACK_FRIDAY


def test_summary_matches_forecast_file(fc, summary):
    t = fc[fc["unique_id"] == "Total"]
    assert summary["forecast_total"] == pytest.approx(t["forecast"].sum(), abs=1)
    assert summary["forecast_total_bf"] == pytest.approx(t["forecast_bf"].sum(), abs=1)
