"""Stage 3: baseline revenue forecast with a rolling-origin backtest.

Candidates are simple, explainable models (statsforecast). With 20 months of history there is
no full second year, so no model can learn seasonality; all are non-seasonal.

    Naive            last month repeated
    WindowAverage    average of the last 3 months, held flat
    RWD              random walk with drift (last value + average monthly change)
    AutoETS          exponential smoothing, trend chosen automatically
    AutoTheta        theta method

Selection: lowest average WAPE across the total and the 5 markets over the backtest
(4 origins x 6-month horizon). The plan is set per market, so market accuracy counts as much as the
total. The chosen model forecasts the total and each market. Market forecasts are also summed
(bottom-up) and compared with the direct total.

80% range: from the chosen model's own backtest errors. Per series, the 80th percentile of the
absolute percentage error over all backtest points (4 origins x 6 steps, pooled because 4 points
per step is too few) is applied either side of the point forecast. The range is symmetric on
purpose: the backtest months were mostly growth months, so its errors lean one way (forecasts too
low), and carrying that lean forward would assume growth the latest months no longer show.
Indicative only.

Black Friday: Nov 2017 was well above Oct and Dec. One November can't be learned, so the
forecast reports a `_bf` (Black Friday) column with an explicit, labelled uplift on Nov 2018.

Run: python -m src.forecast   (needs src.prepare outputs)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta, Naive, RandomWalkWithDrift, WindowAverage

from src import config as cfg

P = cfg.PROCESSED_DIR
MODELS = ["Naive", "WindowAverage", "RWD", "AutoETS", "AutoTheta"]


def make_models():
    return [Naive(), WindowAverage(window_size=3), RandomWalkWithDrift(),
            AutoETS(season_length=1), AutoTheta(season_length=1)]


def load_series() -> pd.DataFrame:
    """Long format for statsforecast: unique_id (market or 'Total'), ds (month start), y (revenue)."""
    tot = pd.read_csv(P / "monthly_total.csv")
    mk = pd.read_csv(P / "monthly_market.csv")
    tot = tot[tot["in_model_window"]].assign(unique_id="Total")
    mk = mk[mk["in_model_window"]].rename(columns={"market": "unique_id"})
    df = pd.concat([tot[["unique_id", "month", "revenue"]], mk[["unique_id", "month", "revenue"]]])
    df["ds"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return df.rename(columns={"revenue": "y"})[["unique_id", "ds", "y"]].reset_index(drop=True)


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    sf = StatsForecast(models=make_models(), freq="MS", n_jobs=1)
    cv = sf.cross_validation(df=df, h=cfg.FORECAST_H, step_size=1, n_windows=cfg.BACKTEST_WINDOWS)
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    cv["step"] = ((cv["ds"].dt.year - cv["cutoff"].dt.year) * 12 + cv["ds"].dt.month - cv["cutoff"].dt.month)
    return cv


def score(cv: pd.DataFrame) -> pd.DataFrame:
    """WAPE (sum |error| / sum actual) and bias (sum error / sum actual) per series and model."""
    rows = []
    for uid, g in cv.groupby("unique_id"):
        for m in MODELS:
            err = g[m] - g["y"]
            rows.append({"unique_id": uid, "model": m,
                         "wape": err.abs().sum() / g["y"].sum(),
                         "bias": err.sum() / g["y"].sum(),
                         "mape": (err.abs() / g["y"]).mean()})
    return pd.DataFrame(rows)


def bottom_up_score(cv: pd.DataFrame, model: str) -> dict:
    mk = cv[cv["unique_id"].isin(cfg.MARKETS)].groupby(["cutoff", "ds"])[[model, "y"]].sum()
    err = mk[model] - mk["y"]
    return {"wape": float(err.abs().sum() / mk["y"].sum()), "bias": float(err.sum() / mk["y"].sum())}


def interval_ratios(cv: pd.DataFrame, model: str, level: float = cfg.INTERVAL_LEVEL) -> pd.DataFrame:
    """Per series: the `level` quantile of |actual / forecast - 1|, pooled over all backtest points."""
    ape = cv.assign(ape=(cv["y"] / cv[model] - 1).abs()).groupby("unique_id")["ape"].quantile(level)
    return pd.DataFrame({"lo_ratio": 1 - ape, "hi_ratio": 1 + ape}).reset_index()


def black_friday_uplift() -> float:
    t = pd.read_csv(P / "monthly_total.csv").set_index("month")["revenue"]
    return float(t["2017-11"] / t[["2017-10", "2017-12"]].mean())


def forecast(df: pd.DataFrame, model: str, ratios: pd.DataFrame, uplift: float) -> pd.DataFrame:
    sf = StatsForecast(models=make_models(), freq="MS", n_jobs=1)
    fc = sf.forecast(df=df, h=cfg.FORECAST_H)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    fc = fc[["unique_id", "ds", model]].rename(columns={model: "forecast"})
    fc["step"] = fc.groupby("unique_id").cumcount() + 1
    fc = fc.merge(ratios, on="unique_id", how="left")
    fc["lo80"] = fc["forecast"] * fc["lo_ratio"]
    fc["hi80"] = fc["forecast"] * fc["hi_ratio"]
    bf = np.where(fc["ds"].dt.month == cfg.BLACK_FRIDAY_MONTH, uplift, 1.0)
    for c in ("forecast", "lo80", "hi80"):
        fc[f"{c}_bf"] = fc[c] * bf
    fc["model"] = model
    fc["month"] = fc["ds"].dt.strftime("%Y-%m")
    return fc.drop(columns=["ds", "lo_ratio", "hi_ratio"])


def main() -> dict:
    df = load_series()
    cv = backtest(df)
    scores = score(cv)
    avg = scores.groupby("model")["wape"].mean().sort_values()
    best = avg.index[0]
    tot = scores[scores["unique_id"] == "Total"].sort_values("wape")
    ratios = interval_ratios(cv, best)
    uplift = black_friday_uplift()
    fc = forecast(df, best, ratios, uplift)

    cv.assign(cutoff=cv["cutoff"].dt.strftime("%Y-%m"), month=cv["ds"].dt.strftime("%Y-%m")).drop(
        columns="ds").round(2).to_csv(P / "backtest_predictions.csv", index=False)
    scores.assign(selected=scores["model"] == best).round(4).to_csv(P / "backtest_scores.csv", index=False)
    cols = ["unique_id", "month", "step", "model", "forecast", "lo80", "hi80",
            "forecast_bf", "lo80_bf", "hi80_bf"]
    fc[cols].round(2).to_csv(P / "forecast.csv", index=False)

    t = fc[fc["unique_id"] == "Total"]
    mk_sum = fc[fc["unique_id"].isin(cfg.MARKETS)]["forecast"].sum()
    summary = {
        "selected_model": best,
        "backtest_avg_wape": {m: round(float(v), 4) for m, v in avg.items()},
        "backtest_total_wape": {r["model"]: round(float(r["wape"]), 4) for _, r in tot.iterrows()},
        "backtest_total_bias": {r["model"]: round(float(r["bias"]), 4) for _, r in tot.iterrows()},
        "bottom_up": {k: round(v, 4) for k, v in bottom_up_score(cv, best).items()},
        "black_friday_uplift": round(uplift, 4),
        "forecast_months": [t["month"].iloc[0], t["month"].iloc[-1]],
        "forecast_total": round(float(t["forecast"].sum()), 0),
        "forecast_total_bf": round(float(t["forecast_bf"].sum()), 0),
        "forecast_total_lo80": round(float(t["lo80"].sum()), 0),
        "forecast_total_hi80": round(float(t["hi80"].sum()), 0),
        "forecast_markets_sum": round(float(mk_sum), 0),
    }
    (P / "forecast_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
