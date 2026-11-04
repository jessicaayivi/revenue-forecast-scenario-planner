"""Stage 3: revenue forecast with a rolling-origin backtest, plus a check against real later months.

Candidates. Five simple, explainable models look at revenue alone (statsforecast). With 20 months of
history there is no full second year, so none can learn seasonality; all are non-seasonal.

    Naive            last month repeated
    WindowAverage    average of the last 3 months, held flat
    RWD              random walk with drift (last value + average monthly change)
    AutoETS          exponential smoothing, trend chosen automatically
    AutoTheta        theta method

A sixth, Drivers, uses the KPIs instead of revenue: per market it forecasts monthly customers with
AutoETS (customers are the driver that turns first; Stage 2 found they explain almost all growth),
holds orders per customer and AOV at their 3-month averages (both barely trend), and multiplies:
revenue = customers x orders per customer x AOV. The total is the sum of the markets. This design was
fixed before looking at how any model did on the check months.

Selection: lowest average WAPE across the total and the 5 markets over the backtest (4 origins, same
horizon as the forecast). The plan is set per market, so market accuracy counts as much as the total.

80% range: from the chosen model's own backtest errors. Per series, the 80th percentile of the absolute
percentage error over all backtest points is applied either side of the point forecast. Symmetric on
purpose: the backtest months were mostly growth months, so its errors lean one way. Indicative only.

Black Friday: one November in the data, so the forecast reports a `_bf` column with an explicit,
labelled uplift on November instead of a learned effect.

Forecast check: the same models, run as if it were CHECK_END (June 2018), scored on the months after it
that we already know (Jul-Aug 2018). This shows whether the backtest would have picked a model that
works on genuinely unseen months.

Run: python -m src.forecast   (needs src.prepare outputs)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta, Naive, RandomWalkWithDrift, WindowAverage

from src import config as cfg

for _name in ("cmdstanpy", "prophet"):
    logging.getLogger(_name).setLevel(logging.WARNING)

P = cfg.PROCESSED_DIR
REVENUE_MODELS = ["Naive", "WindowAverage", "RWD", "AutoETS", "AutoTheta"]
MODELS = REVENUE_MODELS + ["Drivers", "Prophet", "LagRegression"]
LABELS = {"Naive": "Last month repeated", "WindowAverage": "3-month average", "RWD": "Random walk with drift",
          "AutoETS": "AutoETS (exponential smoothing)", "AutoTheta": "AutoTheta",
          "Drivers": "Driver-based (customers x orders/customer x AOV)",
          "Prophet": "Prophet (daily, weekly + yearly seasonality, Black Friday)",
          "LagRegression": "Regression on revenue + 3-month customer momentum"}
# Models whose forecast already contains a Black Friday effect (no extra uplift added).
HAS_BLACK_FRIDAY = {"Prophet"}
BLACK_FRIDAYS = ["2016-11-25", "2017-11-24", "2018-11-23", "2019-11-29"]


# ------------------------------------------------------------------ Prophet (daily data)
def _prophet_months(daily: pd.DataFrame, end: str, h: int) -> pd.Series:
    """Fit Prophet on daily revenue up to the end of month `end`; return monthly sums for the next h months.

    Settings fixed in advance: multiplicative weekly + yearly (Fourier order 3) seasonality and a Black
    Friday holiday (the Friday to the Monday after). Yearly seasonality is forced on: with under two years
    Prophet would switch it off, and testing whether it helps is the point.
    """
    from prophet import Prophet

    # Prophet warns on every fit that yearly seasonality has < 2 years of data; that limit is the point
    # of the test and is documented, so keep the logs readable.
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
    last = pd.Period(end, "M")
    train = daily[daily["ds"] <= last.end_time]
    hol = pd.DataFrame({"holiday": "black_friday", "ds": pd.to_datetime(BLACK_FRIDAYS),
                        "lower_window": 0, "upper_window": 3})
    m = Prophet(seasonality_mode="multiplicative", weekly_seasonality=True, yearly_seasonality=3,
                daily_seasonality=False, holidays=hol)
    m.fit(train[["ds", "y"]])
    fut = pd.DataFrame({"ds": pd.date_range((last + 1).start_time, (last + h).end_time.normalize(), freq="D")})
    pred = m.predict(fut)[["ds", "yhat"]]
    pred["yhat"] = pred["yhat"].clip(lower=0)
    return pred.groupby(pred["ds"].dt.to_period("M"))["yhat"].sum()


def _daily_series() -> dict[str, pd.DataFrame]:
    d = pd.read_csv(P / "daily_market.csv", parse_dates=["date"]).rename(columns={"date": "ds", "revenue": "y"})
    out = {m: g[["ds", "y"]].reset_index(drop=True) for m, g in d.groupby("market")}
    out["Total"] = d.groupby("ds", as_index=False)["y"].sum()
    return out


def prophet_forecasts(cutoffs: list[str], h: int) -> pd.DataFrame:
    """Prophet monthly forecasts per series for each cutoff month: unique_id, cutoff, ds, Prophet."""
    rows = []
    for uid, daily in _daily_series().items():
        for c in cutoffs:
            for per, v in _prophet_months(daily, c, h).items():
                rows.append({"unique_id": uid, "cutoff": pd.Period(c, "M").to_timestamp(),
                             "ds": per.to_timestamp(), "Prophet": v})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ lagged-KPI regression
def _monthly_kpis(end: str) -> pd.DataFrame:
    """Monthly revenue and customers per series (markets + Total), wide by unique_id."""
    mk = _window(pd.read_csv(P / "monthly_market.csv"), end)
    tot = mk.groupby("month", as_index=False)[["revenue", "customers"]].sum().assign(market="Total")
    return pd.concat([mk[["month", "market", "revenue", "customers"]], tot])


def lag_regression_forecasts(cutoffs: list[str], h: int, end: str) -> pd.DataFrame:
    """Direct regression per series and horizon k: y[t+k] = a + b*y[t] + c*customer_momentum[t].

    customer_momentum[t] = customers[t] / customers[t-3] - 1. Only information known at the forecast
    month is used, so the model can be run for real. Fitted by least squares on all rows up to the cutoff.
    """
    k_all = _monthly_kpis(end)
    rows = []
    for uid, g in k_all.groupby("market"):
        g = g.sort_values("month").reset_index(drop=True)
        g["mom"] = g["customers"] / g["customers"].shift(3) - 1
        for c in cutoffs:
            hist = g[g["month"] <= c].reset_index(drop=True)
            x_now = hist.iloc[-1]
            for k in range(1, h + 1):
                t = hist.iloc[:len(hist) - k].dropna(subset=["mom"])
                if len(t) < 4:  # too few rows to fit 3 coefficients: fall back to the last value
                    pred = x_now["revenue"]
                else:
                    X = np.column_stack([np.ones(len(t)), t["revenue"], t["mom"]])
                    y = hist["revenue"].shift(-k).loc[t.index].to_numpy()
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    pred = coef @ np.array([1.0, x_now["revenue"], x_now["mom"]])
                month = pd.Period(c, "M") + k
                rows.append({"unique_id": uid, "cutoff": pd.Period(c, "M").to_timestamp(),
                             "ds": month.to_timestamp(), "LagRegression": max(float(pred), 0.0)})
    return pd.DataFrame(rows)


def _extra_models(cutoffs: list[str], h: int, end: str) -> pd.DataFrame:
    p = prophet_forecasts(cutoffs, h)
    r = lag_regression_forecasts(cutoffs, h, end)
    return p.merge(r, on=["unique_id", "cutoff", "ds"], how="outer")


def make_models():
    return [Naive(), WindowAverage(window_size=3), RandomWalkWithDrift(),
            AutoETS(season_length=1), AutoTheta(season_length=1)]


def _window(df: pd.DataFrame, end: str) -> pd.DataFrame:
    return df[(df["month"] >= cfg.MODEL_START) & (df["month"] <= end)]


def _ds(month: pd.Series) -> pd.Series:
    return pd.PeriodIndex(month, freq="M").to_timestamp()


def _flat(out: pd.DataFrame) -> pd.DataFrame:
    return out.reset_index() if "unique_id" not in out.columns else out


def load_series(end: str = cfg.MODEL_END) -> pd.DataFrame:
    """Revenue in long format: unique_id (market or 'Total'), ds (month start), y."""
    tot = _window(pd.read_csv(P / "monthly_total.csv"), end).assign(unique_id="Total")
    mk = _window(pd.read_csv(P / "monthly_market.csv"), end).rename(columns={"market": "unique_id"})
    df = pd.concat([tot[["unique_id", "month", "revenue"]], mk[["unique_id", "month", "revenue"]]])
    df["ds"] = _ds(df["month"])
    return df.rename(columns={"revenue": "y"})[["unique_id", "ds", "y"]].reset_index(drop=True)


def load_drivers(end: str = cfg.MODEL_END) -> pd.DataFrame:
    """Driver series per market: unique_id = 'market|customers', 'market|opc' or 'market|aov'."""
    mk = _window(pd.read_csv(P / "monthly_market.csv"), end)
    vals = {"customers": mk["customers"], "opc": mk["orders"] / mk["customers"], "aov": mk["revenue"] / mk["orders"]}
    parts = [pd.DataFrame({"unique_id": mk["market"] + "|" + k, "ds": _ds(mk["month"]), "y": v.to_numpy()})
             for k, v in vals.items()]
    return pd.concat(parts, ignore_index=True)


def _driver_models():
    # customers: AutoETS; orders per customer and AOV: 3-month average
    return (StatsForecast(models=[AutoETS(season_length=1)], freq="MS", n_jobs=1),
            StatsForecast(models=[WindowAverage(window_size=3)], freq="MS", n_jobs=1))


def _combine_drivers(c: pd.DataFrame, r: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Customers forecast (column AutoETS) x opc x aov forecasts (column WindowAverage) -> revenue."""
    c = c.assign(market=c["unique_id"].str.split("|").str[0]).rename(columns={"AutoETS": "customers"})
    r = r.assign(market=r["unique_id"].str.split("|").str[0], driver=r["unique_id"].str.split("|").str[1])
    r = r.pivot_table(index=["market"] + keys, columns="driver", values="WindowAverage").reset_index()
    d = c[["market"] + keys + ["customers"]].merge(r, on=["market"] + keys)
    d["Drivers"] = d["customers"] * d["opc"] * d["aov"]
    tot = d.groupby(keys, as_index=False)["Drivers"].sum().assign(market="Total")
    return pd.concat([d[["market"] + keys + ["Drivers"]], tot]).rename(columns={"market": "unique_id"})


def backtest(end: str = cfg.MODEL_END, h: int = cfg.FORECAST_H, n_windows: int = cfg.BACKTEST_WINDOWS) -> pd.DataFrame:
    df = load_series(end)
    sf = StatsForecast(models=make_models(), freq="MS", n_jobs=1)
    cv = _flat(sf.cross_validation(df=df, h=h, step_size=1, n_windows=n_windows))
    drv = load_drivers(end)
    sc, sr = _driver_models()
    is_c = drv["unique_id"].str.endswith("|customers")
    c = _flat(sc.cross_validation(df=drv[is_c], h=h, step_size=1, n_windows=n_windows))
    r = _flat(sr.cross_validation(df=drv[~is_c], h=h, step_size=1, n_windows=n_windows))
    cv = cv.merge(_combine_drivers(c, r, ["cutoff", "ds"]), on=["unique_id", "cutoff", "ds"], how="left")
    cutoffs = sorted(cv["cutoff"].dt.strftime("%Y-%m").unique())
    cv = cv.merge(_extra_models(cutoffs, h, end), on=["unique_id", "cutoff", "ds"], how="left")
    cv["step"] = ((cv["ds"].dt.year - cv["cutoff"].dt.year) * 12 + cv["ds"].dt.month - cv["cutoff"].dt.month)
    return cv


def forecast_all(end: str = cfg.MODEL_END, h: int = cfg.FORECAST_H) -> pd.DataFrame:
    """Point forecasts of every candidate model, wide: unique_id, ds, one column per model."""
    sf = StatsForecast(models=make_models(), freq="MS", n_jobs=1)
    fc = _flat(sf.forecast(df=load_series(end), h=h))
    drv = load_drivers(end)
    sc, sr = _driver_models()
    is_c = drv["unique_id"].str.endswith("|customers")
    c = _flat(sc.forecast(df=drv[is_c], h=h))
    r = _flat(sr.forecast(df=drv[~is_c], h=h))
    fc = fc.merge(_combine_drivers(c, r, ["ds"]), on=["unique_id", "ds"], how="left")
    extra = _extra_models([end], h, end).drop(columns="cutoff")
    return fc.merge(extra, on=["unique_id", "ds"], how="left")


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


def forecast(model: str, ratios: pd.DataFrame, uplift: float) -> pd.DataFrame:
    fc = forecast_all()[["unique_id", "ds", model]].rename(columns={model: "forecast"})
    fc["step"] = fc.groupby("unique_id").cumcount() + 1
    fc = fc.merge(ratios, on="unique_id", how="left")
    fc["lo80"] = fc["forecast"] * fc["lo_ratio"]
    fc["hi80"] = fc["forecast"] * fc["hi_ratio"]
    # models that already learn Black Friday (Prophet) get no extra uplift
    bf_factor = 1.0 if model in HAS_BLACK_FRIDAY else uplift
    bf = np.where(fc["ds"].dt.month == cfg.BLACK_FRIDAY_MONTH, bf_factor, 1.0)
    for c in ("forecast", "lo80", "hi80"):
        fc[f"{c}_bf"] = fc[c] * bf
    fc["model"] = model
    fc["month"] = fc["ds"].dt.strftime("%Y-%m")
    return fc.drop(columns=["ds", "lo_ratio", "hi_ratio"])


def forecast_check() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Plan as of CHECK_END with every model; score on the known months after it.

    Returns (scores per model, monthly Total paths per model with actuals, summary).
    """
    cv = backtest(end=cfg.CHECK_END, h=cfg.CHECK_H)
    bt = score(cv).groupby("model")["wape"].mean()
    fc = forecast_all(end=cfg.CHECK_END, h=cfg.CHECK_H)
    fc["month"] = fc["ds"].dt.strftime("%Y-%m")
    act = load_series(cfg.MODEL_END).assign(month=lambda d: d["ds"].dt.strftime("%Y-%m"))
    known = fc.merge(act[["unique_id", "month", "y"]], on=["unique_id", "month"], how="inner")
    rows = []
    for m in MODELS:
        tot = known[known["unique_id"] == "Total"]
        wapes = [(g[m] - g["y"]).abs().sum() / g["y"].sum() for _, g in known.groupby("unique_id")]
        rows.append({"model": m, "label": LABELS[m], "backtest_avg_wape": bt[m],
                     "check_total_error": tot[m].sum() / tot["y"].sum() - 1,
                     "check_avg_wape": float(np.mean(wapes))})
    sc = pd.DataFrame(rows)
    picked = sc.loc[sc["backtest_avg_wape"].idxmin(), "model"]
    sc["selected_in_check"] = sc["model"] == picked
    paths = fc[fc["unique_id"] == "Total"][["month"] + MODELS].merge(
        act[act["unique_id"] == "Total"][["month", "y"]].rename(columns={"y": "actual"}), on="month", how="left")
    summary = {
        "planning_month": cfg.CHECK_END, "check_months": sorted(known["month"].unique().tolist()),
        "selected_by_backtest": picked,
        "selected_check_total_error": round(float(sc.set_index("model").loc[picked, "check_total_error"]), 4),
        "best_on_check": sc.loc[sc["check_avg_wape"].idxmin(), "model"],
        "check_actual_total": round(float(known[known["unique_id"] == "Total"]["y"].sum()), 2),
    }
    return sc, paths, summary


def main() -> dict:
    cv = backtest()
    scores = score(cv)
    avg = scores.groupby("model")["wape"].mean().sort_values()
    best = avg.index[0]
    tot = scores[scores["unique_id"] == "Total"].sort_values("wape")
    ratios = interval_ratios(cv, best)
    uplift = black_friday_uplift()
    fc = forecast(best, ratios, uplift)

    cv.assign(cutoff=cv["cutoff"].dt.strftime("%Y-%m"), month=cv["ds"].dt.strftime("%Y-%m")).drop(
        columns="ds").round(2).to_csv(P / "backtest_predictions.csv", index=False)
    scores.assign(selected=scores["model"] == best).round(4).to_csv(P / "backtest_scores.csv", index=False)
    cols = ["unique_id", "month", "step", "model", "forecast", "lo80", "hi80",
            "forecast_bf", "lo80_bf", "hi80_bf"]
    fc[cols].round(2).to_csv(P / "forecast.csv", index=False)

    chk, paths, chk_summary = forecast_check()
    chk.round(4).to_csv(P / "forecast_check.csv", index=False)
    paths.round(2).to_csv(P / "forecast_check_paths.csv", index=False)

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
        "check": chk_summary,
    }
    (P / "forecast_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
