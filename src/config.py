"""Project-wide paths and constants. All paths are relative to the repo root.

Every date in the project (labels, comparison periods, baseline, plan, holdout) is derived from
MODEL_START and MODEL_END below, so changing the plan period is a one-line edit.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = ROOT / "docs" / "images"

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

# Only delivered orders count as revenue. Other statuses are reported, then excluded.
KEEP_STATUS = "delivered"

# Modelling window: full months with stable volume (2016 and Sep-Oct 2018 are sparse).
MODEL_START = "2017-01"
MODEL_END = "2018-08"
DATA_END = "2018-08"  # last month with delivered orders


def _p(m) -> pd.Period:
    return pd.Period(m, "M")


def month_label(m) -> str:
    return _p(m).strftime("%b %Y")


def span_label(a, b, short=False) -> str:
    """'Jan 2017 – Jun 2018', 'Jul–Dec 2018' within one year, or 'Jan–Jun' when short."""
    if short:
        return f"{_p(a).strftime('%b')}–{_p(b).strftime('%b')}"
    if _p(a).year == _p(b).year:
        return f"{_p(a).strftime('%b')}–{month_label(b)}"
    return f"{month_label(a)} – {month_label(b)}"


_END = _p(MODEL_END)
HISTORY_LABEL = span_label(MODEL_START, MODEL_END)
HISTORY_MONTHS = (_END - _p(MODEL_START)).n + 1

# Plan = the calendar half-year that contains the first month after the data. Its months up to MODEL_END
# are actuals; the rest are forecast ("latest outlook" = actuals to date + forecast).
_NEXT = _END + 1
_PS = pd.Period(f"{_NEXT.year}-{1 if _NEXT.month <= 6 else 7:02d}", "M")
PLAN_START, PLAN_END = str(_PS), str(_PS + 5)
PLAN_NAME = f"H{1 if _PS.month == 1 else 2} {_PS.year}"
PLAN_LABEL = span_label(PLAN_START, PLAN_END)
ACTUAL_MONTHS = [str(m) for m in pd.period_range(_PS, _END, freq="M")] if _PS <= _END else []
FORECAST_START = str(_NEXT)
FORECAST_LABEL = span_label(FORECAST_START, PLAN_END)
ACTUAL_LABEL = span_label(ACTUAL_MONTHS[0], ACTUAL_MONTHS[-1]) if ACTUAL_MONTHS else ""
PLAN_Q1_LABEL = f"Q{_PS.quarter} {_PS.year} ({span_label(_PS, _PS + 2, short=True)})"
PLAN_Q2_LABEL = f"Q{(_PS + 3).quarter} {(_PS + 3).year} ({span_label(_PS + 3, _PS + 5, short=True)})"
# The half-year before the plan and the same half a year earlier, for comparison.
PREV_HALF = (str(_PS - 6), str(_PS - 1))
PREV_HALF_NAME = f"H{1 if (_PS - 6).month == 1 else 2} {(_PS - 6).year}"
LAST_YEAR_HALF = (str(_PS - 12), str(_PS - 7))
LAST_YEAR_HALF_NAME = f"H{1 if (_PS - 12).month == 1 else 2} {(_PS - 12).year}"

# Forecast check: plan from an earlier month as if the later months were unknown, then score against
# the actuals. CHECK_END is the planning month; months after it up to MODEL_END are the test.
CHECK_END = "2018-06"
CHECK_H = 6
CHECK_LABEL = span_label(str(_p(CHECK_END) + 1), MODEL_END)

# Markets: three largest states on their own, the South grouped, everything else pooled.
SOUTH_STATES = {"PR", "SC", "RS"}
SOLO_STATES = ["SP", "RJ", "MG"]
MARKETS = ["SP", "RJ", "MG", "South", "Rest of Brazil"]


# State names and capital-city coordinates (lat, lon), used to place states on the map.
STATES = {
    "AC": ("Acre", -9.97, -67.81), "AL": ("Alagoas", -9.67, -35.74), "AP": ("Amapá", 0.03, -51.07),
    "AM": ("Amazonas", -3.12, -60.02), "BA": ("Bahia", -12.97, -38.50), "CE": ("Ceará", -3.73, -38.53),
    "DF": ("Distrito Federal", -15.79, -47.88), "ES": ("Espírito Santo", -20.32, -40.34),
    "GO": ("Goiás", -16.68, -49.25), "MA": ("Maranhão", -2.53, -44.30), "MT": ("Mato Grosso", -15.60, -56.10),
    "MS": ("Mato Grosso do Sul", -20.44, -54.65), "MG": ("Minas Gerais", -19.92, -43.94),
    "PA": ("Pará", -1.46, -48.49), "PB": ("Paraíba", -7.12, -34.86), "PR": ("Paraná", -25.43, -49.27),
    "PE": ("Pernambuco", -8.05, -34.88), "PI": ("Piauí", -5.09, -42.80), "RJ": ("Rio de Janeiro", -22.91, -43.17),
    "RN": ("Rio Grande do Norte", -5.79, -35.21), "RS": ("Rio Grande do Sul", -30.03, -51.23),
    "RO": ("Rondônia", -8.76, -63.90), "RR": ("Roraima", 2.82, -60.67), "SC": ("Santa Catarina", -27.60, -48.55),
    "SP": ("São Paulo", -23.55, -46.63), "SE": ("Sergipe", -10.91, -37.07), "TO": ("Tocantins", -10.18, -48.33),
}

# Currency: all amounts are BRL. For display in euros, one constant rate (ECB 2018 annual average,
# series EXR.A.BRL.EUR.SP00.A) so exchange-rate swings do not distort growth.
BRL_PER_EUR = 4.3085
FX_SOURCE = "ECB reference rate, 2018 annual average"


def state_to_market(state: str) -> str:
    if state in SOLO_STATES:
        return state
    if state in SOUTH_STATES:
        return "South"
    return "Rest of Brazil"


# Categories: top N by delivered item revenue in the modelling window, the rest become "Other".
TOP_N_CATEGORIES = 10
OTHER_CATEGORY = "Other"
UNKNOWN_CATEGORY = "unknown"

# Driver tree: revenue = customers x orders per customer x AOV (monthly, per market).
DRIVERS = ["customers", "orders_per_customer", "aov"]

# Period comparisons for the driver decomposition: (label, base start, base end, current start, current end).
_YTD0 = pd.Period(f"{_END.year}-01", "M")
YOY = (f"YoY {span_label(_YTD0, _END, short=True)}", str(_YTD0 - 12), str(_END - 12), str(_YTD0), str(_END))
LAST6 = ("Last 6m vs prior 6m", str(_END - 11), str(_END - 6), str(_END - 5), str(_END))
COMPARISONS = [YOY, LAST6]
YOY_LABEL = f"{span_label(YOY[3], YOY[4], short=True)} {_END.year} vs {_END.year - 1}"
LAST6_LABEL = f"{span_label(LAST6[3], LAST6[4])} vs {span_label(LAST6[1], LAST6[2])}"

# Baseline drivers for scenarios: average of the last N months of the modelling window.
BASELINE_MONTHS = 3
BASELINE_LABEL = span_label(_END - 2, _END)
PRIOR3_LABEL = span_label(_END - 5, _END - 3)

# Forecast: the months from the end of the data to the end of the plan half-year.
FORECAST_H = (_p(PLAN_END) - _END).n
# Rolling-origin backtest: same horizon as the forecast, origin moved 1 month at a time.
BACKTEST_WINDOWS = 4
INTERVAL_LEVEL = 0.80
# Black Friday: one November in the data, so the uplift is a labelled judgement, not a learned effect.
# Uplift = Nov 2017 revenue / average of Oct and Dec 2017.
BLACK_FRIDAY_MONTH = 11

# Scenarios: customers and AOV compound monthly from the baseline; orders per customer is held fixed
# (Stage 2: it barely moves). Presets are derived from observed driver growth, see src/scenarios.py.
SCENARIOS = ["Downside", "Base", "Upside"]
# Sensitivity: extra monthly customer growth used to compare markets for investment.
SENSITIVITY_STEP = 0.01
