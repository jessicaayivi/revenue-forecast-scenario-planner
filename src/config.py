"""Project-wide paths and constants. All paths are relative to the repo root."""
from pathlib import Path

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

# Markets: three largest states on their own, the South grouped, everything else pooled.
SOUTH_STATES = {"PR", "SC", "RS"}
SOLO_STATES = ["SP", "RJ", "MG"]
MARKETS = ["SP", "RJ", "MG", "South", "Rest of Brazil"]


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
COMPARISONS = [
    ("YoY Jan-Aug", "2017-01", "2017-08", "2018-01", "2018-08"),
    ("Last 6m vs prior 6m", "2017-09", "2018-02", "2018-03", "2018-08"),
]

# Baseline drivers for scenarios: average of the last N months of the modelling window.
BASELINE_MONTHS = 3

# Forecast: next 2 quarters after the modelling window (Sep 2018 - Feb 2019).
FORECAST_H = 6
# Rolling-origin backtest: 6-month horizon, origin moved 1 month at a time.
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
