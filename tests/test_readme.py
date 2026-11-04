"""The README quotes headline numbers by hand. Check each one against the pipeline outputs."""
import re

import pytest

from src import config as cfg, memo

README = (cfg.ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def d():
    return memo.load()


def expected(d) -> list[str]:
    tot, msc, csc = d["tot"], d["msc"], d["csc"]
    b, dn, up = tot.loc["Base"], tot.loc["Downside"], tot.loc["Upside"]
    yoy, base, dq, fs = d["yoy"], d["base"], d["dq"], d["fs"]
    wa = d["scores"].groupby("model")["wape"].mean()
    tb = d["scores"][d["scores"]["unique_id"] == "Total"].set_index("model")["bias"][["RWD", "AutoETS", "AutoTheta"]]
    return [
        f"{dq['order_status_counts']['delivered']:,} delivered orders",
        f"R$ {b['six_months']/1e6:.2f}M** for the 6 months",
        f"R$ {b['plan_q1']/1e6:.2f}M in Sep–Nov", f"R$ {b['plan_q2']/1e6:.2f}M in Dec–Feb",
        f"downside of R$ {dn['six_months']/1e6:.2f}M",
        f"fell {-base.loc['Total', 'customers_growth_vs_prior']:.0%} in Jun–Aug",
        f"R$ {(b['six_months']-dn['six_months'])/1e6:.2f}M below plan",
        f"is R$ {up['six_months']/1e6:.2f}M",
        f"({msc.loc['SP', 'revenue_share']:.0%} of revenue)",
        f"worth R$ {msc.loc['SP', 'extra_6m_revenue_per_1pt_customers']/1e3:.0f}k",
        f"(revenue down {-msc.loc['RJ', 'revenue_growth_last6m']:.0%})",
        f"Revenue rose {yoy.loc['Total', 'revenue_growth']:.0%}",
        f"explain {yoy.loc['Total', 'contrib_customers']/yoy.loc['Total', 'revenue_change']:.0%} of that rise",
        f"Only {dq['repeat_rate_model_window']:.0%} of customers",
        f"over-forecast by {tb.min()*100:.0f}–{tb.max():.0%}",
        f"({wa['WindowAverage']:.1%} WAPE",
        f"{dq['orders_excluded_not_delivered']:,}\n  non-delivered orders",
        f"(×{fs['black_friday_uplift']:.2f}, from one November)",
    ]


def test_readme_numbers_match_outputs(d):
    missing = [s for s in expected(d) if s not in README]
    assert not missing, f"README out of date, not found: {missing}"


def test_readme_images_exist():
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README):
        assert (cfg.ROOT / path).exists(), path


def test_readme_names_the_selected_model(d):
    assert d["fs"]["selected_model"] == "WindowAverage" and "3-month average" in README
