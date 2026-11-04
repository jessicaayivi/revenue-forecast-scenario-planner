"""Decision memo tests: the committed memo must match the data, and its claims must still hold."""
import pandas as pd
import pytest

from src import config as cfg, memo

P = cfg.PROCESSED_DIR


@pytest.fixture(scope="module")
def data():
    return memo.load()


def test_committed_memo_is_up_to_date(data):
    assert memo.OUT.read_text(encoding="utf-8") == memo.render(data), \
        "docs/decision_memo.md is stale: run python -m src.memo"


def test_claims_hold(data):
    memo.check_claims(data)


def test_claim_check_catches_a_contradiction(data):
    broken = dict(data)
    msc = data["msc"].copy()
    msc.loc["RJ", "revenue_growth_last6m"] = 0.5
    broken["msc"] = msc
    with pytest.raises(AssertionError, match="RJ shrank"):
        memo.check_claims(broken)


def test_headline_numbers_appear(data):
    text = memo.OUT.read_text(encoding="utf-8")
    tot = pd.read_csv(P / "scenario_summary.csv").query("market == 'Total'").set_index("scenario")
    for s in cfg.SCENARIOS:
        assert f"R$ {tot.loc[s, 'six_months']/1e6:.2f}M" in text


def test_triggers_sit_between_scenarios(data):
    t = memo.triggers(data)
    assert t["down"] < t["base"] < t["up"]


def test_memo_images_exist():
    text = memo.OUT.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("!["):
            path = line.split("(", 1)[1].rstrip(")")
            assert (memo.OUT.parent / path).exists(), path
