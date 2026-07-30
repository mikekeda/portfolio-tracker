"""Unit tests for screener evaluation and scoring.

Pure stdlib — runnable without the app's dependencies:

    python tests/test_screener_config.py        # plain asserts
    pytest tests/test_screener_config.py        # also pytest-compatible

Fixture values are the real Yahoo figures those tickers carried when the
screeners were reworked, so a threshold change that breaks the intent shows up
as a failing named ticker rather than an abstract number.
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.screener_config import SCORE_NORMALIZER, ScreenerCriteria, Sector, get_screener_config

CONFIG = get_screener_config()


def holding(**overrides) -> dict:
    """A holding that passes nothing, so each test states exactly what it relies on."""
    base = {
        "sector": Sector.TECHNOLOGY.value,
        "roic": 0.0,
        "roic_3y_min": None,
        "revenue_growth": 0.0,
        "profit_margins": 0.0,
        "gross_margin": 0.0,
        "operating_margin": 0.0,
        "fcf_margin": 0.0,
        "free_cashflow_yield": 0.0,
        "net_cash": False,
        "net_debt_to_ebitda": None,
        "peg_ratio": None,
        "pe_ratio": None,
        "current_price": 100.0,
        "sma_200": 200.0,
    }
    return {**base, **overrides}


def passes(screener_id: str, fields: dict) -> bool:
    return CONFIG.passes_screener(fields, CONFIG.screeners[screener_id])


# --- pre_profit_growth: the sleeve that previously could only score negative ---


RKLB = dict(profit_margins=-26.9, revenue_growth=63.5, gross_margin=36.6, net_cash=True, current_price=250.0)


def test_pre_profit_growth_accepts_funded_high_margin_growth():
    assert passes("pre_profit_growth", holding(**RKLB))


def test_pre_profit_growth_rejects_thin_gross_margin():
    # LUNR (9.7%), RCAT (7.5%) and RDW (12.9%) grow fast with no unit economics to scale.
    for gross in (9.7, 7.5, 12.9):
        assert not passes("pre_profit_growth", holding(**{**RKLB, "gross_margin": gross})), gross


def test_pre_profit_growth_requires_funding_and_uptrend():
    assert not passes("pre_profit_growth", holding(**{**RKLB, "net_cash": False}))
    assert not passes("pre_profit_growth", holding(**{**RKLB, "current_price": 150.0}))


def test_pre_profit_growth_excludes_profitable_compounders():
    # MU (55.9% net margin) is already scored by the quality screeners; counting
    # it here too would double-count the same growth.
    assert not passes("pre_profit_growth", holding(**{**RKLB, "profit_margins": 55.9, "gross_margin": 72.6}))


# --- red_flag_low_roic: must no longer be the only signal on a scaling company ---


def test_low_roic_red_flag_exempts_high_growth():
    scaling = holding(roic=-22.4, revenue_growth=63.5)
    assert not passes("red_flag_low_roic", scaling)


def test_low_roic_red_flag_still_fires_on_stagnant_business():
    stagnant = holding(roic=2.0, revenue_growth=3.0)
    assert passes("red_flag_low_roic", stagnant)


# --- quality gates: margin, not yield ---


def test_qarp_accepts_capex_heavy_compounder():
    # NVDA: 1.0% FCF yield (fails the old gate) but an 18.3% FCF margin.
    nvda = holding(roic=67.5, peg_ratio=0.54, fcf_margin=18.3, free_cashflow_yield=1.0, profit_margins=63.0)
    assert passes("qarp", nvda)


def test_qarp_rejects_weak_cash_conversion():
    amzn = holding(roic=12.0, peg_ratio=1.24, fcf_margin=1.3, profit_margins=12.2)
    assert not passes("qarp", amzn)


def test_top_quality_proxy_uses_leverage_not_debt_to_equity():
    # MSFT-shaped: mildly net-debt, which a strict net-cash test would reject.
    msft = holding(revenue_growth=18.3, fcf_margin=11.6, net_debt_to_ebitda=0.4, roic=30.0, profit_margins=39.3)
    assert passes("top_quality_proxy", msft)

    levered = {**msft, "net_debt_to_ebitda": 2.5}
    assert not passes("top_quality_proxy", levered)


def test_high_debt_red_flag_ignores_buyback_shrunk_equity():
    # Buybacks push debtToEquity past 150 while leverage vs earnings stays sound.
    buyback_heavy = holding(debtToEquity=400.0, net_debt_to_ebitda=1.2)
    assert not passes("red_flag_high_debt", buyback_heavy)
    assert passes("red_flag_high_debt", holding(net_debt_to_ebitda=4.0))


# --- roic_consistency: durability, not a single TTM window ---


def test_roic_consistency_rejects_cyclical_rebound():
    # A trough year in the 3-year window disqualifies however good today looks.
    cyclical = holding(roic=14.0, roic_3y_min=1.2, revenue_growth=345.7)
    assert not passes("roic_consistency", cyclical)


def test_roic_consistency_accepts_durable_compounder():
    durable = holding(roic=23.3, roic_3y_min=19.8, revenue_growth=24.2)
    assert passes("roic_consistency", durable)


def test_roic_consistency_needs_a_full_three_year_run():
    # get_roic_history returns [] on a partial run, so the field arrives as None.
    assert not passes("roic_consistency", holding(roic=30.0, roic_3y_min=None, revenue_growth=20.0))


# --- sector exclusion and normalisation ---


def test_quality_screeners_skip_financials():
    bank = holding(sector=Sector.FINANCIAL_SERVICES.value, roic=23.9, roic_3y_min=20.0, revenue_growth=15.6)
    assert not passes("roic_consistency", bank)
    assert passes("roic_consistency", {**bank, "sector": Sector.TECHNOLOGY.value})


def test_normalizer_scales_down_for_excluded_sectors():
    unrestricted = CONFIG.score_normalizer(None)
    assert unrestricted == SCORE_NORMALIZER
    assert CONFIG.score_normalizer(Sector.FINANCIAL_SERVICES.value) < unrestricted
    assert CONFIG.score_normalizer(Sector.TECHNOLOGY.value) == unrestricted


def test_unknown_sector_falls_back_to_unrestricted():
    assert CONFIG.score_normalizer("Not A Sector") == SCORE_NORMALIZER


# --- combination bonus ---


def test_bonus_ignores_screeners_reading_the_same_inputs():
    # qarp and top_quality_proxy endorse each other but share roic/fcf_margin/profit_margins.
    assert CONFIG.combination_bonus(["qarp", "top_quality_proxy"]) == 0


def test_bonus_rewards_independent_confirmation():
    # roic_consistency (fundamentals) + momentum_pullback (price) share nothing.
    assert CONFIG.combination_bonus(["roic_consistency", "momentum_pullback"]) > 0


def test_bonus_needs_a_curated_edge():
    assert CONFIG.combination_bonus(["death_cross", "qarp"]) == 0
    assert CONFIG.combination_bonus(["qarp"]) == 0
    assert CONFIG.combination_bonus([]) == 0


def test_bonus_has_diminishing_returns():
    many = ["roic_consistency", "momentum_pullback", "r40_momentum", "breakout_quiet_base", "oversold_uptrend"]
    pairs = sum(
        1
        for a, b in combinations(many, 2)
        if b in CONFIG.screeners[a].combine_with_set or a in CONFIG.screeners[b].combine_with_set
    )
    assert 0 < CONFIG.combination_bonus(many) < 2 * pairs


# --- engine invariants ---


def test_missing_field_fails_rather_than_defaulting():
    assert not passes("roic_consistency", {"sector": Sector.TECHNOLOGY.value})


def test_non_finite_field_fails():
    assert not passes("roic_consistency", holding(roic_3y_min=float("nan"), revenue_growth=20.0))


def test_field_set_covers_field_references():
    # pre_profit_growth compares current_price against FieldRef("sma_200").
    assert "sma_200" in CONFIG.screeners["pre_profit_growth"].field_set


def test_every_criterion_field_is_declared():
    declared = set(CONFIG.get_available_fields())
    for screener in CONFIG.screeners.values():
        assert screener.field_set <= declared, screener.id


def test_no_declared_field_is_unused():
    used: set[str] = set()
    for screener in CONFIG.screeners.values():
        used |= screener.field_set
    assert set(CONFIG.get_available_fields()) == used


def test_criteria_operators_are_supported():
    for screener in CONFIG.screeners.values():
        for criteria in screener.criteria:
            assert isinstance(criteria, ScreenerCriteria)
            assert criteria.operator in CONFIG.get_available_operators()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
