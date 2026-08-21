"""Fund eligibility on both paths that can open a position.

Blanking ETF screener scores to NULL (the correct fix — a fund has no company
fundamentals to score) makes _quality_ok pass them open, since it treats
unknown fundamentals as "no objection". Funds must therefore be excluded from
new buys explicitly, or a momentum ranking alone can open a fund position.

Backtests cannot catch this: features_daily is append-only, so historical rows
keep their old zeros and only the live agent sees NULL.
"""

import numpy as np
import pandas as pd

from datetime import date

from backend.agent.rules_strategy import RulesStrategy
from backend.agent.types import AgentLimits, PortfolioState

LIMITS = AgentLimits(
    max_daily_turnover=0.05,
    min_holdings=5,
    max_position_weight=0.16,
    min_trade_gbp=150.0,
    fx_fee=0.0015,
)

# Two candidates that clear every gate: top-decile momentum, above the 200-day
# average, and no fundamentals to object with.
FEATURES = pd.DataFrame(
    {"dist_sma200": [0.10, 0.10], "roic": [np.nan, np.nan], "screener_score": [np.nan, np.nan]},
    index=["QDVE.DE", "ACME"],
)
COMP = pd.Series({"QDVE.DE": 2.0, "ACME": 1.9})
PCTL = pd.Series({"QDVE.DE": 0.99, "ACME": 0.95})


def _buys(etf_symbols):
    strategy = RulesStrategy(LIMITS)
    return {i.symbol for i in strategy._new_buys(FEATURES, COMP, PCTL, set(), etf_symbols)}


def test_fund_is_not_opened_as_a_new_position():
    assert _buys(frozenset({"QDVE.DE"})) == {"ACME"}


def test_etc_is_excluded_too():
    """SGLN.L-style ETCs are quoteType EQUITY, so they never reach etf_symbols
    and must be caught by the shared is_fund predicate instead."""
    features = FEATURES.rename(index={"QDVE.DE": "SGLN.L"})
    comp = COMP.rename({"QDVE.DE": "SGLN.L"})
    pctl = PCTL.rename({"QDVE.DE": "SGLN.L"})
    strategy = RulesStrategy(LIMITS)
    bought = {i.symbol for i in strategy._new_buys(features, comp, pctl, set(), frozenset())}
    assert bought == {"ACME"}


def test_ordinary_equity_still_qualifies_on_unknown_fundamentals():
    """Guard against over-correcting: pass-open on NaN is deliberate for stocks."""
    assert _buys(frozenset()) == {"QDVE.DE", "ACME"}


def _state(cash_frac, etf_symbols=frozenset()):
    """propose() short-circuits to _deploy above DEPLOY_CASH_FRAC, so cash decides
    which of the two opening paths runs."""
    return PortfolioState(
        date=date(2026, 8, 21),
        total_value_gbp=100_000.0,
        cash_gbp=100_000.0 * cash_frac,
        weights={},
        quantities={},
        currencies={"QDVE.DE": "EUR", "ACME": "USD"},
        tags={},
        etf_symbols=etf_symbols,
    )


def test_deploy_path_does_not_open_funds():
    """_deploy is a separate opening path from _new_buys, reached by an exclusive
    early return in propose() above DEPLOY_CASH_FRAC — backtest day one and any
    large deposit land there."""
    strategy = RulesStrategy(LIMITS)
    state = _state(0.90, frozenset({"QDVE.DE"}))
    assert state.cash_gbp / state.total_value_gbp > strategy.DEPLOY_CASH_FRAC
    symbols = {i.symbol for i in strategy._deploy(FEATURES, COMP, state)}
    assert symbols == {"ACME"}


def test_deploy_path_excludes_etcs_too():
    strategy = RulesStrategy(LIMITS)
    features = FEATURES.rename(index={"QDVE.DE": "SGLN.L"})
    comp = COMP.rename({"QDVE.DE": "SGLN.L"})
    symbols = {i.symbol for i in strategy._deploy(features, comp, _state(0.90))}
    assert symbols == {"ACME"}


def test_deploy_still_opens_ordinary_equities():
    """Guard against over-correcting the fix into a dead deployment path."""
    strategy = RulesStrategy(LIMITS)
    symbols = {i.symbol for i in strategy._deploy(FEATURES, COMP, _state(0.90))}
    assert symbols == {"QDVE.DE", "ACME"}
