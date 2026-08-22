"""
ETF Look-Through Overlay
========================
Fills fund rows in an API response with the fundamentals aggregated from their
constituents by `scripts/update_features.py`.

**Display only.** The overlay must run in the route, never inside the portfolio
builder: `update_features` and `run_position_review` consume the builder's
output, and anything left on a holding dict reaches `features_daily` and from
there the agent's cross-sectional z-scores.

Both pages overlay from here. The Holdings row and the Stock payload have
different shapes but must produce an identical composite score — the Stock
tooltip says so in as many words.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.screener_config import SCORE_NORMALIZER
from models import Instrument, InstrumentYahoo

# Writing 13F to `form13f_score` would fill the 13F column, claiming managers
# hold the *fund*. Its own key keeps it feeding the composite and nothing else.
LOOK_THROUGH_ONLY = {"form13f_score": "look_through_form13f"}

# metric -> Stock `fundamentals` field, for metrics whose units already match.
# Yahoo's fractions vs our percents need conversion and are not in here yet.
FUNDAMENTALS_FIELDS = {
    "recommendation_mean": "recommendationMean",
    "pe_ratio": "peRatio",
    "peg_ratio": "pegRatio",
    "free_cashflow_yield": "fcfYield",  # not a Yahoo field — Stock computes its own
}

# Reconstructed rather than stored as a ratio because computeComposite divides
# score by max. A transport encoding, not a claim that the fund passed gates.
SCREENER_RATIO = "screener_ratio"

# Matches the rounding portfolio.py applies to Yahoo's own value, so the two
# pages feed the composite identical inputs.
REC_MEAN_DP = 2


async def fund_derived_metrics(session: AsyncSession, symbols: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Stored look-through payloads keyed by yahoo_symbol, for funds that have one."""
    query = (
        select(Instrument.yahoo_symbol, InstrumentYahoo.derived_metrics)
        .join(InstrumentYahoo, InstrumentYahoo.instrument_id == Instrument.id)
        .where(InstrumentYahoo.derived_metrics.is_not(None))
    )
    if symbols is not None:
        query = query.where(Instrument.yahoo_symbol.in_(symbols))

    return {symbol: payload for symbol, payload in (await session.execute(query)).all() if payload}


def _screener_pair(ratio: float) -> tuple[float, float]:
    """The aggregated ratio encoded as the (score, max) pair computeComposite divides."""
    return round(ratio * SCORE_NORMALIZER, 2), SCORE_NORMALIZER


def _provenance(payload: dict[str, Any], metrics: dict[str, Any], filled: list[str]) -> dict[str, Any]:
    """The look-through marker fields both page shapes carry.

    `filled` names the metrics this overlay actually supplied, so the UI can mark
    those and leave a fund-reported figure unmarked. `pe_ratio` is the only
    collision today: Yahoo publishes no other fund-level metric we also derive.
    """
    return {
        "look_through": True,
        "look_through_fields": filled,
        "look_through_form13f": metrics.get("form13f_score"),
        "look_through_coverage": payload.get("coverage") or {},
        "look_through_n": payload.get("n_resolved"),
        "look_through_as_of": payload.get("as_of"),
    }


def apply_derived_metrics(holding: dict, payload: dict[str, Any]) -> None:
    """Fill the gaps on a Holdings row from one fund's look-through metrics, in place.

    A figure the fund itself reports is a fact about the fund; a constituent
    aggregate is an estimate of it. The estimate never overwrites the fact —
    Yahoo carries a real trailingPE for four of the six funds, and letting the
    derived value win made Holdings disagree with the Stock page (UKDV 17 vs 15).
    """
    metrics = payload.get("metrics") or {}
    filled = []

    for metric, value in metrics.items():
        # The screener pair has no fund-reported counterpart: every fund is
        # blanked by calculate_screener_results, so this is the only source.
        if metric == SCREENER_RATIO:
            holding["screener_score"], holding["screener_score_max"] = _screener_pair(value)
            filled.append(metric)
            continue

        key = LOOK_THROUGH_ONLY.get(metric, metric)
        if holding.get(key) is not None:
            continue
        holding[key] = round(value, REC_MEAN_DP) if metric == "recommendation_mean" else value
        filled.append(metric)

    holding.update(_provenance(payload, metrics, filled))


def apply_derived_to_instrument(detail: dict, payload: dict[str, Any]) -> None:
    """Write one fund's look-through metrics onto the Stock page payload, in place.

    Same gap-filling precedence as the Holdings row, over FUNDAMENTALS_FIELDS —
    so the two pages cannot show different numbers for the same fund. Tiles
    outside that mapping stay blank for funds until their units are handled.
    """
    metrics = payload.get("metrics") or {}
    filled = []

    ratio = metrics.get(SCREENER_RATIO)
    if ratio is not None:
        detail["screener_score"], detail["screener_score_max"] = _screener_pair(ratio)
        filled.append(SCREENER_RATIO)

    fundamentals = detail["fundamentals"]
    for metric, field in FUNDAMENTALS_FIELDS.items():
        value = metrics.get(metric)
        if value is None or fundamentals.get(field) is not None:
            continue
        fundamentals[field] = round(value, REC_MEAN_DP) if metric == "recommendation_mean" else value
        filled.append(metric)

    # Sits outside `metrics`: a fact about the fund's payouts, not a constituent
    # aggregate. Yahoo omits it for VUAG.L and R1GR.L, which report 0% in truth.
    distribution_yield = payload.get("distribution_yield")
    if distribution_yield is not None and fundamentals.get("dividendYield") is None:
        fundamentals["dividendYield"] = distribution_yield
        filled.append("distribution_yield")

    detail.update(_provenance(payload, metrics, filled))


async def overlay_etf_derived(session: AsyncSession, holdings: list[dict]) -> None:
    """Overlay look-through metrics onto every fund row in `holdings`, in place."""
    by_symbol = await fund_derived_metrics(session, [h["yahoo_symbol"] for h in holdings if h.get("yahoo_symbol")])
    if not by_symbol:
        return

    for holding in holdings:
        payload = by_symbol.get(holding.get("yahoo_symbol"))
        if payload:
            apply_derived_metrics(holding, payload)
