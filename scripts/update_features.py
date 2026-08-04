"""
scripts/update_features.py
==========================
Persist today's point-in-time feature snapshot for every monitored instrument.

Reuses the fully enriched holdings pipeline from the portfolio view (screener,
DCF, ROIC, F-score, thesis rules — all computed from DB caches, no API keys),
and upserts one FeaturesDaily row per instrument. Run nightly from project root:

    python scripts/update_features.py
"""

import asyncio
import math
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app import get_session
from backend.views.portfolio import get_current_portfolio
from config import TIMEZONE, logger
from models import FeaturesDaily, Instrument


# PostgreSQL's wire protocol caps a statement at 32767 bind parameters.
MAX_BIND_PARAMS = 32767


def _num(value) -> float | None:
    """Float or None; NaN/inf become None so they store as NULL, not 'NaN'::float8."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _row_from_holding(h: dict, instrument_id: int) -> dict:
    targets = h.get("analyst_price_targets") or {}
    return {
        "instrument_id": instrument_id,
        "date": datetime.now(TIMEZONE).date(),
        "roic": _num(h.get("roic")),
        "roic_ttm": _num(h.get("roic_ttm")),
        "roic_ttm_trend": _num(h.get("roic_ttm_trend")),
        "gross_margin_trend_4q": _num(h.get("gross_margin_trend_4q")),
        "operating_margin_trend_4q": _num(h.get("operating_margin_trend_4q")),
        "revenue_growth_4q_avg": _num(h.get("revenue_growth_4q_avg")),
        "eps_revision_ratio_30d": _num(h.get("eps_revision_ratio_30d")),
        "eps_next_q_growth": _num(h.get("eps_next_q_growth")),
        "gross_margin": _num(h.get("gross_margin")),
        "operating_margin": _num(h.get("operating_margin")),
        "profit_margin": _num(h.get("profit_margins")),
        "revenue_growth": _num(h.get("revenue_growth")),
        "fcf_yield": _num(h.get("free_cashflow_yield")),
        "debt_to_equity": _num(h.get("debtToEquity")),
        "short_percent_float": _num(h.get("short_percent_of_float")),
        "analyst_rec_mean": _num(h.get("recommendation_mean")),
        "analyst_count": h.get("number_of_analyst_opinions"),
        "analyst_target_upside": _num(h.get("prediction")),
        "target_low": _num(targets.get("low")),
        "target_median": _num(targets.get("median")),
        "target_high": _num(targets.get("high")),
        "forward_pe": _num(h.get("forward_pe_ratio")),
        "peg": _num(h.get("peg_ratio")),
        "ps_ratio": _num(h.get("ps_ratio")),
        "pe_5y_avg_vs_current_pct": _num(h.get("pe_5y_avg_vs_current_pct")),
        "dcf_price": _num(h.get("dcf_price")),
        "dcf_diff": _num(h.get("dcf_diff")),
        "dcf_implied_growth": _num(h.get("dcf_implied_growth")),
        "dcf_low": _num(h.get("dcf_low")),
        "dcf_high": _num(h.get("dcf_high")),
        "avg_pe_5y": _num(h.get("avg_pe")),
        "pe_basis_matches": h.get("pe_basis_matches"),
        "trailing_eps": _num(h.get("trailing_eps")),
        "forward_eps": _num(h.get("forward_eps")),
        "rule_of_40": _num(h.get("rule_of_40_score")),
        "f_score": h.get("f_score"),
        "screener_score": _num(h.get("screener_score")),
        "screener_score_max": _num(h.get("screener_score_max")),
        "passed_screeners": h.get("passedScreeners") or [],
        "thesis_rule_eval": h.get("thesis_rule_eval"),
        "extras": {"dcf_implied_growth_status": h.get("dcf_implied_growth_status")},
        # Column is timestamp without time zone; asyncpg rejects aware datetimes
        "updated_at": datetime.now(TIMEZONE).replace(tzinfo=None),
    }


async def update_features() -> None:
    """Upsert today's FeaturesDaily row for every monitored instrument."""
    async with get_session() as session:
        portfolio = await get_current_portfolio(session=session, show_all=True)
        holdings = portfolio["holdings"]

        id_rows = await session.execute(
            select(Instrument.id, Instrument.yahoo_symbol).where(Instrument.yahoo_symbol.is_not(None))
        )
        id_by_symbol = {sym: inst_id for inst_id, sym in id_rows.all()}

        rows = []
        skipped = 0
        for h in holdings:
            instrument_id = id_by_symbol.get(h["yahoo_symbol"])
            if instrument_id is None:
                skipped += 1
                continue
            rows.append(_row_from_holding(h, instrument_id))

        if not rows:
            logger.warning("Features: nothing to write (0 holdings resolved)")
            return

        # Chunk size is derived, not fixed: one bind parameter per column per row,
        # so adding a column silently shrinks how many rows fit in a statement.
        chunk_size = max(1, MAX_BIND_PARAMS // len(rows[0]))
        for start in range(0, len(rows), chunk_size):
            stmt = pg_insert(FeaturesDaily).values(rows[start : start + chunk_size])
            update_cols = {
                c.name: getattr(stmt.excluded, c.name)
                for c in FeaturesDaily.__table__.columns
                if c.name not in ("id", "instrument_id", "date")
            }
            stmt = stmt.on_conflict_do_update(constraint="uq_features_instrument_date", set_=update_cols)
            await session.execute(stmt)
        await session.commit()

        with_score = sum(1 for r in rows if r["screener_score"] is not None)
        with_roic = sum(1 for r in rows if r["roic"] is not None)
        logger.info(
            "Features: upserted %d rows for %s (%d with screener_score, %d with roic, %d skipped)",
            len(rows),
            rows[0]["date"].isoformat(),
            with_score,
            with_roic,
            skipped,
        )


def main() -> None:
    asyncio.run(update_features())


if __name__ == "__main__":
    main()
