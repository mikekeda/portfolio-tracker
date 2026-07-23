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
from models import FeaturesDaily, Instrument, InstrumentYahoo


def _num(value) -> float | None:
    """Float or None; NaN/inf become None so they store as NULL, not 'NaN'::float8."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _row_from_holding(h: dict, instrument_id: int, margins: dict) -> dict:
    return {
        "instrument_id": instrument_id,
        "date": datetime.now(TIMEZONE).date(),
        "roic": _num(h.get("roic")),
        "gross_margin": margins.get("gross"),
        "operating_margin": margins.get("operating"),
        "profit_margin": _num(h.get("profit_margins")),
        "revenue_growth": _num(h.get("revenue_growth")),
        "fcf_yield": _num(h.get("free_cashflow_yield")),
        "debt_to_equity": _num(h.get("debtToEquity")),
        "short_percent_float": _num(h.get("short_percent_of_float")),
        "analyst_rec_mean": _num(h.get("recommendation_mean")),
        "analyst_count": h.get("number_of_analyst_opinions"),
        "analyst_target_upside": _num(h.get("prediction")),
        "forward_pe": _num(h.get("forward_pe_ratio")),
        "peg": _num(h.get("peg_ratio")),
        "ps_ratio": _num(h.get("ps_ratio")),
        "pe_5y_avg_vs_current_pct": _num(h.get("pe_5y_avg_vs_current_pct")),
        "dcf_price": _num(h.get("dcf_price")),
        "dcf_diff": _num(h.get("dcf_diff")),
        "dcf_implied_growth": _num(h.get("dcf_implied_growth")),
        "rule_of_40": _num(h.get("rule_of_40_score")),
        "f_score": h.get("f_score"),
        "screener_score": _num(h.get("screener_score")),
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

        # Gross/operating margins aren't in the holding dict; extract just those
        # two keys in SQL rather than loading ~1k full info blobs.
        margin_rows = await session.execute(
            select(
                InstrumentYahoo.instrument_id,
                InstrumentYahoo.info["grossMargins"].astext,
                InstrumentYahoo.info["operatingMargins"].astext,
            )
        )
        margins_by_id: dict[int, dict] = {}
        for inst_id, gross, operating in margin_rows.all():
            margins_by_id[inst_id] = {
                "gross": float(gross) * 100.0 if gross is not None else None,
                "operating": float(operating) * 100.0 if operating is not None else None,
            }

        rows = []
        skipped = 0
        for h in holdings:
            instrument_id = id_by_symbol.get(h["yahoo_symbol"])
            if instrument_id is None:
                skipped += 1
                continue
            rows.append(_row_from_holding(h, instrument_id, margins_by_id.get(instrument_id, {})))

        if not rows:
            logger.warning("Features: nothing to write (0 holdings resolved)")
            return

        stmt = pg_insert(FeaturesDaily).values(rows)
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
