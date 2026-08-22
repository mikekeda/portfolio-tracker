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
from datetime import datetime, timedelta

from sqlalchemy import func, null, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app import get_session
from backend.utils.etf_aggregates import METRIC_AGGREGATION, aggregate_fund
from backend.views._etf_overlay import SCREENER_RATIO
from backend.views.portfolio import build_portfolio
from config import TIMEZONE, logger
from models import EtfHolding, FeaturesDaily, Instrument, InstrumentYahoo, TransactionAction, TransactionHistory


# PostgreSQL's wire protocol caps a statement at 32767 bind parameters.
MAX_BIND_PARAMS = 32767

# Both distributing funds held here pay ~1.8x a year, so a full year with no
# payout identifies an accumulating fund. Shorter, and a recent buy looks like one.
MIN_DISTRIBUTION_WINDOW_DAYS = 365

DISTRIBUTION_ACTIONS = (
    TransactionAction.DIVIDEND,
    TransactionAction.DIVIDEND_PROPERTY,
    TransactionAction.DIVIDEND_TAX_EXEMPT,
)


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


def _constituent_metrics(holding: dict) -> dict:
    """One constituent's metrics, with the screener expressed as a ratio of its max.

    A separate view rather than extra keys on the holding dict: these rows are on
    their way into features_daily, and the ratio is not one of its columns.
    """
    metrics = {metric: holding.get(metric) for metric in METRIC_AGGREGATION if metric != SCREENER_RATIO}

    score, maximum = holding.get("screener_score"), holding.get("screener_score_max")
    if score is not None and maximum:
        metrics[SCREENER_RATIO] = score / maximum
    return metrics


async def _accumulating_funds(session: AsyncSession, fund_ids: list[int]) -> set[int]:
    """Funds held long enough to have paid a distribution, that never have.

    Yahoo reports dividendYield for some accumulating ETFs (XNAS.L: 0.0) and
    omits it for others (VUAG.L, R1GR.L), so the tile reads 0.00% for one and
    blank for the next. Payout history settles it from data we already hold.

    Only reaches funds that clear the coverage gate, since it travels on the
    look-through payload: R2SC.L is accumulating but stays blank.
    """
    if not fund_ids:
        return set()

    rows = (
        await session.execute(
            select(
                Instrument.id,
                func.min(TransactionHistory.timestamp).label("first_txn"),
                func.count(TransactionHistory.id)
                .filter(TransactionHistory.action.in_(DISTRIBUTION_ACTIONS))
                .label("distributions"),
            )
            .join(TransactionHistory, TransactionHistory.isin == Instrument.isin)
            .where(Instrument.id.in_(fund_ids))
            .group_by(Instrument.id)
        )
    ).all()

    cutoff = datetime.now(TIMEZONE).replace(tzinfo=None) - timedelta(days=MIN_DISTRIBUTION_WINDOW_DAYS)
    return {fund_id for fund_id, first_txn, distributions in rows if not distributions and first_txn <= cutoff}


async def _update_etf_derived_metrics(
    session: AsyncSession, holdings: list[dict], id_by_symbol: dict[str, int]
) -> None:
    """Aggregate each fund's constituents into fund-level metrics and store them.

    Runs on the un-overlaid universe, after the FeaturesDaily upsert, so the
    inputs are the same constituent figures the Holdings page shows.
    """
    metrics_by_instrument = {
        instrument_id: _constituent_metrics(h)
        for h in holdings
        if (instrument_id := id_by_symbol.get(h["yahoo_symbol"])) is not None
    }

    latest = (
        select(EtfHolding.etf_instrument_id, func.max(EtfHolding.date).label("date"))
        .group_by(EtfHolding.etf_instrument_id)
        .subquery()
    )
    rows = (
        await session.execute(
            select(
                EtfHolding.etf_instrument_id,
                EtfHolding.instrument_id,
                EtfHolding.weight_pct,
                EtfHolding.date,
            ).join(
                latest,
                (latest.c.etf_instrument_id == EtfHolding.etf_instrument_id) & (latest.c.date == EtfHolding.date),
            )
        )
    ).all()

    # Unresolved constituents count toward the fund's published weight: coverage
    # must describe the fund, not the sample we happen to track.
    published: dict[int, float] = {}
    constituents: dict[int, list[tuple[float, dict]]] = {}
    snapshot_date: dict[int, str] = {}
    for fund_id, instrument_id, weight_pct, snapshot in rows:
        published[fund_id] = published.get(fund_id, 0.0) + weight_pct
        snapshot_date[fund_id] = snapshot.isoformat()
        metrics = metrics_by_instrument.get(instrument_id) if instrument_id else None
        if metrics is not None:
            constituents.setdefault(fund_id, []).append((weight_pct, metrics))

    accumulating = await _accumulating_funds(session, list(published))

    written = 0
    for fund_id, total_weight in published.items():
        values, coverage = aggregate_fund(constituents.get(fund_id, []), total_weight)
        payload = (
            {
                "metrics": values,
                "coverage": coverage,
                "n_resolved": len(constituents.get(fund_id, [])),
                "as_of": snapshot_date[fund_id],
                "computed_at": datetime.now(TIMEZONE).date().isoformat(),
                # Outside "metrics": established from payout history, not from
                # constituents. None means unknown, not zero.
                "distribution_yield": 0.0 if fund_id in accumulating else None,
            }
            if values
            else None
        )
        if payload:
            written += 1
        # sqlalchemy null(), not None: None on a JSONB column stores 'null'::jsonb,
        # which is not SQL NULL and slips past every IS NULL check downstream.
        await session.execute(
            update(InstrumentYahoo)
            .where(InstrumentYahoo.instrument_id == fund_id)
            .values(derived_metrics=payload if payload else null())
        )

    await session.commit()
    logger.info(
        "Features: derived metrics for %d/%d funds (%d below the coverage gate)",
        written,
        len(published),
        len(published) - written,
    )


async def update_features() -> None:
    """Upsert today's FeaturesDaily row for every monitored instrument."""
    async with get_session() as session:
        portfolio = await build_portfolio(session, show_all=True)
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

        await _update_etf_derived_metrics(session, holdings, id_by_symbol)


def main() -> None:
    asyncio.run(update_features())


if __name__ == "__main__":
    main()
