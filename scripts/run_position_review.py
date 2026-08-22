"""
scripts/run_position_review.py
==============================
Generate (or refresh) LLM position reviews for held instruments with a thesis.

Skips instruments whose context hash matches the latest row. Run from project root:

    python scripts/run_position_review.py
"""

import asyncio

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app import get_session
from backend.schemas.instrument_thesis import InstrumentThesisSchema
from backend.utils.position_review import (
    MODEL,
    build_context,
    generate_assessment,
    hash_context,
)
from backend.views.portfolio import build_portfolio
from config import logger
from models import Instrument, MarketMetricsDaily, PositionReview


async def run_position_reviews() -> None:
    """Generate position reviews for held instruments; skip unchanged inputs."""
    async with get_session() as session:
        portfolio = await build_portfolio(session, show_all=False)
        holdings = [h for h in portfolio["holdings"] if h["quantity"] > 0]
        if not holdings:
            logger.info("Position-review: no held instruments to assess")
            return

        symbols = [h["yahoo_symbol"] for h in holdings]
        logger.info("Position-review: starting for %d held instruments", len(holdings))

        rows = await session.execute(
            select(Instrument)
            .where(Instrument.yahoo_symbol.in_(symbols))
            .options(
                selectinload(Instrument.yahoo),
                selectinload(Instrument.earnings_reports),
            )
        )
        by_symbol: dict[str, Instrument] = {inst.yahoo_symbol: inst for inst in rows.scalars().all()}

        macro_row = (
            await session.execute(
                select(MarketMetricsDaily).order_by(MarketMetricsDaily.date.desc()).limit(1)
            )
        ).scalar_one_or_none()

        processed = 0
        skipped_unchanged = 0
        skipped_invalid_thesis = 0
        skipped_no_thesis = 0
        saved = 0
        failed_llm = 0
        failed_error = 0

        # Phase 1 — read everything out of ORM state up front. Phase 2 must not
        # touch ORM instances: a rollback there expires them, and lazy-reloading
        # an expired attribute inside the async session raises MissingGreenlet,
        # killing the whole run instead of skipping one instrument.
        review_inputs: list[tuple[str, int, dict, str]] = []
        for h in holdings:
            sym = h["yahoo_symbol"]
            instrument = by_symbol[sym]

            if not instrument.thesis:
                skipped_no_thesis += 1
                processed += 1
                continue

            try:
                thesis = InstrumentThesisSchema.model_validate(instrument.thesis)
            except ValidationError as e:
                logger.warning("[skip] %s — thesis JSONB invalid: %s", sym, e.errors())
                skipped_invalid_thesis += 1
                processed += 1
                continue

            ctx = build_context(holding=h, instrument=instrument, thesis=thesis, macro=macro_row)
            review_inputs.append((sym, instrument.id, ctx, hash_context(ctx)))

        # Phase 2 — LLM calls and inserts, plain data only.
        for sym, instrument_id, ctx, inputs_hash in review_inputs:
            try:
                latest = (
                    await session.execute(
                        select(PositionReview)
                        .where(PositionReview.instrument_id == instrument_id)
                        .order_by(PositionReview.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

                if latest and latest.inputs_hash == inputs_hash:
                    logger.info("[skip] %s — inputs unchanged since %s", sym, latest.created_at.isoformat())
                    skipped_unchanged += 1
                    processed += 1
                    continue

                payload = generate_assessment(ctx, ticker=sym)
                if payload is None:
                    logger.warning("[skip] %s — Gemini call failed", sym)
                    failed_llm += 1
                    processed += 1
                    continue

                session.add(PositionReview(
                    instrument_id=instrument_id,
                    model=MODEL,
                    inputs_hash=inputs_hash,
                    payload=payload,
                    summary=payload["rationale"],
                ))
                await session.commit()
                saved += 1

                logger.info(
                    "[saved] %s | action=%s strength=%s thesis=%s confidence=%.2f",
                    sym,
                    payload["position_action"],
                    payload["signal_strength"],
                    payload["thesis_status"],
                    payload["confidence"],
                )

            except Exception:
                logger.exception("Position-review failed for %s", sym)
                await session.rollback()
                failed_error += 1

            processed += 1

        logger.info(
            "Position-review complete: %d processed | %d saved | %d skipped (unchanged) | "
            "%d skipped (invalid thesis) | %d skipped (no thesis) | %d failed (llm) | "
            "%d failed (error)",
            processed,
            saved,
            skipped_unchanged,
            skipped_invalid_thesis,
            skipped_no_thesis,
            failed_llm,
            failed_error,
        )


def main() -> None:
    asyncio.run(run_position_reviews())


if __name__ == "__main__":
    main()
