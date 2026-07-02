"""
Detect and repair stock-split discontinuities in PricesDaily and HoldingDaily.

``update_data.py`` only appends new rows, but after a stock split Yahoo
rescales the entire price history retroactively and Trading212 rescales the
position. Rows stored before the split keep the old scale, leaving a
factor-sized cliff that corrupts every price-based signal (RSI, SMA, relative
strength, drawdown, beta) and any view comparing holdings snapshots over time
(e.g. Top Movers).

For each split recorded in ``InstrumentYahoo.splits`` the script compares the
stored values on either side of the split date: an adjusted series crosses the
split at ~1x, an unadjusted one at ~1/factor. Affected PricesDaily symbols get
their full stored date range re-downloaded; affected HoldingDaily rows are
rescaled in place (price / factor, quantity * factor).

Safe to run repeatedly: detection passes once history is consistent, the
price re-download upserts on (symbol, date), and the holdings rescale is
only applied while the boundary still looks unadjusted.

Usage:
    python scripts/fix_split_prices.py            # detect and repair
    python scripts/fix_split_prices.py --dry-run  # detect and report only
"""

import argparse
import math
from datetime import date

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from config import BATCH_SIZE_YF, logger
from data import STOCKS_DELISTED
from models import HoldingDaily, Instrument, InstrumentYahoo, PricesDaily
from scripts.update_data import _update_prices, get_session

# Max days between a split and the nearest holdings snapshot on each side for
# confident auto-repair. Larger gaps mean the position was closed around the
# split, and market drift over the gap makes the ratio test unreliable.
HOLDING_GAP_DAYS = 30


def looks_unadjusted(before: float, after: float, factor: float) -> bool:
    """Check whether a price jump across a split matches the unadjusted scale.

    An adjusted series crosses the split at ~1x, an unadjusted one at
    ~1/factor. Compare in log space so the rule is symmetric and also works
    for reverse splits (factor < 1).
    """
    ratio = after / before
    return abs(math.log(ratio * factor)) < abs(math.log(ratio))


def prices_are_unadjusted(session: Session, symbol: str, split_date: date, factor: float) -> bool:
    """Check whether stored prices before the split still use the pre-split scale."""
    before = session.execute(
        select(PricesDaily.close_price)
        .where(PricesDaily.symbol == symbol, PricesDaily.date < split_date)
        .order_by(PricesDaily.date.desc())
        .limit(1)
    ).scalar()
    after = session.execute(
        select(PricesDaily.close_price)
        .where(PricesDaily.symbol == symbol, PricesDaily.date >= split_date)
        .order_by(PricesDaily.date)
        .limit(1)
    ).scalar()
    # Falsy close = split predates stored history (already consistent), no
    # prices yet, or a junk zero row — nothing to compare against either way.
    if not before or not after:
        return False
    return looks_unadjusted(before, after, factor)


def fix_holdings(
    session: Session, instrument_id: int, symbol: str, split_date: date, factor: float, dry_run: bool
) -> None:
    """Rescale HoldingDaily rows stored before a split T212 never back-adjusted."""
    before = session.execute(
        select(HoldingDaily.date, HoldingDaily.current_price)
        .where(HoldingDaily.instrument_id == instrument_id, HoldingDaily.date < split_date)
        .order_by(HoldingDaily.date.desc())
        .limit(1)
    ).first()
    after = session.execute(
        select(HoldingDaily.date, HoldingDaily.current_price)
        .where(HoldingDaily.instrument_id == instrument_id, HoldingDaily.date >= split_date)
        .order_by(HoldingDaily.date)
        .limit(1)
    ).first()
    if not before or not after or not before.current_price or not after.current_price:
        return
    if not looks_unadjusted(before.current_price, after.current_price, factor):
        return
    if (split_date - before.date).days > HOLDING_GAP_DAYS or (after.date - split_date).days > HOLDING_GAP_DAYS:
        logger.warning(
            "%s: %sx split on %s looks unadjusted in holdings_daily but the position has a "
            "gap around the split (%s -> %s) — review and fix manually",
            symbol, factor, split_date, before.date, after.date,
        )
        return

    logger.warning("%s: rescaling holdings_daily rows before %s for %sx split", symbol, split_date, factor)
    if dry_run:
        return
    session.execute(
        update(HoldingDaily)
        .where(HoldingDaily.instrument_id == instrument_id, HoldingDaily.date < split_date)
        .values(
            # ppl / fx_ppl are absolute account-currency P&L — unaffected by splits.
            quantity=HoldingDaily.quantity * factor,
            avg_price=HoldingDaily.avg_price / factor,
            current_price=HoldingDaily.current_price / factor,
        )
    )


def fix_split_prices(dry_run: bool = False) -> None:
    """Repair unadjusted splits: re-download price history, rescale holdings rows."""
    with get_session() as session:
        rows = session.execute(
            select(Instrument.id, Instrument.yahoo_symbol, InstrumentYahoo.splits)
            .join(InstrumentYahoo, InstrumentYahoo.instrument_id == Instrument.id)
            .where(Instrument.yahoo_symbol.isnot(None))
            .where(Instrument.yahoo_symbol.notin_(STOCKS_DELISTED))
        ).all()

        affected_prices: list[str] = []
        for instrument_id, symbol, splits in rows:
            if not splits:
                continue
            for date_str, factor in splits.items():
                # scrub_for_json may have stored a NaN factor as None. Factors
                # near 1 (share consolidations, spin-off adjustments) are
                # smaller than ordinary daily moves, so the ratio test can't
                # tell them from market noise — and a misadjustment that small
                # wouldn't distort any signal materially anyway.
                if not factor or 0.67 < factor < 1.5:
                    continue
                # Slice covers both "YYYY-MM-DD" keys and legacy full-timestamp keys.
                split_date = date.fromisoformat(date_str[:10])
                if symbol not in affected_prices and prices_are_unadjusted(session, symbol, split_date, factor):
                    logger.warning(
                        "%s: %sx split on %s not reflected in stored price history", symbol, factor, date_str
                    )
                    affected_prices.append(symbol)
                fix_holdings(session, instrument_id, symbol, split_date, factor, dry_run)

        if not affected_prices:
            logger.info("Checked %d instruments — no price discontinuities found", len(rows))
            return
        if dry_run:
            logger.info("Dry run — %d symbols need a history re-download: %s", len(affected_prices), affected_prices)
            return

        # Re-download from each symbol's earliest stored row (not a fixed
        # lookback) so the whole stored range ends up on one price scale.
        min_dates: dict[str, date] = dict(
            session.execute(
                select(PricesDaily.symbol, func.min(PricesDaily.date))
                .where(PricesDaily.symbol.in_(affected_prices))
                .group_by(PricesDaily.symbol)
            ).all()
        )
        for i in range(0, len(affected_prices), BATCH_SIZE_YF):
            batch = affected_prices[i : i + BATCH_SIZE_YF]
            _update_prices(session, batch, min(min_dates[s] for s in batch))
            session.commit()
        logger.info("Re-downloaded full history for %d symbols: %s", len(affected_prices), affected_prices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair split discontinuities in PricesDaily and HoldingDaily")
    parser.add_argument("--dry-run", action="store_true", help="Report affected rows without repairing")
    args = parser.parse_args()
    fix_split_prices(dry_run=args.dry_run)
