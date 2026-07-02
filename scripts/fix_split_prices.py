"""
Detect and repair stock-split discontinuities in PricesDaily.

``update_data.py`` only appends new price rows, but after a stock split Yahoo
rescales the entire price history retroactively. Rows stored before the split
keep the old scale, leaving a factor-sized cliff in the series that corrupts
every price-based signal (RSI, SMA, relative strength, drawdown, beta).

For each split recorded in ``InstrumentYahoo.splits`` the script compares the
stored closes on either side of the split date: an adjusted series crosses the
split at ~1x, an unadjusted one at ~1/factor. Flagged symbols get their full
stored date range re-downloaded and upserted.

Safe to run repeatedly: detection passes once history is consistent, and the
re-download upserts on (symbol, date).

Usage:
    python scripts/fix_split_prices.py            # detect and repair
    python scripts/fix_split_prices.py --dry-run  # detect and report only
"""

import argparse
import math
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import BATCH_SIZE_YF, logger
from data import STOCKS_DELISTED
from models import Instrument, InstrumentYahoo, PricesDaily
from scripts.update_data import _update_prices, get_session


def split_is_unadjusted(session: Session, symbol: str, split_date: date, factor: float) -> bool:
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

    # Adjusted history crosses the split at ~1x, unadjusted at ~1/factor.
    # Pick whichever the observed ratio is closer to in log space, so the
    # rule is symmetric and also works for reverse splits (factor < 1).
    ratio = after / before
    return abs(math.log(ratio * factor)) < abs(math.log(ratio))


def fix_split_prices(dry_run: bool = False) -> None:
    """Find symbols with unadjusted splits and re-download their price history."""
    with get_session() as session:
        rows = session.execute(
            select(Instrument.yahoo_symbol, InstrumentYahoo.splits)
            .join(InstrumentYahoo, InstrumentYahoo.instrument_id == Instrument.id)
            .where(Instrument.yahoo_symbol.isnot(None))
            .where(Instrument.yahoo_symbol.notin_(STOCKS_DELISTED))
        ).all()

        affected: list[str] = []
        for symbol, splits in rows:
            if not splits or symbol in affected:
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
                if split_is_unadjusted(session, symbol, split_date, factor):
                    logger.warning(
                        "%s: %sx split on %s not reflected in stored history", symbol, factor, date_str
                    )
                    affected.append(symbol)
                    break

        if not affected:
            logger.info("Checked %d instruments — no split discontinuities found", len(rows))
            return
        if dry_run:
            logger.info("Dry run — %d symbols need a history re-download: %s", len(affected), affected)
            return

        # Re-download from each symbol's earliest stored row (not a fixed
        # lookback) so the whole stored range ends up on one price scale.
        min_dates: dict[str, date] = dict(
            session.execute(
                select(PricesDaily.symbol, func.min(PricesDaily.date))
                .where(PricesDaily.symbol.in_(affected))
                .group_by(PricesDaily.symbol)
            ).all()
        )
        for i in range(0, len(affected), BATCH_SIZE_YF):
            batch = affected[i : i + BATCH_SIZE_YF]
            _update_prices(session, batch, min(min_dates[s] for s in batch))
            session.commit()
        logger.info("Re-downloaded full history for %d symbols: %s", len(affected), affected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair split discontinuities in PricesDaily")
    parser.add_argument("--dry-run", action="store_true", help="Report affected symbols without re-downloading")
    args = parser.parse_args()
    fix_split_prices(dry_run=args.dry_run)
