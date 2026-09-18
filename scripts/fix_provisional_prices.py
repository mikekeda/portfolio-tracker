"""
Refetch price history that still holds intraday bars recorded as the session close.

``update_data.py`` used to store whatever Yahoo returned, so a run during market
hours could persist an unfinished bar as the day's close, and the next runs never
revisited it. It now stops before the current UTC day and refetches a recent
window, which heals anything inside that window but not older rows.

A stored bar is potentially provisional when its ``updated_at`` falls on or before
its own date. This conservative check also includes valid post-close writes. Affected
symbols are re-downloaded over their whole stored range, as after a split, so
the series ends up on one adjustment basis rather than gaining a splice.

Usage:
    python scripts/fix_provisional_prices.py --dry-run
    python scripts/fix_provisional_prices.py
"""

import argparse
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import Date, cast, func, select

from config import BATCH_SIZE_YF, logger
from data import STOCKS_DELISTED
from models import PricesDaily
from scripts.update_data import PRICE_OVERLAP_DAYS, _update_prices, get_session

FOLLOW_UP = "scripts/check_data_integrity.py, then scripts/fix_split_prices.py"


def provisional_symbols(session, cutoff: date) -> dict[str, date]:
    """Earliest stored date per symbol holding a provisional bar older than `cutoff`."""
    rows = session.execute(
        select(PricesDaily.symbol, func.min(PricesDaily.date))
        .where(
            PricesDaily.symbol.notin_(STOCKS_DELISTED),
            PricesDaily.date < cutoff,
            cast(PricesDaily.updated_at, Date) <= PricesDaily.date,
        )
        .group_by(PricesDaily.symbol)
    ).all()
    if not rows:
        return {}
    starts = dict(
        session.execute(
            select(PricesDaily.symbol, func.min(PricesDaily.date))
            .where(PricesDaily.symbol.in_([symbol for symbol, _ in rows]))
            .group_by(PricesDaily.symbol)
        ).all()
    )
    for symbol, first_provisional in sorted(rows):
        logger.info("%s: provisional bars from %s; refetching from %s", symbol, first_provisional, starts[symbol])
    return starts


def fix_provisional_prices(dry_run: bool = False) -> int:
    """Re-download suspect history and fail if any selected bars remain unrepaired."""
    # Bars inside the daily overlap window are refetched by the next update run anyway.
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=PRICE_OVERLAP_DAYS)
    with get_session() as session:
        starts = provisional_symbols(session, cutoff)
        if not starts:
            logger.info("No provisional bars older than %s", cutoff)
            return 0
        if dry_run:
            logger.info("Dry run — %d symbols need a history re-download", len(starts))
            return 0

        symbols = sorted(starts)
        failed = set()
        for i in range(0, len(symbols), BATCH_SIZE_YF):
            batch = symbols[i : i + BATCH_SIZE_YF]
            stored = _update_prices(session, batch, min(starts[s] for s in batch))
            failed.update(set(batch) - stored)
            session.commit()
        # Receiving some rows is not proof Yahoo returned the older bars selected for repair.
        remaining = provisional_symbols(session, cutoff)
        failed.update(set(symbols) & remaining.keys())
        if failed:
            raise RuntimeError("Price repair incomplete; missing downloads or unrepaired bars: " + ", ".join(sorted(failed)))
        logger.info("Re-downloaded full history for %d symbols. Follow up with %s", len(symbols), FOLLOW_UP)
        return len(symbols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refetch price history containing intraday bars stored as closes")
    parser.add_argument("--dry-run", action="store_true", help="Report affected symbols without repairing")
    args = parser.parse_args()
    fix_provisional_prices(dry_run=args.dry_run)
