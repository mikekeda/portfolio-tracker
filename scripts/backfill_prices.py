"""
scripts/backfill_prices.py
==========================
Extend price history for selected symbols by refetching a longer window. Run from project root:

    python scripts/backfill_prices.py                       # glidepath universe, 25 years
    python scripts/backfill_prices.py --years 30
    python scripts/backfill_prices.py --symbols MSFT,NVDA,CNDX.L
    python scripts/backfill_prices.py --all --years 25      # every tracked instrument
    python scripts/backfill_prices.py --dry-run

`update_data.py` seeds a new ticker with HISTORY_YEARS of prices and thereafter
only tops up from the newest stored row, so raising that constant does nothing
for symbols already in the table. This script is the one-off that reaches back.

It refetches the whole window rather than only the missing head: Yahoo restates
adjusted closes after every split and dividend, so appending old rows to rows
fetched years ago would splice two different adjustment bases together and leave
a step in the series. `_update_prices` upserts, so the overlap is rewritten.

Yahoo returns nothing before a symbol's first trade, so asking for more years
than a fund has existed is harmless — the fund simply starts where it started.
A symbol older than --years stops at that window, not at its inception.
"""

import argparse
from datetime import datetime, timedelta

from sqlalchemy import func, select

from config import BATCH_SIZE_YF, HISTORY_YEARS, TIMEZONE, logger
from data import STOCKS_DELISTED
from models import Instrument, PricesDaily
from scripts.optimize_glidepath import ETF_UNIVERSE, STOCK_UNIVERSE
from scripts.update_data import _update_prices, get_session

# Reaching back rewrites every adjusted close in the window, so the split repair
# in fix_split_prices.py is worth a run afterwards.
FOLLOW_UP = "scripts/check_data_integrity.py, then scripts/fix_split_prices.py"


def coverage(session, symbols: list[str]) -> dict[str, tuple]:
    """First date, last date and row count per symbol."""
    rows = session.execute(
        select(PricesDaily.symbol, func.min(PricesDaily.date), func.max(PricesDaily.date), func.count())
        .where(PricesDaily.symbol.in_(symbols))
        .group_by(PricesDaily.symbol)
    ).all()
    return {symbol: (first, last, count) for symbol, first, last, count in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill price history for selected symbols")
    parser.add_argument("--years", type=int, default=25, help=f"years to reach back (seed default {HISTORY_YEARS})")
    parser.add_argument("--symbols", help="comma-separated yahoo symbols; defaults to the glidepath universe")
    parser.add_argument("--all", action="store_true", help="every instrument with a yahoo_symbol")
    parser.add_argument("--dry-run", action="store_true", help="report current coverage and exit")
    args = parser.parse_args()

    if args.years < 1:
        parser.error("--years must be at least 1")
    if args.symbols and args.all:
        parser.error("--symbols and --all are mutually exclusive")

    start = datetime.now(TIMEZONE).date() - timedelta(days=round(args.years * 365.25))
    failed: list[str] = []

    with get_session() as session:
        if args.all:
            # Same exclusion as update_prices: a delisted line returns nothing, or
            # worse returns junk (PJXC.DE's frozen adj_close).
            symbols = sorted(
                {s for s in session.scalars(select(Instrument.yahoo_symbol)).all() if s and s not in STOCKS_DELISTED}
            )
        elif args.symbols:
            symbols = list(dict.fromkeys(s.strip() for s in args.symbols.split(",") if s.strip()))
        else:
            symbols = sorted({**ETF_UNIVERSE, **STOCK_UNIVERSE})

        before = coverage(session, symbols)
        logger.info("Backfilling %s symbols from %s", len(symbols), start)
        if args.dry_run:
            for symbol in symbols:
                first, last, count = before.get(symbol, (None, None, 0))
                logger.info("  %-10s %s .. %s  (%s rows)", symbol, first, last, count)
            return

        # Commit per batch: the rollback inside _update_prices discards the whole
        # open transaction, taking every earlier batch with it.
        for offset in range(0, len(symbols), BATCH_SIZE_YF):
            batch = symbols[offset : offset + BATCH_SIZE_YF]
            stored = _update_prices(session, batch, start)
            if stored:
                session.commit()
            failed.extend(s for s in batch if s not in stored)

    with get_session() as session:
        after = coverage(session, symbols)

    gained = 0
    logger.info("%-10s %-12s %-12s %s", "symbol", "was", "now", "rows added")
    for symbol in symbols:
        old_first, _, old_count = before.get(symbol, (None, None, 0))
        new_first, _, new_count = after.get(symbol, (None, None, 0))
        gained += new_count - old_count
        logger.info("%-10s %-12s %-12s %+d", symbol, old_first or "-", new_first or "-", new_count - old_count)
    logger.info("Added %s rows across %s symbols. Next: %s", gained, len(symbols) - len(failed), FOLLOW_UP)
    if failed:
        logger.error("%s symbols returned no data from Yahoo or were rolled back: %s", len(failed), failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
