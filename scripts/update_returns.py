"""
scripts/update_returns.py
=========================
Fast incremental MWRR/TWRR calculation.

Algorithm:
  TWRR: chain-links daily sub-period return factors, annualises over trading days
  MWRR: IRR of all net deposits/withdrawals + terminal portfolio value, annualised

Usage:
    python scripts/update_returns.py            # fill in missing rows only (fast, daily)
    python scripts/update_returns.py --rebuild  # recalculate and overwrite every row
"""

import argparse
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from numpy_financial import irr
from sqlalchemy import case, func, select, update

from config import SPY, logger
from models import PortfolioDaily, TransactionAction, TransactionHistory, PricesDaily
from scripts.update_data import get_session


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (2 queries total, no per-holding price lookups)
# ──────────────────────────────────────────────────────────────────────────────


def load_portfolio_snapshots(session) -> list[tuple[date, float]]:
    """Return [(date, value), ...] ordered chronologically for all rows with a value."""
    rows = session.execute(
        select(PortfolioDaily.date, PortfolioDaily.value)
        .where(PortfolioDaily.value.isnot(None))
        .order_by(PortfolioDaily.date)
    ).all()
    return [(r.date, r.value) for r in rows]


def load_daily_cash_flows(session) -> dict[date, float]:
    """
    Aggregate deposits and withdrawals from TransactionHistory by calendar date.
    Returns {date: net_flow} where positive = net deposit into the portfolio.
    """
    rows = session.execute(
        select(
            func.date(TransactionHistory.timestamp).label("day"),
            func.sum(
                case(
                    (TransactionHistory.action == TransactionAction.DEPOSIT, TransactionHistory.total),
                    else_=-TransactionHistory.total,
                )
            ).label("net_flow"),
        )
        .where(TransactionHistory.action.in_([TransactionAction.DEPOSIT, TransactionAction.WITHDRAWAL]))
        .group_by(func.date(TransactionHistory.timestamp))
    ).all()
    return {r.day: float(r.net_flow) for r in rows}


# ──────────────────────────────────────────────────────────────────────────────
# Return calculations (pure Python / numpy, no further DB access)
# ──────────────────────────────────────────────────────────────────────────────


def compute_returns(
    snapshots: list[tuple[date, float]],
    cash_flows: dict[date, float],
    target_dates: Optional[set[date]] = None,
) -> list[tuple[date, float, float]]:
    """
    Compute cumulative annualised MWRR and TWRR.

    Iterates over every calendar day between the first cash flow and the last snapshot,
    appending zeros on days with no flow. This ensures irr() sees uniform 1-day periods
    and that TWRR correctly accounts for cash flows between snapshot dates.

    NOTE: PortfolioDaily is expected to have a row for every calendar day (including
    weekends), which is guaranteed by backfill_portfolio_daily.py using timedelta(days=1).
    If the backfill ever skips non-trading days, this function remains correct because
    accumulated_cf collects all flows between consecutive snapshots.

    target_dates: if provided, irr() is only called for those dates; state is still
                  accumulated for every day. Pass None to compute for all dates (rebuild).
    """
    if not snapshots:
        return []
    results: list[tuple[date, float, float]] = []

    snap_dict = dict(snapshots)
    first_snapshot_date = snapshots[0][0]
    last_date = snapshots[-1][0]

    # Extend back if any cash flow pre-dates the first snapshot
    first_date = first_snapshot_date
    if cash_flows and min(cash_flows.keys()) < first_snapshot_date:
        first_date = min(cash_flows.keys())

    twrr_factors: list[float] = []
    mwrr_daily_flows: list[float] = []
    last_value: Optional[float] = None
    accumulated_cf = 0.0
    current_date = first_date
    while current_date <= last_date:
        cf = cash_flows.get(current_date, 0.0)

        # 1. MWRR requires a strict true-daily sequence so irr() correctly represents a daily interval
        mwrr_daily_flows.append(-cf)

        # 2. TWRR processing
        accumulated_cf += cf
        if current_date in snap_dict:
            value = snap_dict[current_date]

            # Determine sub-period return utilizing ALL cash flows accumulated since the last snapshot
            if last_value is not None:
                denominator = last_value + accumulated_cf
                if denominator > 0 and value > 0:
                    factor = value / denominator
                    if not np.isnan(factor) and factor > 0:
                        twrr_factors.append(factor)

            last_value = value
            accumulated_cf = 0.0  # Reset ready for the next period
            # 3. Perform the actual rate calculation if this date needs updating
            if target_dates is None or current_date in target_dates:
                # MWRR calculation
                annual_mwrr = 0.0
                irr_flows = mwrr_daily_flows.copy()
                irr_flows[-1] += value  # terminal value back to investor
                try:
                    daily_rate = irr(irr_flows)
                    if daily_rate is not None and not np.isnan(daily_rate) and daily_rate > -1:
                        annual_mwrr = ((1 + daily_rate) ** 365.25 - 1) * 100.0
                except (ValueError, RuntimeError):
                    logger.debug("IRR did not converge for %s, storing 0.0", current_date)
                # TWRR calculation
                annual_twrr = 0.0
                # True calendar days span tracked precisely from the first portfolio snapshot
                calendar_days = (current_date - first_snapshot_date).days
                if twrr_factors and calendar_days > 0:
                    product = float(np.nanprod(twrr_factors))
                    if product > 0:
                        annual_twrr = ((product ** (365.25 / calendar_days)) - 1) * 100.0
                results.append((current_date, annual_mwrr, annual_twrr))
        current_date += timedelta(days=1)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def update_returns(rebuild: bool = False) -> None:
    logger.info("Updating MWRR/TWRR returns (rebuild=%s)...", rebuild)

    with get_session() as session:
        # 1. Two fast queries — no price lookups, no per-holding joins
        snapshots = load_portfolio_snapshots(session)
        cash_flows = load_daily_cash_flows(session)

        if not snapshots:
            logger.warning("No PortfolioDaily rows found with a value. Run backfill first.")
            return

        logger.info("Loaded %d portfolio snapshots, %d cash-flow dates", len(snapshots), len(cash_flows))

        # 2. Determine which dates need updating
        if not rebuild:
            needs_update: set[date] = set()
            for row in session.execute(
                select(PortfolioDaily.date).where((PortfolioDaily.mwrr.is_(None)) | (PortfolioDaily.twrr.is_(None)))
            ).all():
                needs_update.add(row.date)

            if not needs_update:
                logger.info("All rows up to date — nothing to do.")
                return
            logger.info("%d rows need updating", len(needs_update))

        # 3. Compute returns — only run irr() for dates that actually need updating
        computed = compute_returns(snapshots, cash_flows, target_dates=None if rebuild else needs_update)

        # 4. Write results to DB (sequential UPDATEs, single commit at the end)
        bench_prices = session.execute(
            select(PricesDaily.date, PricesDaily.adj_close_price)
            .where(PricesDaily.symbol == SPY)
            .where(PricesDaily.date >= snapshots[0][0])
            .order_by(PricesDaily.date)
        ).all()

        bench_df = pd.DataFrame(bench_prices, columns=["date", "price"]).set_index("date") if bench_prices else None

        betas = dict(session.execute(
            select(PortfolioDaily.date, PortfolioDaily.beta)
            .where(PortfolioDaily.date.in_([d for d, _, _ in computed]))
        ).all())

        inception = snapshots[0][0]
        bench_start_price = bench_df.iloc[bench_df.index.get_indexer([inception], method="nearest")[0]]["price"] if bench_df is not None else None

        updates = []
        for snap_date, mwrr, twrr in computed:
            alpha = None
            beta = betas.get(snap_date)

            if bench_start_price and beta is not None and snap_date > inception:
                end_price = bench_df.iloc[bench_df.index.get_indexer([snap_date], method="nearest")[0]]["price"]
                days = (snap_date - inception).days
                bench_annual = ((end_price / bench_start_price) ** (365.25 / days)) - 1
                alpha = float(((twrr / 100.0) - (0.04 + beta * (bench_annual - 0.04))) * 100.0)

            updates.append({"_date": snap_date, "_mwrr": float(mwrr), "_twrr": float(twrr), "_alpha": alpha})

        written = len(updates)

        for row in updates:
            update_values = {"mwrr": row["_mwrr"], "twrr": row["_twrr"]}
            if row["_alpha"] is not None:
                update_values["jensens_alpha"] = row["_alpha"]

            session.execute(
                update(PortfolioDaily)
                .where(PortfolioDaily.date == row["_date"])
                .values(**update_values)
            )
        if updates:
            session.commit()
            last = updates[-1]
            logger.info(
                "Done. Updated %d rows. Latest (%s): MWRR=%.2f%%, TWRR=%.2f%%, Alpha=%s",
                written,
                last["_date"],
                last["_mwrr"],
                last["_twrr"],
                f"{last['_alpha']:.2f}%" if last["_alpha"] is not None else "None",
            )
        else:
            logger.info("Done. No rows updated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast incremental MWRR/TWRR calculation")
    parser.add_argument("--rebuild", action="store_true", help="Recalculate and overwrite all rows")
    args = parser.parse_args()
    update_returns(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
