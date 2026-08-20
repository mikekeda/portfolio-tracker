"""
Repair 13F quarters that the normal scrape cannot reach.

`main()` in scrape_13f.py fetches one quarter once a manager has enough history, so damage
older than the latest filing is never revisited. Two known cases:

- Berkshire 2025-03-31 was stored as the 13F-HR/A `0000950123-25-008361` with 0 holdings and
  total_value 0, replacing the original 13F-HR `0000950123-25-005701` (110 rows, $258.7bn).
  The amendment is `amendmentType = NEW HOLDINGS` — a 4-row delta of confidential-treatment
  positions, not a restatement.
- Pershing Square 2026-03-31 predates the split across CIKs 1336528 and 2026053, so the stored
  filing holds only the first CIK's 11 positions.

Forces the full fetch window so scrape_13f's merge paths rebuild the quarter, then saves.
Idempotent: re-running rewrites the same rows.

Run --dry-run first and check the holding counts: a quarter reported far below its expected
size means its original has aged out of the submissions `recent` window and only an amendment
was reachable. Berkshire 2025-03-31 should report 114 holdings.
"""

import argparse

from sqlalchemy import func, select

from config import logger
from models import Form13FFiling, Form13FHolding, Form13FManager
from scripts.scrape_13f import INVESTORS, ScrapedFiling, _save_to_db, investor_ciks, scrape_investor
from scripts.update_data import get_session

# Deep enough to reach any quarter still damaged; SEC's `filings.recent` covers far more.
DEFAULT_QUARTERS = 8


def _damaged_manager_names(session) -> set[str]:
    """Managers holding a filing with no holdings or no total value."""
    rows = session.execute(
        select(Form13FManager.name, Form13FFiling.report_date)
        .join(Form13FFiling, Form13FFiling.manager_id == Form13FManager.id)
        .outerjoin(Form13FHolding, Form13FHolding.filing_id == Form13FFiling.id)
        .group_by(Form13FManager.name, Form13FFiling.report_date, Form13FFiling.total_value)
        .having((func.count(Form13FHolding.id) == 0) | (Form13FFiling.total_value == 0))
    ).all()
    for name, report_date in rows:
        logger.info("Empty filing: %s %s", name, report_date)
    return {name for name, _ in rows}


def fix_13f_backfill(only: str | None = None, quarters: int = DEFAULT_QUARTERS, dry_run: bool = False) -> int:
    """Re-scrape and rewrite damaged or multi-CIK 13F quarters. Returns filings saved."""
    with get_session() as session:
        damaged = _damaged_manager_names(session)
        targets = [
            inv
            for inv in INVESTORS
            if (only is None or inv["name"] == only) and (str(inv["name"]) in damaged or len(investor_ciks(inv)) > 1)
        ]
        if only and not targets:
            raise SystemExit(f"'{only}' is not a tracked investor, or needs no repair.")
        if not targets:
            logger.info("Nothing to repair.")
            return 0

        logger.info("Repairing %d investor(s): %s", len(targets), ", ".join(str(t["name"]) for t in targets))
        results: list[ScrapedFiling] = []
        for inv in targets:
            try:
                # Empty existing_dates forces the full window instead of the latest quarter only.
                results.extend(scrape_investor(inv, existing_dates=set(), default_quarters=quarters))
            except Exception as e:
                logger.exception("Error scraping %s: %s", inv["name"], e)

        if dry_run:
            for r in results:
                logger.info(
                    "DRY RUN %s %s: %d holdings, $%s",
                    r["investor"],
                    r["reportDate"],
                    r["holdingsCount"],
                    f"{r['totalValue']:,}",
                )
            return 0

        if results:
            _save_to_db(session, results)
            logger.info("Saved %d filings", len(results))
        return len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair damaged or multi-CIK 13F quarters")
    parser.add_argument("--investor", help="Repair a single investor by name", default=None)
    parser.add_argument(
        "--quarters", type=int, default=DEFAULT_QUARTERS, help=f"Quarters to fetch (default: {DEFAULT_QUARTERS})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report without writing")
    args = parser.parse_args()
    fix_13f_backfill(only=args.investor, quarters=args.quarters, dry_run=args.dry_run)
