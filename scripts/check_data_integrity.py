"""
Read-only data-integrity audit across prices, instruments, holdings and portfolio.

Bundles every check from the July 2026 data audit that found real corruption:
frozen tickers (BK/TLNE renames), NaN closes on thin European lines, zeroed-out
adjusted prices (PJXC.DE), and the PE-scraper starvation that froze screener
inputs for two months. Each failing check prints WARNING lines with sample rows
and a "fix:" hint, so the output doubles as a triage list. Nothing is repaired
automatically — most findings need judgment (a suspicious move can be a real
crash; an "orphan" symbol can be a rename mid-migration).

Usage:
    python scripts/check_data_integrity.py                # run everything
    python scripts/check_data_integrity.py --check stale_prices
"""

import argparse

from sqlalchemy import text

from config import BENCHES, VIX, logger
from data import QQQ, SP500, STOCKS_ALIASES, STOCKS_DELISTED
from scripts.update_data import get_session

# Findings shown per check before truncating — keeps logs readable when a
# systemic issue (e.g. a broken writer) flags hundreds of rows at once.
MAX_ROWS_SHOWN = 15

STALE_PRICE_DAYS = 7      # > a week without prices on a live symbol = frozen line
STALE_PROFILE_DAYS = 3    # Yahoo profile refresh budget covers the universe daily
STALE_METRICS_DAYS = 2    # held instruments should get metrics every run
# SEC 10-Q deadline is 40-45 days after quarter end, so a PR placeholder older
# than that without a canonical row means the supersede path or a route failed.
STALE_PR_DAYS = 45
# A successful Wisesheets scrape leaves a year-end forward-estimate key, so a
# newest pes key this far in the past means the scrape has failed for months.
STALE_PE_DAYS = 120
PORTFOLIO_DIFF_PCT = 1.0  # T212 total vs holdings sum; small FX drift is normal
# Same bounds as the split detector: an unadjusted 2:1 split shows as ~0.5x.
MOVE_UPPER, MOVE_LOWER = 1.8, 0.55

# (name, description, fix hint, SQL) — every check is a plain read-only query
# so it can be pasted into psql/pgAdmin unchanged when digging into a finding.
CHECKS: list[tuple[str, str, str, str]] = [
    (
        "junk_prices",
        "corrupt price rows (<= 0, NaN, or adj_close above close)",
        "NaN/zero rows: DELETE them (never real prices; Yahoo pads non-traded days). "
        "adj > close on a dividend payer means the whole history predates a re-adjustment: "
        "delete the symbol's rows and let update_data re-download. If adj is ~0 across "
        "years (PJXC.DE case), delete that range — re-downloading returns the same junk.",
        """
        SELECT symbol, count(*) AS bad_rows, min(date) AS first_bad, max(date) AS last_bad
        FROM prices_daily
        WHERE close_price <= 0 OR adj_close_price <= 0
           OR close_price = 'NaN'::float8 OR adj_close_price = 'NaN'::float8
           OR adj_close_price > close_price * 1.001
        GROUP BY symbol
        ORDER BY bad_rows DESC
        """,
    ),
    (
        "stale_prices",
        f"live symbols with no prices for > {STALE_PRICE_DAYS} days (rename/delisting?)",
        "Probe with yf.download(sym). Empty through today = check the news: renamed ticker "
        "-> add to STOCKS_ALIASES + UPDATE instruments.yahoo_symbol + rename prices_daily "
        "rows (BK->BNY playbook); dead line -> add to STOCKS_DELISTED. Rows returned but "
        "old = thin line, leave it a week before delisting.",
        f"""
        SELECT p.symbol, max(p.date) AS last_price,
               p.symbol = ANY(:alias_old) AS old_alias_symbol
        FROM prices_daily p
        JOIN instruments i ON i.yahoo_symbol = p.symbol
        WHERE p.symbol != ALL(:delisted)
        GROUP BY p.symbol
        HAVING max(p.date) < CURRENT_DATE - {STALE_PRICE_DAYS}
        ORDER BY max(p.date)
        """,
    ),
    (
        "orphan_prices",
        "price rows for symbols with no instrument (excluding SP500/QQQ/bench/VIX)",
        "old_alias_symbol = rows under a pre-rename ticker: rename them to the new symbol "
        "(UPDATE prices_daily SET symbol = <new>), don't delete. renamed_ticker = some "
        "instrument likely still points at the old symbol: fix instruments.yahoo_symbol. "
        "Neither flag = true orphan: DELETE FROM prices_daily WHERE symbol = ... "
        "(see Jul 2026 cleanup that removed 369k rows across 165 symbols).",
        """
        SELECT p.symbol, count(*) AS rows, max(p.date) AS last_price,
               p.symbol = ANY(:alias_old) AS old_alias_symbol,
               p.symbol = ANY(:alias_new) AS renamed_ticker
        FROM prices_daily p
        LEFT JOIN instruments i ON i.yahoo_symbol = p.symbol
        WHERE i.id IS NULL AND p.symbol != ALL(:allowed)
        GROUP BY p.symbol
        ORDER BY rows DESC
        """,
    ),
    (
        "price_gaps",
        "holes in price series longer than any holiday cluster (> 6 days)",
        "Probe the window with yf.download(sym, start=prev_date, end=date). Rows with "
        "volume > 0 = real hole: refill with _update_prices(session, [sym], prev_date). "
        "Zero-volume or empty = the line didn't trade (NZM2.DE case), nothing to fix — "
        "charts bridge the gap via connectNulls.",
        """
        SELECT symbol, prev_date, date, date - prev_date AS gap_days
        FROM (SELECT symbol, date,
                     lag(date) OVER (PARTITION BY symbol ORDER BY date) AS prev_date
              FROM prices_daily) t
        WHERE date - prev_date > 6
        ORDER BY gap_days DESC
        """,
    ),
    (
        "suspicious_moves",
        "1-day moves beyond split-like bounds in the last 30 days (missed split?)",
        "Check the company news for that date. Real crash/spike (biotech, leveraged ETF) "
        "= no action. Split Yahoo didn't record = run scripts/fix_split_prices.py; if it "
        "doesn't detect it (e.g. breadth-only symbol with no instrument), re-download the "
        "stored range in place: _update_prices(session, [sym], min_stored_date) — don't "
        "delete first, breadth-only symbols are never re-added from scratch.",
        f"""
        SELECT symbol, date, round(prev_close::numeric, 2) AS prev_close,
               round(adj_close_price::numeric, 2) AS close,
               round((100 * (adj_close_price / prev_close - 1))::numeric) AS pct_move
        FROM (SELECT symbol, date, adj_close_price,
                     lag(adj_close_price) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
              FROM prices_daily
              WHERE date > CURRENT_DATE - 35) t
        WHERE date > CURRENT_DATE - 30 AND prev_close > 0
          -- Postgres treats NaN as greater than any number, so NaN closes
          -- (junk_prices territory) would flag here as fake spikes.
          AND adj_close_price != 'NaN'::float8 AND prev_close != 'NaN'::float8
          AND (adj_close_price / prev_close > {MOVE_UPPER}
               OR adj_close_price / prev_close < {MOVE_LOWER})
        ORDER BY date DESC
        """,
    ),
    (
        "duplicate_yahoo",
        "multiple instruments sharing one yahoo_symbol (double counting)",
        "Two T212 lines (e.g. USD + EUR listing) mapped to one Yahoo symbol. Keep the line "
        "you actually hold; point the other at its real Yahoo listing or NULL its "
        "yahoo_symbol so it stops being tracked.",
        """
        SELECT yahoo_symbol, count(*) AS instruments, array_agg(t212_code) AS t212_codes
        FROM instruments
        WHERE yahoo_symbol IS NOT NULL
        GROUP BY yahoo_symbol
        HAVING count(*) > 1
        """,
    ),
    (
        "held_without_prices",
        "current holdings with no price history at all",
        "yahoo_symbol is wrong or newly aliased. Fix the symbol (STOCKS_ALIASES + UPDATE "
        "instruments.yahoo_symbol), then the next update_data run downloads full history "
        "automatically (new-ticker path, 10y lookback).",
        """
        SELECT i.t212_code, i.yahoo_symbol
        FROM instruments i
        JOIN holdings_daily h ON h.instrument_id = i.id AND h.date = (SELECT max(date) FROM holdings_daily)
        LEFT JOIN prices_daily p ON p.symbol = i.yahoo_symbol
        WHERE p.symbol IS NULL
        GROUP BY i.t212_code, i.yahoo_symbol
        """,
    ),
    (
        "stale_profiles",
        f"Yahoo profiles not fetched for > {STALE_PROFILE_DAYS} days (screener inputs stale)",
        "A handful = normal queue lag; a persistent block = the refresh budget isn't "
        "covering the universe (raise YAHOO_UPDATE_LIMIT) or fetches keep failing (look "
        "for 'Empty Yahoo info' warnings in update_data logs). To force a refetch: "
        "UPDATE instruments_yahoo SET profile_fetched_at = NULL WHERE instrument_id IN (...).",
        f"""
        SELECT i.yahoo_symbol, y.profile_fetched_at,
               i.id IN (SELECT instrument_id FROM holdings_daily
                        WHERE date = (SELECT max(date) FROM holdings_daily)) AS held
        FROM instruments i
        JOIN instruments_yahoo y ON y.instrument_id = i.id
        WHERE i.yahoo_symbol != ALL(:delisted)
          AND (y.profile_fetched_at IS NULL
               OR y.profile_fetched_at < now() - interval '{STALE_PROFILE_DAYS} days')
        ORDER BY held DESC, y.profile_fetched_at NULLS FIRST
        """,
    ),
    (
        "blob_price_drift",
        "cached Yahoo info price > 15% from latest close (stale blob)",
        "If profile_fetched_at is old too, force a refetch (see stale_profiles). If the "
        "blob is fresh, the drift is real (big move since the last fetch) or the "
        "instrument's yahoo_symbol points at a different listing/currency than its "
        "prices_daily rows — compare the two quotes manually.",
        """
        SELECT i.yahoo_symbol,
               (y.info->>'regularMarketPrice')::float AS blob_price,
               round(p.close_price::numeric, 2) AS close_price,
               round(abs((y.info->>'regularMarketPrice')::float / p.close_price - 1)::numeric * 100) AS drift_pct
        FROM instruments i
        JOIN instruments_yahoo y ON y.instrument_id = i.id
        JOIN LATERAL (SELECT close_price FROM prices_daily
                      WHERE symbol = i.yahoo_symbol ORDER BY date DESC LIMIT 1) p ON true
        WHERE y.info ? 'regularMarketPrice' AND p.close_price > 0
          AND p.close_price != 'NaN'::float8
          AND abs((y.info->>'regularMarketPrice')::float / p.close_price - 1) > 0.15
        ORDER BY drift_pct DESC
        """,
    ),
    (
        "stale_metrics",
        f"held instruments without metrics rows for > {STALE_METRICS_DAYS} days",
        "Metrics are written only when update_data fetches the profile, so this usually "
        "means the instrument isn't being fetched — same remedies as stale_profiles. All "
        "held stuck on one shared date = a broken writer (the Apr-Jun 2026 starvation "
        "signature); check what last touched the fetch queue logic.",
        f"""
        SELECT i.yahoo_symbol, max(m.date) AS latest_metrics
        FROM instruments i
        JOIN holdings_daily h ON h.instrument_id = i.id AND h.date = (SELECT max(date) FROM holdings_daily)
        LEFT JOIN instruments_metrics_daily m ON m.instrument_id = i.id
        GROUP BY i.yahoo_symbol
        HAVING max(m.date) < CURRENT_DATE - {STALE_METRICS_DAYS} OR max(m.date) IS NULL
        ORDER BY latest_metrics NULLS FIRST
        """,
    ),
    (
        "portfolio_gaps",
        "missing PortfolioDaily snapshot days (> 3-day hole)",
        "update_data didn't run those days (host down, Celery dead). Recompute snapshots "
        "with scripts/backfill_portfolio_daily.py, then scripts/update_returns.py --rebuild "
        "so MWRR/TWRR incorporate the filled days.",
        """
        SELECT prev_date, date, date - prev_date AS gap_days
        FROM (SELECT date, lag(date) OVER (ORDER BY date) AS prev_date
              FROM portfolio_daily) t
        WHERE date - prev_date > 3
        ORDER BY date DESC
        """,
    ),
    (
        "portfolio_vs_holdings",
        f"snapshot value diverging > {PORTFOLIO_DIFF_PCT}% from GBP sum of holdings",
        "Single old day with a big diff = one corrupted holdings row: compare per-holding "
        "quantity * current_price against adjacent days to find it, then fix that row "
        "(often a GBX/GBP or pre-split scale error). Persistent small diffs = rate or "
        "timing skew, usually fine.",
        f"""
        WITH hs AS (
            SELECT h.date,
                   sum(h.quantity * h.current_price *
                       CASE i.currency
                           WHEN 'GBP' THEN 1.0
                           WHEN 'GBX' THEN 0.01
                           ELSE r.rate
                       END) AS holdings_sum
            FROM holdings_daily h
            JOIN instruments i ON i.id = h.instrument_id
            LEFT JOIN LATERAL (
                SELECT rate FROM currency_rates_daily cr
                WHERE cr.to_currency = 'GBP' AND cr.from_currency = i.currency
                  AND cr.date <= h.date
                ORDER BY cr.date DESC LIMIT 1
            ) r ON true
            GROUP BY h.date
        )
        SELECT p.date, round(p.value::numeric, 2) AS portfolio_value,
               round(hs.holdings_sum::numeric, 2) AS holdings_sum,
               round((100 * (hs.holdings_sum - p.value) / p.value)::numeric, 2) AS diff_pct
        FROM portfolio_daily p
        JOIN hs USING (date)
        WHERE p.value > 0
          AND abs(p.value - hs.holdings_sum) / p.value > {PORTFOLIO_DIFF_PCT / 100}
        ORDER BY p.date DESC
        """,
    ),
    (
        "holdings_gaps",
        "holes inside a position's daily snapshots (5-90 days)",
        "Cross-check against transaction_history: if the position was open through the "
        "gap, snapshots are missing (update_data didn't run) — backfill_portfolio_daily "
        "covers portfolio totals but per-holding rows stay lost; note it and move on. "
        "If it was closed and reopened, the gap is real and correct.",
        """
        SELECT i.yahoo_symbol, t.prev_date, t.date, t.date - t.prev_date AS gap_days
        FROM (SELECT instrument_id, date,
                     lag(date) OVER (PARTITION BY instrument_id ORDER BY date) AS prev_date
              FROM holdings_daily) t
        JOIN instruments i ON i.id = t.instrument_id
        WHERE t.date - t.prev_date BETWEEN 5 AND 90
        ORDER BY gap_days DESC
        """,
    ),
    (
        "currency_rates",
        "GBP conversion rates stale (> 5 days) or with > 4-day holes",
        "Stale = the rate API call in update_data is failing; check run logs. Holes: "
        "scripts/backfill_currency_rates.py refills history. Every non-GBP valuation "
        "silently uses the last known rate until fixed.",
        """
        SELECT from_currency, 'stale' AS issue, max(date)::text AS detail
        FROM currency_rates_daily
        WHERE to_currency = 'GBP'
        GROUP BY from_currency
        HAVING max(date) < CURRENT_DATE - 5
        UNION ALL
        SELECT from_currency, 'gap',
               prev_date::text || ' -> ' || date::text
        FROM (SELECT from_currency, date,
                     lag(date) OVER (PARTITION BY from_currency ORDER BY date) AS prev_date
              FROM currency_rates_daily WHERE to_currency = 'GBP') t
        -- > 5 so a long-weekend cluster (Good Friday + Easter Monday) stays quiet
        WHERE date - prev_date > 5
        """,
    ),
    (
        "txn_reconciliation",
        "current quantity not explained by transaction history (missed CSV import?)",
        "Export the missing period from T212 and re-run scripts/update_history_from_csv.py "
        "(idempotent). If a corporate action T212 recorded outside the CSV explains the "
        "drift, document it — MWRR uses this history as cash-flow input.",
        """
        SELECT i.t212_code, h.quantity AS held,
               round(sum(t.quantity)::numeric, 4) AS from_txns
        FROM instruments i
        JOIN holdings_daily h ON h.instrument_id = i.id AND h.date = (SELECT max(date) FROM holdings_daily)
        JOIN transaction_history t ON t.ticker = i.t212_code
        WHERE t.action IN ('MARKET_BUY', 'LIMIT_BUY', 'MARKET_SELL', 'LIMIT_SELL',
                           'STOCK_SPLIT_OPEN', 'STOCK_SPLIT_CLOSE')
        GROUP BY i.t212_code, h.quantity
        HAVING abs(h.quantity - sum(t.quantity)) > 0.01
        """,
    ),
    (
        "stale_pr_rows",
        f"PR placeholder rows older than {STALE_PR_DAYS} days (supersede or route failed)",
        "canonical_sibling=true = stale duplicate: the supersede window missed the PR row "
        "(pre-Jul-2026 rows predate the ±7-day match) — DELETE the PR row. false = no "
        "route delivered a canonical filing: EU/Canada name with no route at all "
        "(SAF.PA/RHM.DE/MDA.TO), an FPI whose results 6-K sits deeper than the "
        "FPI_SCAN_LIMIT newest filings (NVO/KYIV file weekly buyback 6-Ks), or a UK "
        "name whose Investegate announcements come back as stubs (AZN.L/BARC.L — no "
        "CIK, so the RNS route owns them). The Jul 2026 queue-starvation bug also "
        "showed up here first — if many US names flag at once, check the SEC task logs.",
        f"""
        SELECT i.yahoo_symbol, e.date AS pr_date, CURRENT_DATE - e.date AS age_days,
               EXISTS (SELECT 1 FROM earnings_reports c
                       WHERE c.instrument_id = e.instrument_id
                         AND coalesce(c.metrics->>'report_type', '') != 'PR'
                         AND c.date BETWEEN e.date - 7 AND e.date + 7) AS canonical_sibling
        FROM earnings_reports e
        JOIN instruments i ON i.id = e.instrument_id
        WHERE e.metrics->>'report_type' = 'PR'
          AND e.date < CURRENT_DATE - {STALE_PR_DAYS}
        ORDER BY canonical_sibling DESC, e.date
        """,
    ),
    (
        "pe_history_gaps",
        f"held equities with no PE history, or histories frozen > {STALE_PE_DAYS} days",
        "last_pe_key IS NULL = never seeded: the Wisesheets onboarding queue should pick "
        "it up within a night or two of the next scrape run — if it persists, the symbol "
        "may not be dot-free or its quoteType is wrong. A dated last_pe_key = the nightly "
        "scrape keeps failing for that symbol (it sorts first in the refresh queue, so "
        "it IS being attempted): check the update_pe_data task logs for the parse error. "
        "Commodity ETCs Yahoo mislabels EQUITY are excluded via the earnings requirement.",
        f"""
        SELECT i.yahoo_symbol,
               (SELECT max(k) FROM jsonb_object_keys(coalesce(y.pes, '{{}}'::jsonb)) k) AS last_pe_key,
               h.instrument_id IS NOT NULL AS held
        FROM instruments i
        JOIN instruments_yahoo y ON y.instrument_id = i.id
        LEFT JOIN (
            SELECT instrument_id FROM holdings_daily
            WHERE date = (SELECT max(date) FROM holdings_daily) AND quantity > 0
        ) h ON h.instrument_id = i.id
        WHERE coalesce(y.info->>'quoteType', 'EQUITY') = 'EQUITY'
          AND (
            (h.instrument_id IS NOT NULL
             AND (y.pes IS NULL OR y.pes = '{{}}'::jsonb)
             AND y.earnings IS NOT NULL AND y.earnings != '{{}}'::jsonb)
            OR (SELECT max(k) FROM jsonb_object_keys(coalesce(y.pes, '{{}}'::jsonb)) k)
               < (CURRENT_DATE - {STALE_PE_DAYS})::text
          )
        ORDER BY (h.instrument_id IS NOT NULL) DESC, last_pe_key NULLS FIRST
        """,
    ),
]


def run_check(session, name: str, description: str, fix: str, sql: str) -> int:
    """Run one audit query and log its findings. Returns the finding count."""
    params = {}
    if ":allowed" in sql:
        # SP500/QQQ constituents are tracked for breadth metrics without instruments.
        params["allowed"] = SP500 + QQQ + list(BENCHES) + [VIX]
    if ":delisted" in sql:
        params["delisted"] = list(STOCKS_DELISTED)
    if ":alias_old" in sql:
        # Known renames: a finding on an old symbol means rows/instruments still
        # need migrating; one on a new symbol means an instrument likely still
        # points at the pre-rename ticker.
        params["alias_old"] = list(STOCKS_ALIASES.keys())
        params["alias_new"] = list(STOCKS_ALIASES.values())
    rows = session.execute(text(sql), params).all()
    if not rows:
        logger.info("PASS  %s", name)
        return 0
    logger.warning("FAIL  %s — %d finding(s): %s", name, len(rows), description)
    for row in rows[:MAX_ROWS_SHOWN]:
        logger.warning("      %s", dict(row._mapping))
    if len(rows) > MAX_ROWS_SHOWN:
        logger.warning("      ... and %d more", len(rows) - MAX_ROWS_SHOWN)
    logger.warning("      fix: %s", fix)
    return len(rows)


def check_data_integrity(only: str | None = None) -> int:
    """Run all (or one) integrity checks. Returns the total finding count."""
    checks = [c for c in CHECKS if only is None or c[0] == only]
    if not checks:
        raise SystemExit(f"Unknown check '{only}'. Available: {', '.join(c[0] for c in CHECKS)}")

    total = 0
    failed = 0
    with get_session() as session:
        for name, description, fix, sql in checks:
            findings = run_check(session, name, description, fix, sql)
            total += findings
            failed += bool(findings)
    logger.info("Done: %d/%d checks failed, %d finding(s) total", failed, len(checks), total)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run read-only data-integrity checks")
    parser.add_argument("--check", help="Run a single check by name", default=None)
    args = parser.parse_args()
    check_data_integrity(only=args.check)
