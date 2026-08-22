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

from config import BENCHES, LOOKTHROUGH_MIN_EXPOSURE_GBP, VIX, logger
from data import ETF_HOLDING_SOURCES, QQQ, SP500, STOCKS_ALIASES, STOCKS_DELISTED, WISESHEETS_NO_PE
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
# 13F is due 45 days after quarter end; past that plus a few days' grace, a manager still
# behind the newest filed quarter has stopped filing rather than being late.
STALE_13F_DAYS = 50
# The refresh runs weekly; a gap past this means it stopped, whether or not
# the index itself moved (an unchanged fetch still stamps updated_at).
STALE_ETF_HOLDING_DAYS = 30
# Derived metrics are rewritten by every update_features run, so any gap at all
# means the nightly job has been failing past its last few attempts.
STALE_DERIVED_METRICS_DAYS = 3
# Mirrors MIN_COVERAGE in backend/utils/etf_aggregates.py, as a percentage.
MIN_LOOKTHROUGH_COVERAGE_PCT = 80
PORTFOLIO_DIFF_PCT = 1.0  # T212 total vs holdings sum; small FX drift is normal
# Same bounds as the split detector: an unadjusted 2:1 split shows as ~0.5x.
MOVE_UPPER, MOVE_LOWER = 1.8, 0.55
# Wrong-currency rows on LSE lines: Yahoo intermittently serves the USD line's
# quote on a GBP/GBX series, so the day-over-day ratio lands on the USD->GBP
# rate (or its inverse / x100 GBX variants) instead of a market move.
FX_MIN_MOVE = 0.15       # smaller implied moves are indistinguishable from market noise
FX_RATE_TOL = 0.02       # ratio must match the day's rate within 2%
FX_REVERT_TOL = 0.05     # next close back within 5% of prev = transient junk row
FX_INCEPTION_ROWS = 20   # series-start window where USD junk rows cluster

# (name, description, fix hint, SQL) — every check is a plain read-only query
# so it can be pasted into psql/pgAdmin unchanged when digging into a finding.
CHECKS: list[tuple[str, str, str, str]] = [
    (
        "etf_holding_prices",
        "material ETF look-through constituents with missing or stale prices",
        "The look-through drops a constituent it cannot price, shifting that weight into "
        "the fund's residual. Only slices above LOOKTHROUGH_MIN_EXPOSURE_GBP are reported: "
        "most constituents are not in SP500/QQQ so nothing fetches them, which is expected "
        "and immaterial. A finding is usually a stale yahoo_symbol on the resolved "
        "Instrument (ECM.L for RS Group, renamed RS1.L): fix instruments.yahoo_symbol and "
        "add the old ticker to STOCKS_ALIASES.",
        f"""
        WITH fx AS (
            SELECT from_currency, rate FROM currency_rates_daily
            WHERE date = (SELECT max(date) FROM currency_rates_daily)
        ),
        fund_value AS (
            SELECT i.id AS instrument_id,
                   h.quantity * h.current_price * (CASE
                       WHEN i.currency = 'GBX' THEN 0.01
                       WHEN i.currency = 'GBP' THEN 1.0
                       ELSE (SELECT rate FROM fx WHERE from_currency = i.currency)
                   END) AS gbp
            FROM holdings_daily h
            JOIN instruments i ON i.id = h.instrument_id
            WHERE h.date = (SELECT max(date) FROM holdings_daily) AND h.quantity > 0
        ),
        latest AS (
            SELECT etf_instrument_id, max(date) AS date
            FROM etf_holdings GROUP BY etf_instrument_id
        )
        SELECT ci.yahoo_symbol AS sym, f.yahoo_symbol AS fund,
               round((fv.gbp * ec.weight_pct / 100.0)::numeric, 2) AS exposure_gbp,
               count(p.date) AS obs, max(p.date) AS last_price
        FROM etf_holdings ec
        JOIN latest l ON l.etf_instrument_id = ec.etf_instrument_id AND l.date = ec.date
        JOIN fund_value fv ON fv.instrument_id = ec.etf_instrument_id
        JOIN instruments f ON f.id = ec.etf_instrument_id
        JOIN instruments ci ON ci.id = ec.instrument_id
        LEFT JOIN prices_daily p ON p.symbol = ci.yahoo_symbol
        WHERE fv.gbp * ec.weight_pct / 100.0 >= :lt_floor
        GROUP BY ci.yahoo_symbol, f.yahoo_symbol, fv.gbp, ec.weight_pct
        HAVING count(p.date) = 0 OR max(p.date) < CURRENT_DATE - {STALE_PRICE_DAYS}
        ORDER BY fv.gbp * ec.weight_pct / 100.0 DESC
        """,
    ),
    (
        "etf_holding_snapshots",
        "held ETFs whose stored holdings are missing, un-refreshed, or do not sum to ~100%",
        "Each held ETF in ETF_HOLDING_SOURCES should have been refreshed recently and sum "
        "near 100%. missing = update_etf_holdings.py has never succeeded for it; "
        "not_refreshed = the weekly job stopped (checked is max(updated_at), which every "
        "run stamps even when the index has not moved); low_sum = the "
        "issuer file was truncated, so the fund's look-through silently under-attributes.",
        f"""
        WITH latest AS (
            SELECT etf_instrument_id, max(date) AS date
            FROM etf_holdings GROUP BY etf_instrument_id
        ),
        snap AS (
            SELECT i.yahoo_symbol, l.date, max(ec.updated_at)::date AS checked,
                   sum(ec.weight_pct) AS total_pct, count(ec.id) AS rows,
                   count(ec.instrument_id) AS resolved_rows,
                   sum(ec.weight_pct) FILTER (WHERE ec.instrument_id IS NOT NULL) AS resolved_pct
            FROM instruments i
            LEFT JOIN latest l ON l.etf_instrument_id = i.id
            LEFT JOIN etf_holdings ec
                   ON ec.etf_instrument_id = i.id AND ec.date = l.date
            WHERE i.yahoo_symbol = ANY(:sourced_etfs)
            GROUP BY i.yahoo_symbol, l.date
        )
        SELECT yahoo_symbol, date, checked, round(total_pct::numeric, 2) AS total_pct, rows,
               resolved_rows, round(resolved_pct::numeric, 2) AS resolved_pct,
               CASE WHEN date IS NULL THEN 'missing'
                    WHEN checked < CURRENT_DATE - {STALE_ETF_HOLDING_DAYS} THEN 'not_refreshed'
                    ELSE 'low_sum' END AS problem
        FROM snap
        WHERE date IS NULL
           OR checked < CURRENT_DATE - {STALE_ETF_HOLDING_DAYS}
           OR total_pct < 95.0
        ORDER BY yahoo_symbol
        """,
    ),
    (
        "etf_derived_metrics",
        "sourced ETFs whose look-through fundamentals are missing or stale",
        "update_features.py aggregates each fund's constituents into "
        "instruments_yahoo.derived_metrics, after the FeaturesDaily upsert. missing means a "
        "fund with enough resolved weight to qualify has no payload, so that job did not "
        "reach the end; stale means it stopped running. A fund below the coverage gate is "
        "NOT reported: R2SC.L resolves at ~7% of fund weight and is withheld on purpose, so "
        "flagging it would fail this check every night for working as designed.",
        f"""
        WITH latest AS (
            SELECT etf_instrument_id, max(date) AS date
            FROM etf_holdings GROUP BY etf_instrument_id
        ),
        resolved AS (
            SELECT e.etf_instrument_id,
                   sum(e.weight_pct) FILTER (WHERE e.instrument_id IS NOT NULL)
                       / NULLIF(sum(e.weight_pct), 0) * 100 AS resolved_pct
            FROM etf_holdings e
            JOIN latest l ON l.etf_instrument_id = e.etf_instrument_id AND l.date = e.date
            GROUP BY e.etf_instrument_id
        )
        SELECT i.yahoo_symbol,
               (y.derived_metrics->>'computed_at')::date AS computed_at,
               (y.derived_metrics->>'n_resolved')::int AS n_resolved,
               round(r.resolved_pct::numeric, 1) AS resolved_pct,
               round((y.derived_metrics->'coverage'->>'pe_ratio')::numeric, 1) AS pe_coverage_pct,
               CASE WHEN y.derived_metrics IS NOT NULL THEN 'stale' ELSE 'missing' END AS problem
        FROM instruments i
        JOIN instruments_yahoo y ON y.instrument_id = i.id
        LEFT JOIN resolved r ON r.etf_instrument_id = i.id
        WHERE i.yahoo_symbol = ANY(:sourced_etfs)
          AND (
                (y.derived_metrics IS NOT NULL
                 AND (y.derived_metrics->>'computed_at')::date < CURRENT_DATE - {STALE_DERIVED_METRICS_DAYS})
             OR (y.derived_metrics IS NULL
                 AND coalesce(r.resolved_pct, 0) >= {MIN_LOOKTHROUGH_COVERAGE_PCT})
              )
        ORDER BY i.yahoo_symbol
        """,
    ),
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
        "fx_mixed_rows",
        "LSE rows whose 1-day move matches the USD/GBP rate (wrong-currency data)",
        "A hit usually marks a wrong-currency row or the edge of a longer wrong-currency "
        "stretch, not a market move (real whipsaws land here occasionally — check the news "
        "first, e.g. bid rumours). Probe Yahoo's current data for the window: if it now "
        "disagrees with the stored rows (SGLN.L case) re-download in place with "
        "_update_prices(session, [sym], stretch_start). If Yahoo still serves the junk "
        "(VUAG.L/XNAS.L case) repair locally — delete junk inception rows or FX-convert "
        "the stretch, see scripts/fix_fx_mixed_prices.sql. Scan level continuity around a "
        "flagged edge before fixing: only stretch edges match the rate, not their interior.",
        f"""
        WITH moves AS (
            SELECT symbol, date, adj_close_price AS close,
                   lag(adj_close_price) OVER (PARTITION BY symbol ORDER BY date) AS prev,
                   lead(adj_close_price) OVER (PARTITION BY symbol ORDER BY date) AS next,
                   row_number() OVER (PARTITION BY symbol ORDER BY date) AS rn
            FROM prices_daily
            WHERE symbol LIKE '%.L'
              AND adj_close_price != 'NaN'::float8 AND adj_close_price > 0
        )
        SELECT m.symbol, m.date, round(m.prev::numeric, 2) AS prev,
               round(m.close::numeric, 2) AS close,
               round((100 * (m.close / m.prev - 1))::numeric) AS pct_move,
               m.rn <= {FX_INCEPTION_ROWS} AS near_inception,
               abs(m.next / m.prev - 1) < {FX_REVERT_TOL} AS reverts_next_day,
               m.date > CURRENT_DATE - 30 AS recent
        FROM moves m
        JOIN LATERAL (
            SELECT rate FROM currency_rates_daily r
            WHERE r.from_currency = 'USD' AND r.to_currency = 'GBP' AND r.date <= m.date
            ORDER BY r.date DESC LIMIT 1
        ) r ON true
        WHERE m.prev > 0
          AND abs(m.close / m.prev - 1) >= {FX_MIN_MOVE}
          AND (m.close / m.prev BETWEEN r.rate * {1 - FX_RATE_TOL} AND r.rate * {1 + FX_RATE_TOL}
               OR m.close / m.prev BETWEEN 1 / r.rate * {1 - FX_RATE_TOL} AND 1 / r.rate * {1 + FX_RATE_TOL}
               OR m.close / m.prev BETWEEN 100 * r.rate * {1 - FX_RATE_TOL} AND 100 * r.rate * {1 + FX_RATE_TOL}
               OR m.close / m.prev BETWEEN 1 / (100 * r.rate) * {1 - FX_RATE_TOL} AND 1 / (100 * r.rate) * {1 + FX_RATE_TOL})
          AND (m.rn <= {FX_INCEPTION_ROWS}
               OR abs(m.next / m.prev - 1) < {FX_REVERT_TOL}
               OR m.date > CURRENT_DATE - 30)
        ORDER BY m.symbol, m.date
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
        "quote_age_days in the hundreds = Yahoo's quote endpoint is frozen for that line "
        "(SMSN.L/KAP.L GDR case): the blob refetches daily but serves years-old data — "
        "only prices_daily is trustworthy for it, nothing to fix locally. Small "
        "quote_age_days = real drift (big move since last fetch) or the yahoo_symbol "
        "points at a different listing/currency than its prices_daily rows.",
        """
        SELECT i.yahoo_symbol,
               (y.info->>'regularMarketPrice')::float AS blob_price,
               round(p.close_price::numeric, 2) AS close_price,
               round(abs((y.info->>'regularMarketPrice')::float / p.close_price - 1)::numeric * 100) AS drift_pct,
               CURRENT_DATE - to_timestamp((y.info->>'regularMarketTime')::bigint)::date AS quote_age_days
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
        f"snapshot value (ex cash) diverging > {PORTFOLIO_DIFF_PCT}% from GBP sum of holdings",
        "Single old day with a big diff = one corrupted holdings row: compare per-holding "
        "quantity * current_price against adjacent days to find it, then fix that row "
        "(often a GBX/GBP or pre-split scale error). Big diff on a crash/rally day whose "
        "holdings rows were last written before the US close (check max(updated_at)) = "
        "intraday timing skew vs the backfilled close-based value — real and expected. "
        "Persistent small diffs = rate or timing skew, usually fine.",
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
        SELECT p.date, round((p.value - p.cash)::numeric, 2) AS invested_value,
               round(hs.holdings_sum::numeric, 2) AS holdings_sum,
               round((100 * (hs.holdings_sum - (p.value - p.cash)) / (p.value - p.cash))::numeric, 2) AS diff_pct
        FROM portfolio_daily p
        JOIN hs USING (date)
        WHERE p.value - p.cash > 0
          AND abs(p.value - p.cash - hs.holdings_sum) / (p.value - p.cash) > {PORTFOLIO_DIFF_PCT / 100}
        ORDER BY p.date DESC
        """,
    ),
    (
        "holdings_gaps",
        "holes inside a position's daily snapshots (5-90 days) with no sell in the gap",
        "The position was open through the gap yet has no snapshots — update_data didn't "
        "run those days. backfill_portfolio_daily covers portfolio totals but per-holding "
        "rows stay lost; note it and move on. (Closed-and-reopened positions are excluded "
        "automatically via a sell transaction inside the gap window.)",
        """
        SELECT i.yahoo_symbol, t.prev_date, t.date, t.date - t.prev_date AS gap_days
        FROM (SELECT instrument_id, date,
                     lag(date) OVER (PARTITION BY instrument_id ORDER BY date) AS prev_date
              FROM holdings_daily) t
        JOIN instruments i ON i.id = t.instrument_id
        WHERE t.date - t.prev_date BETWEEN 5 AND 90
          AND NOT EXISTS (
            SELECT 1 FROM transaction_history th
            WHERE th.isin = i.isin
              AND th.action::text LIKE '%SELL%'
              AND th.timestamp >= t.prev_date AND th.timestamp < t.date
          )
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
        "escaped_summaries",
        "earnings summaries with literal \\n instead of real newlines (LLM double-escape)",
        "Gemini structured output double-escaped the JSON string values; the summary "
        "renders as one blob with visible \\n on the Stock page. Run "
        "scripts/fix_earnings_summary_escapes.sql to repair. New rows should not appear "
        "— _undo_double_escapes in scripts/_earnings_common.py de-escapes at generation; "
        "if they do, the model found a new escape variant: inspect the raw summary.",
        """
        SELECT i.yahoo_symbol, e.date, e.metrics->>'report_type' AS report_type,
               (length(e.summary) - length(replace(e.summary, chr(92) || 'n', ''))) / 2
                   AS literal_newlines
        FROM earnings_reports e
        JOIN instruments i ON i.id = e.instrument_id
        WHERE strpos(e.summary, chr(92) || 'n') > 0
          AND strpos(e.summary, chr(10)) = 0
        ORDER BY e.date DESC
        """,
    ),
    (
        "pe_history_gaps",
        f"held equities with no PE history, or histories frozen > {STALE_PE_DAYS} days",
        "last_pe_key IS NULL = never seeded: the Wisesheets onboarding queue should pick "
        "it up within a night or two of the next scrape run — if it persists, the symbol "
        "may not be dot-free or its quoteType is wrong. A dated last_pe_key with scrape "
        "errors in the update_pe_data logs = the nightly scrape is failing. A dated key "
        "with clean logs (EU semi-annual reporters frozen at 12-31, cluster of .PA/.L "
        "names) = Wisesheets has no newer period yet — clears when H1 results publish. "
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
          AND i.yahoo_symbol != ALL(:wisesheets_no_pe)
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
    (
        "form13f_empty_filings",
        "13F filings stored with no holdings or no total value",
        "A parse failure or a NEW HOLDINGS amendment overwrote a good quarter. Re-fetch with "
        "scripts/fix_13f_backfill.py, which merges the original and its amendment.",
        """
        SELECT m.name, f.report_date, f.form, f.accession_number, f.total_value,
               count(h.id) AS holdings
        FROM form13f_filings f
        JOIN form13f_managers m ON m.id = f.manager_id
        LEFT JOIN form13f_holdings h ON h.filing_id = f.id
        GROUP BY m.name, f.report_date, f.form, f.accession_number, f.total_value
        HAVING count(h.id) = 0 OR f.total_value = 0
        ORDER BY f.report_date DESC
        """,
    ),
    (
        "form13f_stale_managers",
        "managers still behind the newest filed quarter after the 13F deadline",
        "Usually the manager re-filed under a new CIK: check EDGAR for a 13F-NT naming a "
        "successor and add it to that investor's 'ciks' in scripts/scrape_13f.py, then run "
        "scripts/fix_13f_backfill.py. Stale managers are excluded from consensus and scoring.",
        """
        WITH consensus AS (SELECT max(report_date) AS quarter FROM form13f_filings),
        latest AS (
            SELECT m.name, max(f.report_date) AS latest_filed
            FROM form13f_managers m
            JOIN form13f_filings f ON f.manager_id = m.id
            GROUP BY m.name
        )
        SELECT l.name, l.latest_filed, c.quarter AS consensus_quarter,
               CURRENT_DATE - c.quarter AS days_since_quarter_end
        FROM latest l CROSS JOIN consensus c
        WHERE l.latest_filed < c.quarter
          AND CURRENT_DATE > c.quarter + :stale_13f_days
        ORDER BY l.latest_filed, l.name
        """,
    ),
    (
        "form13f_cusip_casing",
        "13F holdings stored with a lower-case CUSIP",
        "The consensus dicts key on the raw CUSIP, so one filer's casing splits a name into two "
        "entries. Run scripts/fix_13f_cusip_case.sql; the writer normalises via normalize_cusip().",
        """
        SELECT m.name, h.cusip, max(h.issuer) AS issuer, count(*) AS rows
        FROM form13f_holdings h
        JOIN form13f_filings f ON f.id = h.filing_id
        JOIN form13f_managers m ON m.id = f.manager_id
        WHERE h.cusip <> upper(h.cusip)
        GROUP BY m.name, h.cusip
        ORDER BY count(*) DESC
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
    if ":wisesheets_no_pe" in sql:
        params["wisesheets_no_pe"] = list(WISESHEETS_NO_PE)
    if ":alias_old" in sql:
        # Known renames: a finding on an old symbol means rows/instruments still
        # need migrating; one on a new symbol means an instrument likely still
        # points at the pre-rename ticker.
        params["alias_old"] = list(STOCKS_ALIASES.keys())
        params["alias_new"] = list(STOCKS_ALIASES.values())
    if ":lt_floor" in sql:
        params["lt_floor"] = LOOKTHROUGH_MIN_EXPOSURE_GBP
    if ":sourced_etfs" in sql:
        params["sourced_etfs"] = [s for s, spec in ETF_HOLDING_SOURCES.items() if spec]
    if ":stale_13f_days" in sql:
        params["stale_13f_days"] = STALE_13F_DAYS
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
