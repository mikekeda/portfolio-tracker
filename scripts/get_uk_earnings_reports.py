"""
Fetches the latest earnings-related RNS announcement for each eligible UK
instrument (``yahoo_symbol`` ending in ``.L``) via Investegate and runs the
shared LLM analysis pipeline. Sibling to ``get_earnings_reports.py`` (US SEC).

Coverage model — "latest-real, forward-looking":
  * Walks an instrument's recent earnings-relevant RNS announcements
    newest-first and processes the first that passes every gate. Older
    candidates already in the DB short-circuit the loop.
  * Three filters protect the LLM from non-results disclosures:
      1. Headline classifier (matches Interim / Preliminary / Final /
         Full Year / Annual / Trading Update; rejects PDMR Shareholding,
         Holding(s) in Company, Form 8.3/8.5, Transaction in Own Shares).
      2. Body size gate (``MIN_BODY_CHARS``) rejects publication-notice
         stubs ("Annual Report has been published…") before the LLM call.
      3. LLM ``is_earnings_report`` flag — last-resort skip.

Selection: never-seen instruments first, then by most recent past Yahoo
earnings date desc — same priority as the SEC route.

Eligibility: ``yahoo_symbol LIKE '%.L' AND cik IS NULL``. Investegate keys
per-company URLs by TIDM (LSE-listings only), so a country filter would
lose non-UK names that report via RNS on LSE. The CIK-null clause avoids
double-processing FPIs that already route through SEC; the dual-listed UK
majors (SHEL, BP, AZN, GSK, HSBA, BARC, LLOY, NWG) lack a CIK locally
because populate_cik.py matches on US tickers only and so flow here.

Source: ``https://www.investegate.co.uk/company/{TIDM}`` — third-party
aggregator (LSE's own feed is paid; lse.com is bot-blocked). ~50 rows per
page; we paginate lazily via ``?page=N`` (cheap names cost one fetch)
because daily 8.3/8.5/buyback RNS for high-churn tickers (BARC, GSK, LLOY,
HSBA, STAN, SMT., JD.) push the latest earnings off page 1.
"""

import argparse
import re
from collections import Counter
from datetime import date, datetime
from time import perf_counter, sleep

import requests
from bs4 import BeautifulSoup
from sqlalchemy import Date, cast, func, or_, select
from sqlalchemy.sql import text as sql_text

from config import logger
from data import ETC_SYMBOLS, STOCKS_DELISTED
from models import EarningsReport, HoldingDaily, Instrument, InstrumentYahoo
from scripts._earnings_common import (
    DATA_DIR,
    PR_REPORT_TYPE,
    extract_text_from_html,
    summarize_with_llm,
)
from scripts.update_data import get_session

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15 PortfolioTracker/1.0"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

INVESTEGATE_BASE = "https://www.investegate.co.uk"
INVESTEGATE_COMPANY_URL = INVESTEGATE_BASE + "/company/{tidm}"

# Per-instrument listing-page cap. Walked lazily — page N+1 only fetched when
# pages 1..N yielded no save and no DB hit. ~50 rows/page; high-churn names
# (BARC, HSBA) need many pages because daily 8.3/8.5/buyback RNS dominate.
MAX_LISTING_PAGES = 10

# Throttle between paginated fetches inside one instrument; instrument-level
# politeness is handled by the 1s sleep in the driver.
INVESTEGATE_PAGE_DELAY = 0.3

# Plain-text body size below which we treat the announcement as a stub (e.g.
# "Annual Report has been published, available at https://…" publication
# notices) and skip without calling the LLM. Observed sizes:
#   - real Interim/Final/Prelim/Annual press releases: 50–170 KB
#   - publication-notice stubs: 4–7 KB
# 15 KB sits comfortably between them.
MIN_BODY_CHARS = 15000

# Outcome tags returned by get_uk_earnings_report and tallied for the run-end
# summary. Mutually exclusive — exactly one is returned per call.
OUTCOME_SAVED = "saved"
OUTCOME_UPDATE_NOOP = "update-noop"
OUTCOME_EXHAUSTED = "exhausted"
OUTCOME_NO_MATCH = "no-match"
OUTCOME_LISTING_FAILED = "listing-failed"
OUTCOME_NOT_LISTED = "not-listed"

# Non-earnings-issuing listings that pass Yahoo's quoteType=EQUITY filter.
# ETCs and GDRs are sometimes tagged EQUITY by Yahoo even though they don't
# publish UK RNS earnings. Verified empirically via shortName lookup —
# extend as new ones appear in the holdings.
# Union, not replacement: the ETCs come from the shared list, the other two are
# here for unrelated reasons.
NON_EARNINGS_LISTINGS = ETC_SYMBOLS | frozenset({
    "SMSN.L",  # Samsung Electronics GDR (Korean issuer, depository receipt)
    "PHNX.L",  # Phoenix Group Holdings — TODO: real UK earnings issuer but
               # Investegate has no /company/PHNX page (verified 2026-04-26).
               # Remove from this set once an alt source is wired up.
})

# Headline → type discriminator. Order matters — first match wins, so the more
# specific patterns are listed before broader ones. All matches are
# case-insensitive.
#
# We deliberately MATCH publication-notice headlines like "Annual Financial
# Report" (banks/oil/pharma) and "Annual Report and Accounts" (smaller caps)
# even though they are sometimes stubs — the size gate above weeds out the
# stubs while still letting through the genuine annual document where it is
# the actual filing.
HEADLINE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("INTERIM", re.compile(r"\b(half[- ]year(?:ly)?|interim\s+(?:results|statement|report))\b", re.IGNORECASE)),
    ("PRELIM", re.compile(r"\b(prelim(?:inary)?\s+results)\b", re.IGNORECASE)),
    ("FINAL", re.compile(r"\bfinal\s+results\b", re.IGNORECASE)),
    # Full-year press releases used by AZN, GSK, SHEL, BP, RIO, etc. that were
    # not caught by FINAL/PRELIM/ANNUAL — added 2026-04 after first dry-run.
    (
        "FULLYEAR",
        re.compile(
            r"\b(full[- ]?year\s+results|q[1-4]\s+and\s+full[- ]?year\s+results|fourth\s+quarter\s+(?:and\s+)?(?:full[- ]?year\s+)?results)\b",
            re.IGNORECASE,
        ),
    ),
    ("ANNUAL", re.compile(r"\bannual\s+(?:report|results|financial\s+report)\b", re.IGNORECASE)),
    ("TRADING", re.compile(r"\b(trading\s+(?:update|statement)|q[1-4]\s+trading|quarterly\s+update)\b", re.IGNORECASE)),
)


def classify_headline(headline: str) -> str | None:
    """Return the report TYPE for a headline, or None if not earnings-related."""
    for type_name, pattern in HEADLINE_PATTERNS:
        if pattern.search(headline):
            return type_name
    return None


def yahoo_symbol_to_tidm_candidates(yahoo_symbol: str) -> list[str]:
    """Return Investegate TIDM candidates to try, in priority order.

    Yahoo uses dashes for hyphenated TIDMs (e.g. ``BT-A.L``) where Investegate
    uses dots (``BT.A``), so we always convert dashes to dots first.

    Some London tickers natively END in a dot (``BP.``, ``JD.``) but Yahoo
    drops the trailing dot, so ``BP.L`` → ``BP`` and ``JD.L`` → ``JD``. To
    recover those we add a trailing-dot variant as a fallback whenever the
    base candidate has no internal dot.

    Returns an empty list if the symbol is not a ``.L`` listing.
    """
    if not yahoo_symbol.endswith(".L"):
        return []
    base = yahoo_symbol[:-2].replace("-", ".")
    candidates = [base]
    if "." not in base:
        candidates.append(base + ".")
    return candidates


# ─── Listing parser ─────────────────────────────────────────────────────────


def _parse_announcement_row(row) -> dict | None:
    """Parse a single announcement row from the Investegate company page table.

    Returns dict with keys ``date`` (datetime.date), ``headline`` (str),
    ``url`` (absolute URL), or None if the row doesn't look like an
    announcement.
    """
    cells = row.find_all("td")
    if len(cells) < 4:
        return None

    # The Investegate row layout is: Date | Time | Source-link | Announcement-link
    # The announcement link has class="announcement-link"; the source link
    # (RNS / PRN / etc) is on the third cell and must NOT be picked up here.
    link = row.find("a", class_="announcement-link", href=True)
    if not link:
        return None

    href = link["href"]
    if "/announcement/" not in href:
        return None

    headline = link.get_text(strip=True)
    if not headline:
        return None

    # Parse date from the first cell. Format observed: "24 Apr 2026"
    date_text = cells[0].get_text(strip=True)
    try:
        announcement_date = datetime.strptime(date_text, "%d %b %Y").date()
    except ValueError:
        return None

    url = href if href.startswith("http") else INVESTEGATE_BASE + href

    return {"date": announcement_date, "headline": headline, "url": url}


def _fetch_listing_page_for_tidm(tidm: str, page: int) -> list[dict]:
    """Fetch one Investegate company page and parse announcement rows."""
    url = INVESTEGATE_COMPANY_URL.format(tidm=tidm)
    if page > 1:
        url = f"{url}?page={page}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Investegate listing fetch failed for TIDM=%s page=%d: %s", tidm, page, e)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    announcements: list[dict] = []
    for row in soup.find_all("tr"):
        parsed = _parse_announcement_row(row)
        if parsed is not None:
            announcements.append(parsed)

    return announcements


def iter_listing_pages(yahoo_symbol: str, max_pages: int = MAX_LISTING_PAGES):
    """Yield ``(page_number, rows)`` for ``yahoo_symbol``'s Investegate listing.

    Resolves the TIDM on the first non-empty page (handles ``BP.L`` → ``BP.``
    fallback once) and reuses it. Stops on empty page or ``max_pages``.
    """
    working_tidm: str | None = None
    for page in range(1, max_pages + 1):
        if page > 1:
            sleep(INVESTEGATE_PAGE_DELAY)

        if working_tidm is None:
            for tidm in yahoo_symbol_to_tidm_candidates(yahoo_symbol):
                rows = _fetch_listing_page_for_tidm(tidm, page=page)
                if rows:
                    working_tidm = tidm
                    logger.debug(
                        "Investegate listing for %s served via TIDM=%s (page %d, %d rows)",
                        yahoo_symbol,
                        tidm,
                        page,
                        len(rows),
                    )
                    yield page, rows
                    break
            if working_tidm is None:
                return
        else:
            rows = _fetch_listing_page_for_tidm(working_tidm, page=page)
            if not rows:
                return
            logger.debug(
                "Investegate listing for %s page %d (%d rows)",
                yahoo_symbol,
                page,
                len(rows),
            )
            yield page, rows


def find_earnings_announcements(announcements: list[dict]) -> list[tuple[dict, str]]:
    """Return earnings-relevant announcements paired with their type, newest-first."""
    # Dedupe on (date, type) — Investegate sometimes rebroadcasts the same RNS
    # under "RNS Reach" with a different URL but identical body (seen on TSCO.L
    # 2026-04-16). No UK issuer publishes two distinct earnings of the same
    # type on the same day, so this is safe and avoids a redundant fetch.
    seen: set[tuple[date, str]] = set()
    out: list[tuple[dict, str]] = []
    for a in announcements:
        type_name = classify_headline(a["headline"])
        if type_name is None:
            continue
        key = (a["date"], type_name)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, type_name))
    return out


# ─── Announcement fetch + cache ─────────────────────────────────────────────


def get_announcement_html(ticker: str, announcement_date: date, report_type: str, url: str) -> str | None:
    """Return announcement HTML body, using local disk cache.

    ``report_type`` is the full ``RNS-<TYPE>`` string used both for the cache
    filename and stamped into ``EarningsReport.metrics``, keeping disk and DB
    naming aligned.

    Cache layout: data/filings/{ticker}/{YYYY-MM-DD}_{report_type}.html
    On HTTP error returns None and logs a warning.
    """
    ticker_dir = DATA_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{announcement_date.isoformat()}_{report_type}.html"
    file_path = ticker_dir / filename

    if file_path.exists():
        logger.debug("Loading cached announcement from %s", file_path)
        return file_path.read_text(encoding="utf-8")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Announcement download failed for %s %s: %s", ticker, url, e)
        return None

    html = response.text
    # Publication notices are stubs that Investegate may later back with the real
    # body; caching one would pin the stub verdict for good.
    if len(extract_text_from_html(html)) >= MIN_BODY_CHARS:
        file_path.write_text(html, encoding="utf-8")
    return html


# ─── Per-instrument flow ────────────────────────────────────────────────────


def get_uk_earnings_report(ticker: str, session, instrument_id: int) -> str:
    """Fetch + analyse the latest earnings-relevant RNS for one ``.L`` ticker.

    Walks Investegate pages lazily, newest-first. Short-circuits on first DB
    hit (no-op) or saves the first row passing the size + LLM gates. Returns
    one of the ``OUTCOME_*`` tags for batch tallying.
    """
    if not ticker.endswith(".L"):
        logger.debug("[skip] %s is not a .L listing", ticker)
        return OUTCOME_NOT_LISTED

    accumulated_rows: list[dict] = []
    seen_keys: set[tuple[date, str]] = set()
    skipped_stubs = 0
    candidates_walked = 0
    pages_used = 0

    for _page_num, page_rows in iter_listing_pages(ticker):
        pages_used += 1
        accumulated_rows.extend(page_rows)

        all_candidates = find_earnings_announcements(accumulated_rows)
        new_candidates = [(a, t) for a, t in all_candidates if (a["date"], t) not in seen_keys]
        if not new_candidates:
            continue

        for announcement, type_name in new_candidates:
            announcement_date = announcement["date"]
            seen_keys.add((announcement_date, type_name))
            candidates_walked += 1
            report_type = f"RNS-{type_name}"

            existing = session.execute(
                select(EarningsReport).filter(
                    EarningsReport.instrument_id == instrument_id,
                    EarningsReport.date == announcement_date,
                )
            ).scalar_one_or_none()
            if existing:
                # First-candidate page-1 hits stay at DEBUG to keep noop runs quiet;
                # post-stub or paginated hits log at INFO so the fallback is visible.
                if skipped_stubs > 0 or pages_used > 1:
                    logger.info(
                        "[update-noop] %s — already up-to-date at %s (walked %d stub(s), %d page(s))",
                        ticker,
                        announcement_date.isoformat(),
                        skipped_stubs,
                        pages_used,
                    )
                else:
                    logger.debug(
                        "[skip-existing] %s already in DB (%s period %s)",
                        ticker,
                        report_type,
                        announcement_date.isoformat(),
                    )
                return OUTCOME_UPDATE_NOOP

            # Sentinel persisting an LLM verdict that this announcement is not a
            # results document, so nightly re-runs don't re-analyse the same RNS.
            nonearnings_marker = DATA_DIR / ticker / f"{announcement_date.isoformat()}_{report_type}.nonearnings"
            if nonearnings_marker.exists():
                logger.debug(
                    "[skip] %s %s period %s — marked non-earnings on a previous run",
                    ticker,
                    report_type,
                    announcement_date.isoformat(),
                )
                continue

            html = get_announcement_html(ticker, announcement_date, report_type, announcement["url"])
            if html is None:
                continue

            text = extract_text_from_html(html)
            if len(text) < MIN_BODY_CHARS:
                # DEBUG: the per-instrument summary line carries the stub count.
                logger.debug(
                    "[skip-stub] %s %s period %s — body %d chars < %d (publication notice); trying earlier candidate",
                    ticker,
                    report_type,
                    announcement_date,
                    len(text),
                    MIN_BODY_CHARS,
                )
                skipped_stubs += 1
                continue

            result = summarize_with_llm(text, ticker, report_type=report_type, period=announcement_date.isoformat())
            if result is None:
                logger.warning("[error] LLM failed for %s %s — trying earlier candidate", ticker, announcement_date)
                continue

            if result.get("is_earnings_report") is False:
                nonearnings_marker.touch()
                logger.info(
                    "[skip] %s %s period %s — LLM flagged as non-earnings; trying earlier candidate",
                    ticker,
                    report_type,
                    announcement_date,
                )
                continue

            summary = result.get("summary", "")
            metrics = {k: v for k, v in result.items() if k != "summary"}
            metrics["report_type"] = report_type
            metrics["headline"] = announcement["headline"]
            metrics["source_url"] = announcement["url"]

            earnings_report = EarningsReport(
                instrument_id=instrument_id,
                date=announcement_date,
                summary=summary,
                metrics=metrics,
            )
            session.add(earnings_report)
            session.commit()

            assessment = metrics.get("investment_assessment") or {}
            logger.info(
                "[saved] %s %s period %s | rec=%s conv=%s | pages=%d | %s",
                ticker,
                report_type,
                announcement_date,
                assessment.get("recommendation", "—"),
                assessment.get("conviction", "—"),
                pages_used,
                announcement["headline"][:80],
            )
            return OUTCOME_SAVED

    if pages_used == 0:
        logger.warning(
            "No announcements parsed for %s (tried TIDMs %s)",
            ticker,
            yahoo_symbol_to_tidm_candidates(ticker),
        )
        return OUTCOME_LISTING_FAILED

    if candidates_walked == 0:
        logger.info(
            "[no-match] %s — none of %d announcements across %d page(s) look like earnings",
            ticker,
            len(accumulated_rows),
            pages_used,
        )
        return OUTCOME_NO_MATCH

    logger.info(
        "[exhausted] %s — %d earnings-relevant candidates rejected (stubs/non-earnings) across %d page(s)",
        ticker,
        candidates_walked,
        pages_used,
    )
    return OUTCOME_EXHAUSTED


# ─── Selection / batch driver ───────────────────────────────────────────────


def _latest_yahoo_earnings_date(earnings: dict | None) -> str | None:
    """Most recent key from Yahoo's earnings JSON that is strictly before today."""
    if not earnings:
        return None
    today_iso = date.today().isoformat()
    return next((d for d in sorted(earnings.keys(), reverse=True) if d < today_iso), None)


def get_uk_earnings_reports(limit: int = 10, only_holdings: bool = False) -> None:
    """Fetch and process UK RNS earnings reports for eligible ``.L`` instruments.

    Eligibility:
        - yahoo_symbol LIKE '%.L'
        - cik IS NULL (instruments with a CIK go through the SEC pipeline)
        - InstrumentYahoo row exists (so we can rank by Yahoo earnings dates)

    Selection (mirrors the US script):
        - Work-pending gate: an instrument is a candidate only while its newest
          canonical (non-PR) summary predates its latest Yahoo earnings date.
          Instruments with no Yahoo earnings data stay in the pool — we cannot
          tell when new results are due — and fall to the end of the ordering.
        1. Never-seen instruments (has_reports = 0) first.
        2. Within that, most recent past Yahoo earnings announcement date desc.
    """

    with get_session() as session:
        today_iso = date.today().isoformat()

        max_earnings_date_expr = (
            select(func.max(sql_text("date")))
            .select_from(func.jsonb_object_keys(InstrumentYahoo.earnings).alias("date"))
            .where(sql_text("date < :today_iso").bindparams(today_iso=today_iso))
            .correlate(InstrumentYahoo)
            .scalar_subquery()
        )

        has_reports = (
            select(func.count(EarningsReport.id))
            .where(EarningsReport.instrument_id == Instrument.id)
            .correlate(Instrument)
            .scalar_subquery()
        )

        # Newest canonical (non-PR) summary per instrument. PR placeholders must
        # not count here — they'd mask the still-missing RNS results they stand in for.
        last_canonical_created = (
            select(func.max(EarningsReport.created_at))
            .where(
                EarningsReport.instrument_id == Instrument.id,
                func.coalesce(EarningsReport.metrics["report_type"].astext, "") != PR_REPORT_TYPE,
            )
            .correlate(Instrument)
            .scalar_subquery()
        )

        # Skip non-equity vehicles by Yahoo quoteType — they never publish RNS
        # earnings. coalesce-to-EQUITY keeps rows with a missing quoteType in
        # the pool rather than silently dropping them.
        non_equity_quote_types = ("ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY")
        equity_only = func.coalesce(InstrumentYahoo.info["quoteType"].astext, "EQUITY").notin_(
            non_equity_quote_types
        )

        query = (
            select(
                Instrument.id,
                Instrument.yahoo_symbol,
                InstrumentYahoo.earnings,
                has_reports.label("has_reports"),
            )
            .join(InstrumentYahoo)
            .filter(
                Instrument.yahoo_symbol.like("%.L"),
                Instrument.cik.is_(None),
                equity_only,
                Instrument.yahoo_symbol.notin_(NON_EARNINGS_LISTINGS),
                # Delisted lines never publish new RNS (AGR.L sat permanently in
                # the never-seen queue burning Investegate fetches every night).
                Instrument.yahoo_symbol.notin_(STOCKS_DELISTED),
                # Work-pending gate: drop instruments whose newest canonical summary
                # already postdates their last Yahoo announcement — same starvation
                # fix as the SEC route (2026-07). Instruments without Yahoo earnings
                # data stay in: we cannot tell when their results are due.
                or_(
                    last_canonical_created.is_(None),
                    max_earnings_date_expr.is_(None),
                    cast(last_canonical_created, Date) < cast(max_earnings_date_expr, Date),
                ),
            )
            .order_by(
                # Boolean never-seen flag, NOT the raw count: ordering by count keeps
                # low-count instruments permanently ahead of multi-report ones.
                (has_reports == 0).desc(),
                max_earnings_date_expr.desc().nulls_last(),
            )
            .limit(limit)
        )

        if only_holdings:
            latest_date = session.execute(select(func.max(HoldingDaily.date))).scalar()
            if latest_date:
                held_ids = (
                    session.execute(
                        select(HoldingDaily.instrument_id)
                        .where(HoldingDaily.date == latest_date)
                        .where(HoldingDaily.quantity > 0)
                    )
                    .scalars()
                    .all()
                )
                query = query.filter(Instrument.id.in_(held_ids))
                logger.info("only_holdings=True: restricting to %d held instruments", len(held_ids))
            else:
                logger.warning("only_holdings=True but no holdings found — processing nothing")
                return

        result = session.execute(query).all()
        total = len(result)
        start_ts = perf_counter()

        eligible_total = (
            session.execute(
                select(func.count())
                .select_from(Instrument)
                .join(InstrumentYahoo)
                .where(
                    Instrument.yahoo_symbol.like("%.L"),
                    Instrument.cik.is_(None),
                    equity_only,
                    Instrument.yahoo_symbol.notin_(NON_EARNINGS_LISTINGS),
                    Instrument.yahoo_symbol.notin_(STOCKS_DELISTED),
                )
            ).scalar()
            or 0
        )
        backlog_total = (
            session.execute(
                select(func.count())
                .select_from(Instrument)
                .join(InstrumentYahoo)
                .where(
                    Instrument.yahoo_symbol.like("%.L"),
                    Instrument.cik.is_(None),
                    equity_only,
                    Instrument.yahoo_symbol.notin_(NON_EARNINGS_LISTINGS),
                    has_reports == 0,
                )
            ).scalar()
            or 0
        )

        never_seen = sum(1 for r in result if r.has_reports == 0)
        updates = total - never_seen
        yahoo_dates = [d for d in (_latest_yahoo_earnings_date(r.earnings) for r in result) if d]
        date_span = f"{min(yahoo_dates)} → {max(yahoo_dates)}" if yahoo_dates else "n/a"
        logger.info(
            "UK RNS: processing %d instruments (%d never-seen, %d updates) | "
            "Backlog: %d never-seen of %d eligible | Yahoo earnings span: %s | first 10: %s",
            total,
            never_seen,
            updates,
            backlog_total,
            eligible_total,
            date_span,
            [r.yahoo_symbol for r in result[:10]],
        )

        outcomes: Counter[str] = Counter()
        processed = 0
        for row in result:
            logger.debug(
                "── %s  (Yahoo last earnings: %s)",
                row.yahoo_symbol,
                _latest_yahoo_earnings_date(row.earnings) or "n/a",
            )

            try:
                outcome = get_uk_earnings_report(
                    ticker=row.yahoo_symbol,
                    session=session,
                    instrument_id=row.id,
                )
                outcomes[outcome] += 1
            except Exception:
                logger.exception("UK RNS processing failed for %s", row.yahoo_symbol)
                session.rollback()
                outcomes["error"] += 1

            processed += 1
            if processed % 10 == 0 or processed == total:
                elapsed_min = (perf_counter() - start_ts) / 60
                logger.info("UK RNS progress: %d/%d processed (elapsed %.1fm)", processed, total, elapsed_min)

            sleep(1)

        logger.info(
            "UK RNS task complete: %d instruments processed in %.1fm | "
            "saved=%d update-noop=%d exhausted=%d no-match=%d listing-failed=%d errors=%d",
            processed,
            (perf_counter() - start_ts) / 60,
            outcomes[OUTCOME_SAVED],
            outcomes[OUTCOME_UPDATE_NOOP],
            outcomes[OUTCOME_EXHAUSTED],
            outcomes[OUTCOME_NO_MATCH],
            outcomes[OUTCOME_LISTING_FAILED],
            outcomes["error"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and summarise UK RNS earnings via Investegate")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of instruments to process (default: 10)")
    parser.add_argument(
        "--only-holdings",
        action="store_true",
        default=False,
        help="Restrict to instruments currently held in the portfolio",
    )
    args = parser.parse_args()
    get_uk_earnings_reports(limit=args.limit, only_holdings=args.only_holdings)
