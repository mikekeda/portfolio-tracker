"""
Fetches the latest 10-Q / 10-K / 20-F / 40-F filing for each eligible instrument
from the SEC EDGAR API and extracts structured metrics via an LLM.

Coverage model — "latest-only, forward-looking":
  * Each run processes the *single most recent* SEC filing per instrument
    (``get_latest_filing_metadata_any``). Historical filings are never
    backfilled, even if Yahoo lists many prior earnings announcements that
    have no matching ``EarningsReport`` row.
  * Rationale: the dashboard ranks current buy/sell signal and forward
    guidance — a 2024 10-Q adds little for a 10-year-horizon investor and
    would multiply LLM cost by ~4×/year of history.
  * Consequence: an instrument that becomes eligible today (e.g. CIK
    populated, Yahoo earnings data arrived) jumps straight to its current
    latest filing; prior filings it missed while ineligible stay
    permanently unprocessed. If you need historical coverage you'd add an
    opt-in ``--backfill-history N`` mode; this script does not provide one.

Selection:
  * Work-pending gate: an instrument is a candidate only while its newest
    canonical (non-PR) summary predates its latest Yahoo earnings date.
    Once the post-announcement filing is summarised it leaves the queue,
    so a fixed ORDER BY + LIMIT cannot starve the remaining instruments.
  1. Instruments with zero ``EarningsReport`` rows ("never-seen") first,
     so brand-new instruments in the portfolio get their current filing
     summarised before we refresh existing ones.
  2. Within that, order by the most recent past Yahoo earnings date
     descending — recent reporters before stale ones.

Per-instrument decision flow lives in ``get_earnings_report``; the summary
logged at the start of ``get_earnings_reports`` shows the never-seen /
updates split and the global never-seen backlog.
"""

import argparse
from datetime import date
from time import perf_counter, sleep

import requests
from sqlalchemy import Date, cast, func, or_, select
from sqlalchemy.sql import text as sql_text

from config import SEC_USER_AGENT, logger
from models import EarningsReport, HoldingDaily, Instrument, InstrumentYahoo
from scripts._earnings_common import (
    DATA_DIR,
    PR_REPORT_TYPE,
    _check_file_exists_for_date,
    extract_text_from_html,
    summarize_with_llm,
)
from scripts.update_data import get_session

HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"


def get_latest_filing_metadata(cik: str, form_types: tuple[str, ...] = ("10-Q", "10-K")):
    """
    Fetches the most recent filing metadata for the given CIK and form types.

    For US domestic companies the default ("10-Q", "10-K") works.
    Foreign private issuers (e.g. ASML, NVO, NU) file 20-F (annual) instead of 10-K,
    and Canadian companies file 40-F.  Pass form_types=("20-F",) or ("40-F",) explicitly,
    or use the helper get_latest_filing_metadata_any() which tries both automatically.
    """

    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    response.raise_for_status()
    data = response.json()

    filings = data["filings"]["recent"]

    for i, form in enumerate(filings["form"]):
        if form in form_types:
            return {
                "accessionNumber": filings["accessionNumber"][i],
                "primaryDocument": filings["primaryDocument"][i],
                "form": form,
                "reportDate": filings["reportDate"][i],
            }

    return None


def get_latest_filing_metadata_any(cik: str):
    """
    Tries domestic (10-Q/10-K) first, then foreign (20-F/40-F) as fallback.
    Covers US companies, foreign private issuers, and Canadian filers.
    """
    result = get_latest_filing_metadata(cik, form_types=("10-Q", "10-K"))
    if result is None:
        result = get_latest_filing_metadata(cik, form_types=("20-F", "40-F"))
    return result


def get_filing_html(cik: str, ticker: str, metadata: dict) -> str:
    """
    Retrieves the filing HTML content.
    Checks local disk first; if missing, downloads from SEC and saves it.
    """
    accession = metadata["accessionNumber"].replace("-", "")
    primary_doc = metadata["primaryDocument"]
    report_date = metadata["reportDate"]
    form = metadata["form"]

    # 1. Construct Local Path: data/filings/{TICKER}/{YYYY-MM-DD}_{FORM}.html
    ticker_dir = DATA_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize form name just in case (e.g. "10-Q/A" -> "10-Q_A")
    safe_form = form.replace("/", "_")
    filename = f"{report_date}_{safe_form}.html"
    file_path = ticker_dir / filename

    # 2. Check Cache
    if file_path.exists():
        logger.debug("Loading cached filing from %s", file_path)
        return file_path.read_text(encoding="utf-8")

    # 3. Download if missing
    url = SEC_ARCHIVES_URL.format(cik=cik, accession=accession, primary_document=primary_doc)

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    html_content = response.content.decode("utf-8", errors="replace")

    # 4. Save to Disk
    file_path.write_text(html_content, encoding="utf-8")

    return html_content


def _latest_yahoo_earnings_date(earnings: dict | None) -> str | None:
    """Most recent key from Yahoo's earnings JSON that is strictly before today.
    Returns an ISO date string, or None if the dict is empty / has only future dates."""
    if not earnings:
        return None
    today_iso = date.today().isoformat()
    return next((d for d in sorted(earnings.keys(), reverse=True) if d < today_iso), None)


def get_earnings_report(ticker: str, cik: str, session, instrument_id: int):
    """
    Fetches, processes, and saves the *latest* SEC earnings filing for a given ticker.

    Decision flow:
    1. Fetch filing metadata from SEC (lightweight JSON, tries 10-Q/10-K then 20-F/40-F).
    2. DB record exists for that report_date → skip (already analysed).
    3. HTML cached on disk for that report_date → regenerate summary from cache
       (post-table-truncation scenario: re-run LLM without re-downloading).
    4. Neither DB nor cache → download from SEC and analyse.

    Scope note: "latest" means the single most recent 10-Q/10-K (or 20-F/40-F).
    If older filings were never processed (common after an instrument just became
    eligible), they are NOT backfilled here — see the module docstring.

    Note: we intentionally do NOT compare the SEC report_date to the Yahoo announcement
    date. Those are different things (period end vs. announcement day) and comparing them
    caused legitimate new filings to be skipped.
    """

    # 1. Filing metadata
    metadata = get_latest_filing_metadata_any(cik)

    if not metadata:
        logger.warning("No supported filings (10-Q/10-K/20-F/40-F) found for %s CIK %s", ticker, cik)
        return None

    report_date = metadata["reportDate"]
    form = metadata["form"]

    # 2. DB check — skip if a canonical row already exists. A PR placeholder is
    # superseded at save time (step 7), not here: deleting it now would leak an
    # uncommitted delete into the shared session when the download/LLM step bails.
    existing_report = session.execute(
        select(EarningsReport).filter(
            EarningsReport.instrument_id == instrument_id, EarningsReport.date == date.fromisoformat(report_date)
        )
    ).scalar_one_or_none()

    pr_placeholder = None
    if existing_report:
        # PR rows are placeholders ingested via ingest_earnings_pr.py; replace
        # them with the canonical SEC filing. Legacy rows (pre-cce2a1f3) lack
        # report_type → fall into the else branch and are kept as-is.
        if existing_report.metrics.get("report_type") == PR_REPORT_TYPE:
            pr_placeholder = existing_report
        else:
            logger.debug("[skip] %s already in DB (form %s, period %s)", ticker, form, report_date)
            return existing_report

    # 3. Check HTML cache
    safe_form = form.replace("/", "_")
    html_cached = _check_file_exists_for_date(ticker, report_date, safe_form)

    if html_cached:
        logger.debug("[cache] %s %s period %s — regenerating summary (no download)", ticker, form, report_date)
    else:
        logger.debug("[download] %s %s period %s — fetching from SEC", ticker, form, report_date)

    # 4. Get HTML (returns cached file if present, downloads otherwise)
    try:
        html_content = get_filing_html(cik, ticker, metadata)
    except requests.RequestException as e:
        logger.warning("%s download failed: %s", ticker, e)
        return None

    # 5. Extract text
    text = extract_text_from_html(html_content)

    # 6. LLM analysis
    result = summarize_with_llm(text, ticker, report_type=form, period=report_date)

    if result is None:
        logger.warning("[error] LLM failed for %s %s — skipping DB save", ticker, report_date)
        return None

    if result.get("is_earnings_report") is False:
        logger.info("[skip] %s %s period %s — LLM flagged as non-earnings disclosure", ticker, form, report_date)
        return None

    summary = result.get("summary", "")
    metrics = {k: v for k, v in result.items() if k != "summary"}
    metrics["report_type"] = form
    metrics["source_url"] = SEC_ARCHIVES_URL.format(
        cik=cik,
        accession=metadata["accessionNumber"].replace("-", ""),
        primary_document=metadata["primaryDocument"],
    )

    # 7. Save — replace the PR placeholder first to free the (instrument_id, date) slot
    if pr_placeholder is not None:
        logger.info(
            "[supersede] %s: replacing PR row %d with canonical %s (period %s)",
            ticker,
            pr_placeholder.id,
            form,
            report_date,
        )
        session.delete(pr_placeholder)
        session.flush()

    earnings_report = EarningsReport(
        instrument_id=instrument_id,
        date=date.fromisoformat(report_date),
        summary=summary,
        metrics=metrics,
    )
    session.add(earnings_report)
    session.commit()

    eps_guidance = (metrics.get("guidance") or {}).get("eps_guidance") or {}
    assessment = metrics.get("investment_assessment") or {}
    logger.info(
        "[saved] %s %s period %s | rec=%s conv=%s | EPS next_q=%s next_y=%s growth=%s%%",
        ticker,
        form,
        report_date,
        assessment.get("recommendation", "—"),
        assessment.get("conviction", "—"),
        eps_guidance.get("next_quarter"),
        eps_guidance.get("next_year"),
        eps_guidance.get("growth_pct"),
    )
    return earnings_report


def get_earnings_reports(limit: int = 100, only_holdings: bool = False):
    """
    Fetch and process earnings reports for instruments with earnings data.

    Args:
        limit: Maximum number of instruments to process per invocation.
        only_holdings: When True, restrict to instruments currently held in the portfolio
                       (latest HoldingDaily snapshot with quantity > 0).

    Selection priority (see module docstring for the full model):
        1. Never-seen instruments (``has_reports = 0``) first.
        2. Within that, most recent past Yahoo earnings announcement date first.

    Per-instrument behaviour: only the *latest* SEC filing is considered — older
    unprocessed filings are never backfilled by this script.

    Truncation note:
        If the earnings_reports table is truncated (e.g. to force re-summarisation with a
        new prompt), this function will detect that the HTML filing is already cached on disk
        and skip the SEC download — it will only re-run the LLM summarisation step and
        re-insert the record into the database.
    """

    with get_session() as session:
        # Get max earnings date from JSONB keys (most recent earnings date that's before today)
        today_iso = date.today().isoformat()

        max_earnings_date_expr = (
            select(func.max(sql_text("date")))
            .select_from(func.jsonb_object_keys(InstrumentYahoo.earnings).alias("date"))
            .where(sql_text("date < :today_iso").bindparams(today_iso=today_iso))
            .correlate(InstrumentYahoo)
            .scalar_subquery()
        )

        # Check if instrument has any earnings reports in database
        has_reports = (
            select(func.count(EarningsReport.id))
            .where(EarningsReport.instrument_id == Instrument.id)
            .correlate(Instrument)
            .scalar_subquery()
        )

        # Newest canonical (non-PR) summary per instrument. PR placeholders must
        # not count here — they'd mask the still-missing SEC filing they stand in for.
        last_canonical_created = (
            select(func.max(EarningsReport.created_at))
            .where(
                EarningsReport.instrument_id == Instrument.id,
                func.coalesce(EarningsReport.metrics["report_type"].astext, "") != PR_REPORT_TYPE,
            )
            .correlate(Instrument)
            .scalar_subquery()
        )

        query = (
            select(
                Instrument.id,
                Instrument.yahoo_symbol,
                Instrument.cik,
                InstrumentYahoo.earnings,
                has_reports.label("has_reports"),
            )
            .join(InstrumentYahoo)
            .filter(
                Instrument.cik.is_not(None),
                InstrumentYahoo.earnings != "{}",
                max_earnings_date_expr.is_not(None),
                # Work-pending gate: drop instruments whose newest canonical summary
                # already postdates their last Yahoo announcement. Without it a fixed
                # ORDER BY + LIMIT re-selects the same instruments every night and the
                # rest starve (holdings sat on PR placeholders for months, 2026-07).
                or_(
                    last_canonical_created.is_(None),
                    cast(last_canonical_created, Date) < cast(max_earnings_date_expr, Date),
                ),
            )
            .order_by(
                # Boolean never-seen flag, NOT the raw count: ordering by count keeps
                # low-count instruments permanently ahead of multi-report holdings.
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

        # Global backlog — eligible instruments (CIK + Yahoo earnings data) that have
        # no EarningsReport row yet. Compared against the current batch's never-seen
        # count so the log tells you whether today's run will close the gap.
        eligible_total = (
            session.execute(
                select(func.count())
                .select_from(Instrument)
                .join(InstrumentYahoo)
                .where(
                    Instrument.cik.is_not(None),
                    InstrumentYahoo.earnings != "{}",
                    max_earnings_date_expr.is_not(None),
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
                    Instrument.cik.is_not(None),
                    InstrumentYahoo.earnings != "{}",
                    max_earnings_date_expr.is_not(None),
                    has_reports == 0,
                )
            ).scalar()
            or 0
        )

        # Summary of what got selected: never-seen (top-priority bucket) vs updates
        # (already have at least one report on file), plus the span of Yahoo-indicated
        # earnings announcement dates in this batch.
        never_seen = sum(1 for r in result if r.has_reports == 0)
        updates = total - never_seen
        yahoo_dates = [d for d in (_latest_yahoo_earnings_date(r.earnings) for r in result) if d]
        date_span = f"{min(yahoo_dates)} → {max(yahoo_dates)}" if yahoo_dates else "n/a"
        logger.info(
            "SEC: processing %d instruments (%d never-seen, %d updates) | Backlog: %d never-seen of %d eligible | "
            "Yahoo earnings span: %s | first 10: %s",
            total,
            never_seen,
            updates,
            backlog_total,
            eligible_total,
            date_span,
            [r.yahoo_symbol for r in result[:10]],
        )

        processed = 0
        for row in result:
            last_earnings_date = _latest_yahoo_earnings_date(row.earnings)
            if not last_earnings_date:
                logger.warning("No valid earnings date found for %s despite query filter, skipping", row.yahoo_symbol)
                continue

            logger.debug("── %s  (Yahoo last earnings: %s)", row.yahoo_symbol, last_earnings_date)

            try:
                get_earnings_report(
                    ticker=row.yahoo_symbol,
                    cik=row.cik,
                    session=session,
                    instrument_id=row.id,
                )
            except Exception:
                logger.exception("SEC processing failed for %s", row.yahoo_symbol)
                session.rollback()
            processed += 1
            if processed % 10 == 0 or processed == total:
                elapsed_min = (perf_counter() - start_ts) / 60
                logger.info(
                    "SEC progress: %d/%d processed (elapsed %.1fm)",
                    processed,
                    total,
                    elapsed_min,
                )

            sleep(1)

        logger.info(
            "SEC task complete: %d instruments processed in %.1fm",
            processed,
            (perf_counter() - start_ts) / 60,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and summarise SEC earnings reports")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of instruments to process (default: 10)")
    parser.add_argument(
        "--only-holdings",
        action="store_true",
        default=False,
        help="Restrict to instruments currently held in the portfolio",
    )
    args = parser.parse_args()
    get_earnings_reports(limit=args.limit, only_holdings=args.only_holdings)
