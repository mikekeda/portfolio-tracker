"""Ingest a single earnings-release PDF and persist as ``EarningsReport`` (``report_type='PR'``).

Earnings press releases hit IR sites on the day of the earnings call, weeks before
the corresponding SEC 10-Q / 10-K / 20-F / 40-F / 6-K filing lands in EDGAR. This
script lets the dashboard pick up forward-looking signal same-day.

Usage::

    python scripts/ingest_earnings_pr.py \\
        --url https://s206.q4cdn.com/.../2026q1-alphabet-earnings-release.pdf \\
        --ticker GOOGL \\
        --period-end 2026-03-31 \\
        --release-date 2026-04-24

The persisted row uses ``period-end`` as ``EarningsReport.date`` — matching the
SEC ingestion convention (``EarningsReport.date == EDGAR reportDate == fiscal
period-end``). The ``release-date`` is stored in ``metrics["release_date"]`` so
the API can surface the announcement separately.

When the canonical SEC filing arrives, ``scripts/get_earnings_reports.py``
deletes the PR row at the same ``(instrument_id, period_end)`` and inserts the
10-Q/10-K/20-F/40-F/6-K in its place.
"""

import argparse
from datetime import date
from pathlib import Path

import requests
from sqlalchemy import select

from config import SEC_USER_AGENT, logger
from models import EarningsReport, Instrument
from scripts._earnings_common import DATA_DIR, PR_REPORT_TYPE, summarize_pdf_with_llm
from scripts.update_data import get_session


def _download_pdf(url: str, cache_path: Path) -> bytes:
    """Return the PDF bytes, fetching from ``url`` if not already cached on disk."""
    if cache_path.exists():
        logger.debug("Loading cached PDF from %s", cache_path)
        return cache_path.read_bytes()

    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=60, headers={"User-Agent": SEC_USER_AGENT})
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not resp.content.startswith(b"%PDF"):
        raise ValueError(
            f"URL did not return a PDF (Content-Type={content_type!r}, "
            f"first bytes={resp.content[:8]!r})"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    return resp.content


def ingest(
    url: str,
    ticker: str,
    period_end: date,
    release_date: date,
    force: bool = False,
) -> None:
    with get_session() as session:
        instrument = session.execute(
            select(Instrument).where(Instrument.yahoo_symbol == ticker)
        ).scalar_one_or_none()
        if not instrument:
            logger.error("No instrument found for ticker %s — aborting", ticker)
            return

        # PR and canonical SEC rows both key on period_end, so the
        # UniqueConstraint(instrument_id, date) guarantees at most one row.
        existing = session.execute(
            select(EarningsReport)
            .where(EarningsReport.instrument_id == instrument.id)
            .where(EarningsReport.date == period_end)
        ).scalar_one_or_none()

        existing_pr: EarningsReport | None = None
        if existing:
            # Legacy SEC rows (pre-cce2a1f3) have metrics without "report_type".
            # Treat any non-"PR" value (incl. legacy) as canonical and skip.
            existing_type = existing.metrics.get("report_type")
            if existing_type != PR_REPORT_TYPE:
                logger.info(
                    "%s: canonical earnings report for fiscal period %s already exists "
                    "(id=%d, type=%s) — skipping PR ingest",
                    ticker,
                    period_end,
                    existing.id,
                    existing_type or "legacy",
                )
                return
            if not force:
                logger.info(
                    "%s: PR for fiscal period %s already exists (id=%d) — pass --force to overwrite",
                    ticker,
                    period_end,
                    existing.id,
                )
                return
            existing_pr = existing

        cache_path = DATA_DIR / ticker / f"{release_date.isoformat()}_PR.pdf"
        try:
            pdf_bytes = _download_pdf(url, cache_path)
        except (requests.RequestException, ValueError) as e:
            logger.error("PDF download failed for %s (%s): %s", ticker, url, e)
            return

        result = summarize_pdf_with_llm(
            pdf_bytes=pdf_bytes,
            ticker=ticker,
            report_type=PR_REPORT_TYPE,
            period=period_end.isoformat(),
        )
        if result is None:
            logger.error("LLM analysis failed for %s — not persisting", ticker)
            return

        if result["is_earnings_report"] is False:
            logger.warning(
                "%s: LLM flagged %s as a non-earnings disclosure — not persisting",
                ticker,
                cache_path.name,
            )
            return

        summary = result["summary"]
        metrics = {k: v for k, v in result.items() if k != "summary"}
        metrics["report_type"] = PR_REPORT_TYPE
        # Stored explicitly so _build_earnings_reports can surface the announcement
        # without a Yahoo-match. EarningsReport.date already carries the period-end.
        metrics["release_date"] = release_date.isoformat()
        metrics["source_url"] = url

        if existing_pr:
            existing_pr.summary = summary
            existing_pr.metrics = metrics
            session.commit()
            row_id = existing_pr.id
            action = "updated"
        else:
            row = EarningsReport(
                instrument_id=instrument.id,
                date=period_end,
                summary=summary,
                metrics=metrics,
            )
            session.add(row)
            session.commit()
            row_id = row.id
            action = "saved"

        eps = metrics["guidance"]["eps_guidance"] or {}
        assessment = metrics["investment_assessment"] or {}
        logger.info(
            "[%s] %s PR fiscal=%s release=%s id=%d | rec=%s conv=%s | EPS next_q=%s next_y=%s growth=%s%%",
            action,
            ticker,
            period_end,
            release_date,
            row_id,
            assessment.get("recommendation", "—"),
            assessment.get("conviction", "—"),
            eps.get("next_quarter"),
            eps.get("next_year"),
            eps.get("growth_pct"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a single earnings-release PDF as EarningsReport(report_type='PR').",
    )
    parser.add_argument("--url", required=True, help="HTTPS URL to the earnings-release PDF")
    parser.add_argument("--ticker", required=True, help="Yahoo symbol (must match Instrument.yahoo_symbol)")
    parser.add_argument(
        "--period-end",
        required=True,
        help="Fiscal period end date stated on the PDF cover (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--release-date",
        required=True,
        help="Press-release date stated on the PDF cover (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing PR row for the same fiscal period",
    )
    args = parser.parse_args()

    ingest(
        url=args.url,
        ticker=args.ticker,
        period_end=date.fromisoformat(args.period_end),
        release_date=date.fromisoformat(args.release_date),
        force=args.force,
    )


if __name__ == "__main__":
    main()
