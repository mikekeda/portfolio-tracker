"""Ingest a single earnings-release (PDF or HTML page) and persist as ``EarningsReport`` (``report_type='PR'``).

Earnings press releases hit IR sites on the day of the earnings call, weeks before
the corresponding SEC 10-Q / 10-K / 20-F / 40-F / 6-K filing lands in EDGAR. This
script lets the dashboard pick up forward-looking signal same-day.

Usage::

    # Direct PDF (Alphabet, ASML, ...)
    python scripts/ingest_earnings_pr.py \\
        --url https://s206.q4cdn.com/.../2026q1-alphabet-earnings-release.pdf \\
        --ticker GOOGL --period-end 2026-03-31 --release-date 2026-04-29

    # Server-rendered HTML page (Meta, ...)
    python scripts/ingest_earnings_pr.py \\
        --url https://investor.atmeta.com/investor-news/.../Meta-Reports-First-Quarter-2026-Results/default.aspx \\
        --ticker META --period-end 2026-03-31 --release-date 2026-04-29

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
from urllib.parse import urlparse

from curl_cffi import requests as creq
from curl_cffi.requests.exceptions import RequestException
from sqlalchemy import select

from config import logger
from models import EarningsReport, Instrument
from scripts._earnings_common import (
    DATA_DIR,
    PR_REPORT_TYPE,
    extract_text_from_html,
    summarize_pdf_with_llm,
    summarize_with_llm,
)
from scripts.update_data import get_session

# Real press-release HTML runs 5-20k chars; below this is almost certainly
# a JS-rendered shell with only nav/footer — bail before the Gemini call.
MIN_HTML_BODY_CHARS = 2000


def _fetch(url: str) -> tuple[bytes, str]:
    """GET ``url`` via curl_cffi impersonating Chrome.

    IR sites commonly sit behind Cloudflare/Akamai TLS-fingerprint sniffing
    (e.g. ADMA), where plain ``requests`` hangs. curl_cffi works for those and
    for unprotected sites alike, so we use it uniformly.
    """
    parsed = urlparse(url)
    resp = creq.get(
        url,
        impersonate="chrome131",
        timeout=60,
        headers={"Referer": f"{parsed.scheme}://{parsed.netloc}/"},
    )
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "")


def _download(url: str, cache_dir: Path, stem: str) -> tuple[bytes, str]:
    """Return ``(content_bytes, kind)`` where ``kind`` is ``"pdf"`` or ``"html"``.

    Cache file is ``cache_dir/{stem}.{pdf|html}``; subsequent calls reuse it.
    Kind is sniffed from the response Content-Type header (with PDF magic-byte
    fallback). Raises ValueError on anything that isn't PDF or HTML.
    """
    pdf_path = cache_dir / f"{stem}.pdf"
    html_path = cache_dir / f"{stem}.html"
    if pdf_path.exists():
        logger.debug("Loading cached PDF from %s", pdf_path)
        return pdf_path.read_bytes(), "pdf"
    if html_path.exists():
        logger.debug("Loading cached HTML from %s", html_path)
        return html_path.read_bytes(), "html"

    logger.info("Downloading %s", url)
    content, content_type_raw = _fetch(url)

    content_type = content_type_raw.lower()
    is_pdf = "pdf" in content_type or content.startswith(b"%PDF")

    cache_dir.mkdir(parents=True, exist_ok=True)
    if is_pdf:
        pdf_path.write_bytes(content)
        return content, "pdf"
    if "html" in content_type:
        html_path.write_bytes(content)
        return content, "html"
    raise ValueError(
        f"URL did not return PDF or HTML (Content-Type={content_type!r}, "
        f"first bytes={content[:8]!r})"
    )


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

        # Both PR and canonical rows key on period_end, so one row max per fiscal period.
        existing = session.execute(
            select(EarningsReport)
            .where(EarningsReport.instrument_id == instrument.id)
            .where(EarningsReport.date == period_end)
        ).scalar_one_or_none()

        existing_pr: EarningsReport | None = None
        if existing:
            # pre-cce2a1f3 rows lack report_type; treat any non-"PR" as canonical.
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

        cache_dir = DATA_DIR / ticker
        stem = f"{release_date.isoformat()}_PR"
        try:
            content, kind = _download(url, cache_dir, stem)
        except (RequestException, ValueError) as e:
            logger.error("Download failed for %s (%s): %s", ticker, url, e)
            return

        if kind == "pdf":
            result = summarize_pdf_with_llm(
                pdf_bytes=content,
                ticker=ticker,
                report_type=PR_REPORT_TYPE,
                period=period_end.isoformat(),
            )
        else:
            text = extract_text_from_html(content.decode("utf-8", errors="replace"))
            if len(text) < MIN_HTML_BODY_CHARS:
                logger.warning(
                    "%s: extracted only %d chars from %s — likely a JS-rendered shell. "
                    "Find the underlying press-release URL (often /press/... or PR Newswire).",
                    ticker, len(text), url,
                )
                return
            result = summarize_with_llm(
                text=text,
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
                url,
            )
            return

        summary = result["summary"]
        metrics = {k: v for k, v in result.items() if k != "summary"}
        metrics["report_type"] = PR_REPORT_TYPE
        # Stored explicitly so _build_earnings_reports doesn't need a Yahoo-match.
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
        description="Ingest a single earnings-release (PDF or HTML) as EarningsReport(report_type='PR').",
    )
    parser.add_argument("--url", required=True, help="HTTPS URL to the earnings-release PDF or HTML page")
    parser.add_argument("--ticker", required=True, help="Yahoo symbol (must match Instrument.yahoo_symbol)")
    parser.add_argument(
        "--period-end",
        required=True,
        help="Fiscal period end date stated on the press release (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--release-date",
        required=True,
        help="Press-release date stated on the press release (YYYY-MM-DD)",
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
