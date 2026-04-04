"""
Fetches the latest 10-Q or 10-K filing for a given ticker from the SEC EDGAR API
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup
from google import genai
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.sql import text as sql_text, text

from config import GEMINI_API_KEY, logger
from models import EarningsReport, HoldingDaily, Instrument, InstrumentYahoo
from scripts.update_data import get_session

# Silence verbose third-party SDK loggers (AFC status, HTTP request lines).
# Note: google-genai uses "google_genai" (underscore) as its logger namespace, not "google.genai".
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# SEC requires a User-Agent in the format: "Company Name email@example.com"
# TODO: Replace with your actual contact info
USER_AGENT = "PortfolioTracker/1.0 (admin@example.com)"

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}

# SEC Endpoints
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"

# Local Storage Configuration
DATA_DIR = Path("data/filings")
MODEL = "gemini-3-flash-preview"

# Gemini client — created once at import time
_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


# Pydantic models for structured output - focused on most important metrics for retail investors
class EPSGuidance(BaseModel):
    """EPS guidance is most important for stock price (Price = EPS × PE)."""

    next_quarter: float | None = Field(None, description="EPS guidance for next quarter")
    next_year: float | None = Field(None, description="EPS guidance for full year")
    growth_pct: float | None = Field(None, description="Implied YoY growth percentage from guidance")


class RevenueGuidance(BaseModel):
    """Revenue guidance for forward periods."""

    next_quarter: float | None = Field(None, description="Revenue guidance for next quarter in millions")
    next_year: float | None = Field(None, description="Revenue guidance for full year in millions")
    growth_pct: float | None = Field(None, description="Implied growth percentage")


class Guidance(BaseModel):
    """Forward-looking guidance from management - CRITICAL for valuation."""

    eps_guidance: EPSGuidance | None = Field(None, description="EPS guidance is most important for stock price")
    revenue_guidance: RevenueGuidance | None = None
    operating_margin_guidance: float | None = Field(
        None, description="Guided operating margin % for next period, if explicitly stated (e.g. 'we expect operating margin of ~20%')"
    )
    outlook_commentary: str | None = Field(
        None,
        max_length=500,
        description="1-2 sentence summary of management's qualitative forward outlook (only if no specific numbers are provided above)",
    )


class ConsensusComparison(BaseModel):
    """How actual results and guidance compare to analyst consensus estimates."""

    revenue_beat_pct: float | None = Field(
        None, description="How much revenue beat/missed consensus (positive = beat, negative = miss)"
    )
    eps_beat_pct: float | None = Field(None, description="How much EPS beat/missed consensus")
    guidance_vs_consensus: Literal["above", "below", "in-line"] | None = Field(
        None, description="How guidance compares to consensus"
    )


class InvestmentAssessment(BaseModel):
    """LLM assessment of whether the stock is a good buy based on the report."""

    recommendation: Literal["buy", "hold", "avoid", "consider"] = Field(
        ..., description="Overall recommendation: buy (attractive), hold (neutral), avoid (unattractive), consider (worth researching)"
    )
    conviction: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in the recommendation based on clarity of data"
    )
    rationale: str = Field(
        "",
        description="2-3 sentence rationale for the recommendation",
        max_length=500,
    )
    key_catalysts: list[str] = Field(
        default_factory=list,
        description="Top 2-4 positive catalysts or growth drivers from the report",
    )
    key_concerns: list[str] = Field(
        default_factory=list,
        description="Top 2-4 risks or headwinds from the report",
    )
    quality_signals: Literal["strong", "adequate", "weak"] | None = Field(
        None,
        description="Earnings quality: FCF vs net income, margin trends, capital allocation",
    )


class EarningsReportMetrics(BaseModel):
    """Structured metrics extracted from earnings report - focused on forward-looking data."""

    guidance: Guidance = Field(..., description="Forward-looking guidance from management - CRITICAL for valuation")
    consensus_comparison: ConsensusComparison | None = None
    investment_assessment: InvestmentAssessment | None = Field(
        None,
        description="Assessment of whether the stock is a good buy based on the report",
    )
    summary: str = Field(
        ...,
        description="Markdown-formatted summary (300-500 words) with sections: Key Financial Results, Strategic Highlights, Risks & Headwinds, Future Outlook, Investment Thesis, Bottom Line. Use bold for key numbers, include specific percentages and comparisons.",
    )


def get_latest_filing_metadata(cik: str, form_types: tuple[str, ...] = ("10-Q", "10-K")):
    """
    Fetches the most recent filing metadata for the given CIK and form types.

    For US domestic companies the default ("10-Q", "10-K") works.
    Foreign private issuers (e.g. ASML, NVO, NU) file 20-F (annual) instead of 10-K,
    and Canadian companies file 40-F.  Pass form_types=("20-F",) or ("40-F",) explicitly,
    or use the helper get_latest_filing_metadata_any() which tries both automatically.
    """

    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
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
        logger.debug(f"Loading cached filing from {file_path}")
        return file_path.read_text(encoding="utf-8")

    # 3. Download if missing
    url = SEC_ARCHIVES_URL.format(cik=cik, accession=accession, primary_document=primary_doc)

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    html_content = response.content.decode("utf-8", errors="replace")

    # 4. Save to Disk
    file_path.write_text(html_content, encoding="utf-8")

    return html_content


def extract_text_from_html(html_content: str) -> str:
    """Parses HTML content to extract clean text for LLM."""

    # Parse HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript", "ix:header", "xbrl"]):
        tag.decompose()

    # Get text, use a separator to keep paragraphs distinct
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Filter out common XBRL noise lines
        if (
            line.startswith("http://")
            or line.startswith("https://")
            or line.startswith("iso4217:")
            or line.startswith("xbrli:")
        ):
            continue
        lines.append(line)

    text = "\n".join(lines)

    return text


def summarize_with_llm(text: str, ticker: str, form: str) -> dict[str, Any] | None:
    """
    Extracts structured metrics and generates summary from earnings report.
    Returns dict with structured metrics and summary text, or None on error.
    """
    logger.info(f"Analyzing {form} for {ticker} ({len(text)} characters)")

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set in config.py or environment variables")
        return None

    try:
        client = _get_genai_client()

        # Truncate text if it's too long
        max_chars = 500000
        if len(text) > max_chars:
            logger.info(f"Truncating text from {len(text)} to {max_chars} characters")
            text = text[:max_chars]

        prompt = f"""
        You are a value-oriented financial analyst helping a retail investor decide whether {ticker} is a good buy for their portfolio. Analyze this {form} earnings report and extract structured metrics while generating a comprehensive summary.

        **CRITICAL: EPS Guidance is the highest priority** - Stock price = EPS × PE ratio. If EPS guidance grows 10%, stock should theoretically grow 10% with same PE.

        **Extract the following structured data:**

        1. **Guidance (MOST IMPORTANT - REQUIRED):**
           - Scan ALL of: "Outlook", "Guidance", "Forecast", "Expectations", "Financial Outlook", "Business Outlook", "Looking Ahead", MD&A forward-looking sections.
           - **EPS guidance**: "expects EPS of $X.XX", "forecasts EPS between $X and $Y", "diluted EPS guidance of $X.XX–$Y.YY".
             Calculate growth_pct vs prior year actual EPS if available.
           - **Revenue guidance**: extract in millions for next quarter and full year; calculate growth_pct if prior period available.
           - **Operating margin guidance**: capture if stated explicitly (e.g. "operating margin of approximately 20%", "adjusted EBIT margin ~15%").
           - **Outlook commentary**: if no specific numbers exist, write 1–2 sentences summarising management's qualitative forward outlook (e.g. "expects double-digit revenue growth driven by cloud segment").
           - **IMPORTANT**: For numeric fields, only populate from explicitly stated numbers. Do NOT infer or estimate numbers not in the text.

        2. **Consensus Comparison (if available):**
           - Look for "consensus", "analyst estimates", "Street estimates", "beats", "misses"
           - Extract revenue_beat_pct and eps_beat_pct; note guidance_vs_consensus (above/below/in-line)
           - Only extract if explicitly mentioned. Do NOT calculate if consensus is not stated.

        3. **Investment Assessment (REQUIRED - answer "is this a good buy?"):**
           - **recommendation**: "buy" (attractive), "hold" (neutral), "avoid" (unattractive), or "consider" (worth researching)
           - **conviction**: "high", "medium", or "low" based on clarity of data and management credibility
           - **rationale**: 2-3 sentences explaining why (e.g., "Strong guidance raise and margin expansion support buy. Debt paydown reduces risk.")
           - **key_catalysts**: 2-4 positive drivers (e.g., "AI revenue accelerating", "Market share gains", "Buyback program")
           - **key_concerns**: 2-4 risks (e.g., "Macro headwinds", "Customer concentration", "Margin pressure")
           - **quality_signals**: "strong" (FCF > net income, margins expanding), "adequate", or "weak" (FCF concerns, margin compression)
           - Base assessment on: growth vs valuation, earnings quality, risk/reward, management credibility

        **Generate Summary (Markdown, 350-550 words):**

        ### Key Financial Results
        - Lead with revenue, EPS, net income with YoY/QoQ comparisons
        - Include gross margin and operating margin with basis point changes
        - Segment performance with specific percentages; beat/miss vs. consensus if mentioned

        ### Strategic Highlights
        - Major strategic initiatives; operational improvements; capital allocation (buybacks, dividends, debt)

        ### Risks & Headwinds
        - Top 3-5 risks with specific details; regulatory, competitive, macro concerns

        ### Future Outlook
        - **EPS guidance prominently featured** (next quarter/year with growth %)
        - Revenue guidance; CapEx plans; management commentary; capital allocation plans

        ### Investment Thesis
        - 2-3 bullet points: bull case, bear case, key metric to watch

        ### Bottom Line
        - 1-2 sentences: Direct answer "Good buy because..." or "Avoid because..." or "Hold and watch..."
        - Include the single most important factor from the report

        **Formatting:**
        - Use **bold** for key numbers and percentages
        - Use millions as unit (e.g., $13,640 million not $13.64 billion)
        - Be specific with numbers; if a metric is not available, use null

        --- REPORT TEXT ---
        {text}
        """

        # Use structured output with JSON schema
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": EarningsReportMetrics.model_json_schema(),
            },
        )

        # Parse and validate JSON response with Pydantic
        result_dict = json.loads(response.text)
        validated_result = EarningsReportMetrics.model_validate(result_dict)

        # Convert back to dict for storage (Pydantic model to dict)
        result = validated_result.model_dump()

        logger.info("LLM extraction complete for %s (%d chars processed)", ticker, len(text))
        return result

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return None


def _check_file_exists_for_date(ticker: str, report_date: str) -> bool:
    """Check if any filing file exists for the given date (handles form variations like 10-Q, 10-Q/A, etc.)."""
    ticker_dir = DATA_DIR / ticker
    if not ticker_dir.exists():
        return False

    for file_path in ticker_dir.iterdir():
        if file_path.is_file() and file_path.name.startswith(f"{report_date}_") and file_path.suffix == ".html":
            return True
    return False


def get_earnings_report(ticker: str, cik: str, session, instrument_id: int):
    """
    Fetches, processes, and saves the latest SEC earnings filing for a given ticker.

    Decision flow:
    1. Fetch filing metadata from SEC (lightweight JSON, tries 10-Q/10-K then 20-F/40-F).
    2. DB record exists for that report_date → skip (already analysed).
    3. HTML cached on disk for that report_date → regenerate summary from cache
       (post-table-truncation scenario: re-run LLM without re-downloading).
    4. Neither DB nor cache → download from SEC and analyse.

    Note: we intentionally do NOT compare the SEC report_date to the Yahoo announcement
    date.  Those are different things (period end vs. announcement day) and comparing them
    caused legitimate new filings to be skipped.
    """

    # 1. Filing metadata
    metadata = get_latest_filing_metadata_any(cik)

    if not metadata:
        logger.warning("No supported filings (10-Q/10-K/20-F/40-F) found for %s CIK %s", ticker, cik)
        return None

    report_date = metadata["reportDate"]
    form = metadata["form"]

    # 2. DB check — primary skip condition
    existing_report = session.execute(
        select(EarningsReport).filter(
            EarningsReport.instrument_id == instrument_id, EarningsReport.date == date.fromisoformat(report_date)
        )
    ).scalar_one_or_none()

    if existing_report:
        logger.debug("  [skip] %s already in DB (form %s, period %s)", ticker, form, report_date)
        return existing_report

    # 3. Check HTML cache
    html_cached = _check_file_exists_for_date(ticker, report_date)

    if html_cached:
        logger.debug("  [cache] %s %s period %s — regenerating summary (no download)", ticker, form, report_date)
    else:
        logger.debug("  [download] %s %s period %s — fetching from SEC", ticker, form, report_date)

    # 4. Get HTML (returns cached file if present, downloads otherwise)
    try:
        html_content = get_filing_html(cik, ticker, metadata)
    except requests.RequestException as e:
        logger.warning("%s download failed: %s", ticker, e)
        return None

    # 5. Extract text
    text = extract_text_from_html(html_content)

    # 6. LLM analysis
    result = summarize_with_llm(text, ticker, form)

    if result is None:
        logger.warning("  [error] LLM failed for %s %s — skipping DB save", ticker, report_date)
        return None

    summary = result.get("summary", "")
    metrics = {k: v for k, v in result.items() if k != "summary"}

    # 7. Save
    earnings_report = EarningsReport(
        instrument_id=instrument_id,
        date=date.fromisoformat(report_date),
        summary=summary,
        metrics=metrics,
    )
    session.add(earnings_report)
    session.commit()

    eps_guidance = (metrics.get("guidance") or {}).get("eps_guidance") or {}
    assessment = (metrics.get("investment_assessment") or {})
    logger.info(
        "  [saved] %s %s period %s | rec=%s conv=%s | EPS next_q=%s next_y=%s growth=%s%%",
        ticker, form, report_date,
        assessment.get("recommendation", "—"), assessment.get("conviction", "—"),
        eps_guidance.get("next_quarter"), eps_guidance.get("next_year"), eps_guidance.get("growth_pct"),
    )
    return earnings_report


def get_earnings_reports(limit: int = 100, only_holdings: bool = False):
    """
    Fetch and process earnings reports for instruments with earnings data.

    Args:
        limit: Maximum number of instruments to process.
        only_holdings: When True, restrict to instruments currently held in the portfolio
                       (latest HoldingDaily snapshot with quantity > 0).

    Prioritizes tickers that:
    1. Don't have existing reports in database yet (has_reports = 0 first)
    2. Have more recent earnings dates (more likely to have new filings)

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
            .where(text("date < :today_iso").bindparams(today_iso=today_iso))
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

        query = (
            select(
                Instrument.id,
                Instrument.yahoo_symbol,
                Instrument.cik,
                InstrumentYahoo.earnings,
            )
            .join(InstrumentYahoo)
            .filter(
                Instrument.cik.is_not(None),
                InstrumentYahoo.earnings != "{}",
                max_earnings_date_expr.is_not(None),
            )
            .order_by(
                has_reports.asc(),
                max_earnings_date_expr.desc().nulls_last(),
            )
            .limit(limit)
        )

        if only_holdings:
            latest_date = session.execute(select(func.max(HoldingDaily.date))).scalar()
            if latest_date:
                held_ids = session.execute(
                    select(HoldingDaily.instrument_id)
                    .where(HoldingDaily.date == latest_date)
                    .where(HoldingDaily.quantity > 0)
                ).scalars().all()
                query = query.filter(Instrument.id.in_(held_ids))
                logger.info("only_holdings=True: restricting to %d held instruments", len(held_ids))
            else:
                logger.warning("only_holdings=True but no holdings found — processing nothing")
                return

        result = session.execute(query).all()

        total = len(result)
        start_ts = perf_counter()
        logger.info(
            "Processing %d instruments (first 10: %s)",
            total,
            [r.yahoo_symbol for r in result[:10]],
        )

        processed = 0
        for row in result:
            # Most recent Yahoo earnings date before today (used only for logging context)
            last_earnings_date = next(
                (d for d in sorted(row.earnings.keys(), reverse=True) if d < date.today().isoformat()), None
            )
            if not last_earnings_date:
                logger.warning("No valid earnings date found for %s despite query filter, skipping", row.yahoo_symbol)
                continue

            logger.debug("── %s  (Yahoo last earnings: %s)", row.yahoo_symbol, last_earnings_date)

            get_earnings_report(
                ticker=row.yahoo_symbol,
                cik=row.cik,
                session=session,
                instrument_id=row.id,
            )
            processed += 1
            if processed % 10 == 0 or processed == total:
                elapsed = perf_counter() - start_ts
                logger.info(
                    "Earnings progress: %d/%d processed (elapsed %.1fs)",
                    processed,
                    total,
                    elapsed,
                )

            sleep(1)

        logger.info("Earnings task complete: %d instruments processed in %.1fs", processed, perf_counter() - start_ts)


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
