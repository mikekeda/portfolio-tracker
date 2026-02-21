"""
Fetches the latest 10-Q or 10-K filing for a given ticker from the SEC EDGAR API
"""

import json
import os
from datetime import date
from pathlib import Path
from time import sleep
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup
from google import genai
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.sql import text as sql_text, text

from config import GEMINI_API_KEY, logger
from models import EarningsReport, Instrument, InstrumentYahoo
from scripts.update_data import get_session

# SEC requires a User-Agent in the format: "Company Name email@example.com"
# TODO: Replace with your actual contact info
USER_AGENT = "PortfolioTracker/1.0 (admin@example.com)"

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}

# SEC Endpoints
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_document}"

# Local Storage Configuration
DATA_DIR = Path("data/filings")
MODEL = "gemini-3-flash-preview"


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


def get_latest_filing_metadata(cik: str, form_types: tuple[str] = ("10-Q", "10-K")):
    """Fetches filing metadata for the given CIK and form types."""

    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    data = response.json()

    filings = data["filings"]["recent"]

    # Collect all matching filings
    for i, form in enumerate(filings["form"]):
        if form in form_types:
            # Return the most recent one

            return {
                "accessionNumber": filings["accessionNumber"][i],
                "primaryDocument": filings["primaryDocument"][i],
                "form": form,
                "reportDate": filings["reportDate"][i],
            }

    return None


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
        client = genai.Client(api_key=GEMINI_API_KEY)

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
           - Look for EPS guidance in sections like "Outlook", "Guidance", "Forecast", "Expectations", "Q4 2025", "FY 2025", etc.
           - EPS guidance may be stated as: "expects EPS of $X.XX", "forecasts EPS between $X and $Y", "guidance of $X.XX"
           - Extract next quarter EPS guidance if provided; extract full year EPS guidance if provided
           - Calculate growth_pct by comparing guidance to prior year actual EPS (if available)
           - Revenue guidance: extract in millions for next quarter and full year; calculate growth_pct if prior period available
           - **IMPORTANT**: If management does NOT provide specific guidance numbers, set all guidance fields to null. Do NOT infer or estimate.

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

        eps_guidance = (result.get("guidance") or {}).get("eps_guidance") or {}
        assessment = result.get("investment_assessment") or {}
        logger.info(
            f"Extracted structured data for {ticker} - EPS guidance: next_q={eps_guidance.get('next_quarter')}, "
            f"next_y={eps_guidance.get('next_year')}, growth={eps_guidance.get('growth_pct')}% | "
            f"Assessment: {assessment.get('recommendation', 'N/A')} ({assessment.get('conviction', 'N/A')})"
        )
        return result

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return None


def get_earnings_report(ticker: str, cik: str, last_earnings_date: str, session, instrument_id: int):
    """Fetches, processes, and saves earnings report for a given ticker."""

    # 1. Get Filing (Latest or Closest to Date)
    metadata = get_latest_filing_metadata(cik)

    if not metadata:
        logger.warning("No ('10-Q', '10-K') filings found for %s CIK %s", ticker, cik)
        return None

    report_date = metadata["reportDate"]

    # Don't download older reports (ISO dates can be compared as strings)
    (DATA_DIR / ticker).mkdir(parents=True, exist_ok=True)  # ensure dir exists
    existing_filings = os.listdir(DATA_DIR / ticker)
    if metadata["reportDate"] < last_earnings_date and len(existing_filings) > 0:
        logger.info(
            "No new filings found for %s (needed %s, we have: %s)", ticker, last_earnings_date, existing_filings
        )
        return None

    # Check if we already have this report in the database
    existing_report = session.execute(
        select(EarningsReport).filter(
            EarningsReport.instrument_id == instrument_id, EarningsReport.date == date.fromisoformat(report_date)
        )
    ).scalar_one_or_none()

    if existing_report:
        logger.info("Earnings report for %s on %s already exists in database, skipping", ticker, report_date)
        return existing_report

    # 2. Get HTML (Download or Cache)
    html_content = get_filing_html(cik, ticker, metadata)

    # 3. Extract Text
    text = extract_text_from_html(html_content)

    # 4. Extract structured metrics and generate summary with LLM
    result = summarize_with_llm(text, ticker, metadata["form"])

    if result is None:
        logger.warning("Failed to generate summary for %s %s, skipping database save", ticker, report_date)
        return None

    # Extract summary and metrics from structured output
    summary = result.get("summary", "")
    metrics = {k: v for k, v in result.items() if k != "summary"}  # All fields except summary

    # 5. Save to database
    earnings_report = EarningsReport(
        instrument_id=instrument_id,
        date=date.fromisoformat(report_date),
        summary=summary,
        metrics=metrics,  # Store structured metrics
    )
    session.add(earnings_report)
    session.commit()

    # Log key metrics
    eps_guidance = metrics.get("guidance", {}).get("eps_guidance", {})
    if eps_guidance:
        logger.info(
            "Saved earnings report for %s on %s - EPS guidance: next_q=%s, next_y=%s, growth=%s%%",
            ticker,
            report_date,
            eps_guidance.get("next_quarter"),
            eps_guidance.get("next_year"),
            eps_guidance.get("growth_pct"),
        )
    else:
        logger.info("Saved earnings report for %s on %s to database", ticker, report_date)

    return earnings_report


def _check_file_exists_for_date(ticker: str, report_date: str) -> bool:
    """Check if any filing file exists for the given date (handles form variations like 10-Q, 10-Q/A, etc.)."""
    ticker_dir = DATA_DIR / ticker
    if not ticker_dir.exists():
        return False

    # Check for files starting with the date (handles 10-Q, 10-Q/A, 10-K, etc.)
    for file_path in ticker_dir.iterdir():
        if file_path.is_file() and file_path.name.startswith(f"{report_date}_") and file_path.suffix == ".html":
            return True
    return False


def get_earnings_reports(limit: int = 100):
    """
    Fetch and process earnings reports for instruments with earnings data.

    Prioritizes tickers that:
    1. Don't have existing reports in database yet (has_reports = 0 first)
    2. Have more recent earnings dates (more likely to have new filings)
    """

    with get_session() as session:
        # Get max earnings date from JSONB keys (most recent earnings date that's before today)
        # This ensures we only consider valid dates for processing
        today_iso = date.today().isoformat()

        max_earnings_date_expr = (
            select(func.max(sql_text("date")))
            .select_from(func.jsonb_object_keys(InstrumentYahoo.earnings).alias("date"))
            .where(text("date < :today_iso").bindparams(today_iso=today_iso))  # Only consider dates before today
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

        result = session.execute(
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
                max_earnings_date_expr.is_not(None),  # Must have at least one valid earnings date (before today)
            )
            .order_by(
                has_reports.asc(),  # Prioritize instruments without any reports yet
                max_earnings_date_expr.desc().nulls_last(),  # Then by most recent earnings date
            )
            .limit(limit)
        ).all()

        logger.info("Getting the %d recent earnings reports: %s", limit, [r.yahoo_symbol for r in result])

        for row in result:
            # Find the most recent earnings date that's before today
            last_earnings_date = next(
                (d for d in sorted(row.earnings.keys(), reverse=True) if d < date.today().isoformat()), None
            )
            # This should never be None due to the query filter, but keep as safety check
            if not last_earnings_date:
                logger.warning("No valid earnings date found for %s despite query filter, skipping", row.yahoo_symbol)
                continue

            # Check if file already exists (handles form variations)
            if _check_file_exists_for_date(row.yahoo_symbol, last_earnings_date):
                continue

            logger.info("Fetching filings for %s (last earnings: %s)", row.yahoo_symbol, last_earnings_date)

            get_earnings_report(
                ticker=row.yahoo_symbol,
                cik=row.cik,
                last_earnings_date=last_earnings_date,
                session=session,
                instrument_id=row.id,
            )

            sleep(1)


if __name__ == "__main__":
    get_earnings_reports(10)
