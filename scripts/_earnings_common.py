"""
Shared helpers for earnings-report ingestion pipelines (US SEC + UK RNS).

Provides:
- Gemini client factory (`_get_genai_client`)
- Pydantic schema (`EarningsReportMetrics` and friends) with an
  `is_earnings_report` gate that callers use to skip non-results disclosures
  (e.g. 6-K board-change press releases, RNS share-buyback announcements).
- HTML → plain-text extraction (`extract_text_from_html`)
- LLM analysis (`summarize_with_llm`) with prompt variants per report type
  (US 10-Q/10-K/20-F/40-F/6-K, UK RNS interim/final/preliminary/annual/trading)
- File-cache helpers and disk-layout constants (`DATA_DIR`, `MODEL`,
  `_check_file_exists_for_date`).

This module is import-only — it does not touch the database. Callers own
selection, persistence, and per-route logging.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, logger

# Silence verbose third-party SDK loggers (AFC status, HTTP request lines).
# Note: google-genai uses "google_genai" (underscore) as its logger namespace.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

DATA_DIR = Path("data/filings")
# GA model — the gemini-3-flash-preview line graduated as 3.5 Flash; preview
# endpoints get retired without notice, so pin the stable id.
MODEL = "gemini-3.5-flash"
# Near-deterministic output: extraction and scoring should not vary run-to-run.
TEMPERATURE = 0.1

# Report type discriminator persisted in EarningsReport.metrics["report_type"]
ReportType = Literal[
    "10-Q",
    "10-K",
    "20-F",
    "40-F",
    "6-K",
    "RNS-INTERIM",
    "RNS-FINAL",
    "RNS-PRELIM",
    "RNS-ANNUAL",
    "RNS-TRADING",
    "PR",
]

# PR (press release) rows are intentionally temporary — superseded by the
# canonical SEC filing in get_earnings_reports.py once it lands on EDGAR.
PR_REPORT_TYPE = "PR"

_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


# ─── Pydantic schema ────────────────────────────────────────────────────────


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
        None,
        description="Guided operating margin % for next period, if explicitly stated (e.g. 'we expect operating margin of ~20%')",
    )
    outlook_commentary: str | None = Field(
        None,
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
        ...,
        description="Overall recommendation: buy (attractive), hold (neutral), avoid (unattractive), consider (worth researching)",
    )
    conviction: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "Strength of the investment signal, not just data clarity. "
            "High = strong, consistent evidence supports the call (e.g. guidance raised, margins expanding, moat intact) — "
            "reserve for the strongest ~10% of reports. "
            "Medium = mixed signals or limited forward visibility. "
            "Low = contradictory data, one-off items dominating, or unclear long-term trajectory."
        ),
    )
    rationale: str = Field(
        "",
        description="2-3 sentence rationale for the recommendation",
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
        description=(
            "Earnings quality for a long-term investor. "
            "Strong = FCF >= reported net income, gross/operating margins stable or expanding, "
            "and disciplined capital use (e.g. moat-building R&D or strategic CapEx aligned with "
            "growth; prudent buybacks or debt paydown that does not starve investment—not financial engineering). "
            "Adequate = FCF roughly tracks net income, margins flat, neutral capital allocation. "
            "Weak = FCF significantly below net income (accruals concern), margin compression, "
            "share dilution, or rising debt without clear return on that investment."
        ),
    )


class EarningsReportMetrics(BaseModel):
    """Structured metrics extracted from earnings report - focused on forward-looking data."""

    is_earnings_report: bool = Field(
        True,
        description=(
            "False if this document is NOT an earnings/results disclosure. "
            "Set False for: SEC 6-Ks announcing only board changes, dividend declarations, "
            "share buybacks, or routine governance matters; RNS announcements like "
            "'Director/PDMR Shareholding', 'Holding(s) in Company', 'Transaction in Own Shares', "
            "'Form 8.3'/'Form 8.5', or any non-results regulatory filing. "
            "When False, the caller should skip persisting this row."
        ),
    )
    guidance: Guidance = Field(..., description="Forward-looking guidance from management - CRITICAL for valuation")
    consensus_comparison: ConsensusComparison | None = None
    investment_assessment: InvestmentAssessment | None = Field(
        None,
        description="Assessment of whether the stock is a good buy based on the report",
    )
    summary: str = Field(
        ...,
        description=(
            "Markdown-formatted summary (350-500 words) with sections: Key Financial Results, "
            "Strategic Highlights, Risks & Headwinds, Future Outlook, Investment Thesis, Bottom Line. "
            "Use bold for key numbers, include specific percentages and comparisons."
        ),
    )


# ─── HTML → text ────────────────────────────────────────────────────────────


def extract_text_from_html(html_content: str) -> str:
    """Parses HTML content to extract clean text for LLM."""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript", "ix:header", "xbrl"]):
        tag.decompose()

    # Get text, use a separator to keep paragraphs distinct
    text = soup.get_text(separator="\n")

    # Strip NUL (0x00) bytes — Postgres rejects them in TEXT/VARCHAR columns
    # and they sometimes appear in RNS HTML payloads (observed on SPX.L
    # 2026-03-10 RNS-FULLYEAR). Other C0 control bytes are harmless to LLMs
    # and to the DB, so we leave them alone.
    text = text.replace("\x00", "")

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

    return "\n".join(lines)


# ─── Cache helpers ──────────────────────────────────────────────────────────


def _check_file_exists_for_date(ticker: str, report_date: str, report_type: str = "") -> bool:
    """Check if any filing file exists for the given date (handles form variations)."""
    ticker_dir = DATA_DIR / ticker
    if not ticker_dir.exists():
        return False

    prefix = f"{report_date}_{report_type}" if report_type else f"{report_date}_"

    for file_path in ticker_dir.iterdir():
        if file_path.is_file() and file_path.name.startswith(prefix) and file_path.suffix == ".html":
            return True
    return False


# ─── LLM analysis ───────────────────────────────────────────────────────────


# Default prompt body shared across all routes. {ticker}/{report_type}/{period}
# are bound by the caller; {regional_notes} adds report-type-specific guidance.
_BASE_PROMPT = """
You are a financial analyst helping a retail investor decide whether {ticker} is a
good long-term investment. This investor holds stocks in a tax-free ISA account with
a 10+ year horizon. They are growth and quality investors — they want businesses that
can compound capital over a decade, not short-term trades. Assess from a long-term
compounding perspective: a weak next quarter that doesn't damage the long-term thesis
is not a reason to avoid; deteriorating ROIC or returns on capital (only when stated
or clearly inferable from the filing), moat erosion, or management capital
misallocation IS a reason to avoid.

Analyze this {report_type} report (period: {period}) and extract structured metrics
while generating a comprehensive summary.

{regional_notes}

**CRITICAL: EPS Guidance is the highest priority** - Stock price = EPS × PE ratio. If EPS guidance grows 10%, stock should theoretically grow 10% with same PE.

**Extract the following structured data:**

0. **is_earnings_report (REQUIRED):** True if this document is an earnings/results
   disclosure (10-Q, 10-K, 20-F, 40-F, 6-K results, RNS half-year/final/preliminary/
   annual results, RNS trading updates with revenue figures). False for governance/
   capital-action filings (board changes, dividend declarations alone, share
   buybacks, director shareholdings, holdings notifications, Form 8.3/8.5).
   When False, populate the rest of the fields with safe defaults and a 1-line
   summary explaining why this is not an earnings report.

1. **Guidance (MOST IMPORTANT - REQUIRED):**
   - Scan ALL of: "Outlook", "Guidance", "Forecast", "Expectations", "Financial Outlook", "Business Outlook", "Looking Ahead", MD&A forward-looking sections.
   - **EPS guidance** (currency-neutral; preserve the report's stated unit — UK companies often quote EPS in pence, e.g. "178.5p"): patterns like "expects EPS of X.XX", "forecasts EPS between X and Y", "diluted EPS guidance of X.XX–Y.YY".
     Calculate growth_pct vs prior year actual EPS if available.
   - **Revenue guidance**: extract in millions for next quarter and full year; calculate growth_pct if prior period available.
   - **Operating margin guidance**: capture if stated explicitly (e.g. "operating margin of approximately 20%", "adjusted EBIT margin ~15%").
   - **Outlook commentary**: if no specific numbers exist, write 1–2 sentences
     summarising management's qualitative forward outlook
     (e.g. "expects double-digit revenue growth driven by cloud segment").
   - **IMPORTANT**: For numeric fields, only populate from explicitly stated numbers. Do NOT infer or estimate numbers not in the text.

2. **Consensus Comparison (if available):**
   - Look for "consensus", "analyst estimates", "Street estimates", "beats", "misses"
   - Extract revenue_beat_pct and eps_beat_pct; note guidance_vs_consensus (above/below/in-line)
   - Only extract if explicitly mentioned. Do NOT calculate if consensus is not stated.

3. **Investment Assessment (REQUIRED - answer "is this a good buy?"):**
   - **recommendation**: "buy" (attractive), "hold" (neutral), "avoid" (unattractive), or "consider" (worth researching)
   - **conviction**: "high", "medium", or "low" — strength of the investment signal (consistent evidence vs mixed vs contradictory), not merely how clear the filing text is
   - **Calibration**: most filings describe routine quarters. Across a large sample expect
     roughly 20% "buy", 30% "consider", 35% "hold", 15% "avoid" — an in-line quarter with no
     change to the long-term thesis is a "hold", not a "buy". Reserve "buy" for concrete
     positive evidence in THIS report (accelerating growth, expanding margins or returns on
     capital, raised guidance) and "high" conviction for the strongest ~10% of reports.
     Never pair "buy" with "high" conviction unless key_concerns are identified and clearly
     outweighed by the evidence. Companies narrate their own results favourably — judge the
     numbers, not the framing.
   - **rationale**: 2-3 sentences explaining why (e.g., "Strong guidance raise and margin expansion support buy. Debt paydown reduces risk.")
   - **key_catalysts**: 2-4 positive drivers (e.g., "AI revenue accelerating", "Market share gains", "Buyback program")
   - **key_concerns**: 2-4 risks (e.g., "Macro headwinds", "Customer concentration", "Margin pressure")
   - **quality_signals**: use the schema rubric (FCF vs net income, margins, capital allocation, dilution/debt)
   - **capital_allocation** (add to rationale if relevant): Is the company reinvesting in moat-building (R&D, capacity)? Buying back shares or diluting? Taking on debt for growth or for financial engineering?
   - Base assessment on: long-term compounding potential, earnings quality, moat durability, management capital allocation, risk/reward over 10 years — NOT short-term price momentum

**Generate Summary (Markdown, 350-500 words):**

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
- Use millions as unit (e.g., 13,640 million not 13.64 billion); preserve the report's reporting currency symbol ($, £, €, …) — do NOT convert
- Be specific with numbers; if a metric is not available, use null

--- REPORT TEXT ---
{text}
"""


_REGIONAL_NOTES_US = ""

_REGIONAL_NOTES_FPI = """
**Foreign private issuer notes:**
- This company files with the SEC as a foreign private issuer.
- 20-F is the annual report (US 10-K equivalent). 6-K covers any other interim disclosure (results, press releases, governance).
- Many of these companies report only half-yearly or annually — there may be no "next quarter" guidance.
- Outlook sections may quote currencies other than USD (EUR, GBP, JPY). Do not force conversion to USD.
"""

_REGIONAL_NOTES_PR = """
**Earnings press-release notes:**
- This is a same-day earnings press release (typically issued on the day of the earnings call), NOT a regulatory filing.
- The corresponding SEC 10-Q / 10-K / 20-F / 6-K filing usually lands 2-4 weeks later with full GAAP detail; this row is intended to be superseded by that SEC row.
- Press releases focus on headline P&L, segment commentary, and forward guidance. Balance-sheet detail is often limited or absent.
- Numbers may be preliminary; minor restatements occasionally appear in the eventual SEC filing.
- Currency, EPS units, and accounting basis (GAAP / non-GAAP / adjusted) follow the company's filing convention — preserve them as stated.
"""

_REGIONAL_NOTES_RNS = """
**UK RNS (LSE) notes:**
- This is a UK regulatory news announcement, not a SEC filing.
- **Reporting currency**: Default is GBP (£). Several dual-listed FTSE 100 names report in USD ($) (e.g. SHEL, BP, AAL, GLEN, RIO, AZN); a few in EUR (€). Preserve the report's native currency in the summary — do NOT convert to GBP.
- **EPS is usually quoted in pence (p), not pounds.** "EPS of 178.5p" means 178.5 pence per share (≈ £1.785). Keep the unit exactly as the report states it; if the report uses pence, write "178.5p" in the summary and populate the schema's eps_guidance.next_year as 178.5 (the numeric value in pence). Do NOT silently convert pence to pounds.
- **Adjusted / Underlying / Headline measures dominate UK reporting.** When both statutory (IFRS) and adjusted (or "underlying" / "headline" / "pre-exceptional") figures are given, the adjusted figure is what management is signalling on and what consensus tracks. Prefer the adjusted figure for eps_guidance and revenue_guidance, and call out the statutory-vs-adjusted gap in the summary if material (especially if adjusting items look recurring rather than one-off).
- **"Like-for-like" (LFL) sales growth** is the standard UK retail/consumer organic-growth metric (e.g. "LFL sales growth of 3.2%"). Capture it as a top-line signal in the summary; it is roughly equivalent to US "comparable sales" or "same-store sales".
- UK companies typically report half-yearly: "Interim Results" / "Half Year Results" (H1, six months ended …) and "Final Results" / "Preliminary Results" (full year, twelve months ended …). The schema's eps_guidance.next_quarter and revenue_guidance.next_quarter fields will almost always be null for UK reports — populate next_year instead.
- **The `period` parameter passed to this prompt is the RNS announcement date, NOT the financial period-end** (announcement is typically 1–3 months after period-end). Derive the actual reporting period (e.g. "six months ended 30 June 2025", "year ended 31 December 2024") from the report text and use that in the summary.
- "Trading Update" / "Trading Statement" announcements often only contain like-for-like sales growth and qualitative commentary, not full P&L. Treat these as forward-looking signals (is_earnings_report=True) but expect consensus_comparison to be null.
- "Annual Report and Accounts" is the full glossy annual report (different from "Final Results" — Final is the press release with the headline P&L, Annual Report is the full document).
- IFRS terminology: "operating profit" not "operating income"; "Profit before tax" (PBT) not "Pre-tax income"; "Profit attributable to equity holders" not "Net income"; "EPS" or "Earnings per share" (basic / diluted / adjusted / underlying).
- **Dividend cadence**: UK companies typically pay an interim dividend (announced with H1 Interim Results) and a final dividend (announced with Final/Preliminary Results, paid after AGM approval). DPS is a meaningful capital-allocation signal — capture both interim and final amounts if stated, plus the implied full-year DPS and any progressive-dividend policy commentary.
"""


def _regional_notes_for(report_type: str) -> str:
    if report_type == "PR":
        return _REGIONAL_NOTES_PR
    if report_type in ("20-F", "40-F", "6-K"):
        return _REGIONAL_NOTES_FPI
    if report_type.startswith("RNS-"):
        return _REGIONAL_NOTES_RNS
    return _REGIONAL_NOTES_US


def summarize_with_llm(
    text: str,
    ticker: str,
    report_type: str,
    period: str,
) -> dict[str, Any] | None:
    """
    Extract structured metrics and generate summary from earnings/RNS report text.

    Args:
        text: Plain-text body of the report (already HTML-stripped).
        ticker: Yahoo Finance symbol (used in the prompt, not for fetch).
        report_type: One of the ReportType literal values (10-Q, RNS-INTERIM, …).
            Selects the regional note appended to the prompt.
        period: ISO date or human-readable period for context (e.g. "2025-09-30").

    Returns dict with structured metrics plus `summary`, or None on API error.
    Caller is expected to inspect `result["is_earnings_report"]` and skip
    persistence when False.
    """
    logger.info(
        "Analyzing %s for %s period %s (%d characters)",
        report_type,
        ticker,
        period,
        len(text),
    )

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set in config.py or environment variables")
        return None

    try:
        client = _get_genai_client()

        max_chars = 500000
        if len(text) > max_chars:
            logger.info("Truncating text from %d to %d characters", len(text), max_chars)
            text = text[:max_chars]

        prompt = _BASE_PROMPT.format(
            ticker=ticker,
            report_type=report_type,
            period=period,
            regional_notes=_regional_notes_for(report_type),
            text=text,
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "temperature": TEMPERATURE,
                "response_mime_type": "application/json",
                "response_json_schema": EarningsReportMetrics.model_json_schema(),
            },
        )

        result_dict = json.loads(response.text)
        validated_result = EarningsReportMetrics.model_validate(result_dict)
        return validated_result.model_dump()

    except Exception as e:
        logger.error("Error calling Gemini API: %s", e, exc_info=True)
        return None


def summarize_pdf_with_llm(
    pdf_bytes: bytes,
    ticker: str,
    report_type: str,
    period: str,
) -> dict[str, Any] | None:
    """Run the same extraction pipeline as ``summarize_with_llm`` but with a PDF source.

    Sends the PDF directly to Gemini's multimodal endpoint instead of pre-extracting
    text. Lets the model own table/multi-column layout parsing, which is where local
    PDF extractors (pypdf / pdfplumber) typically misread financial tables.

    Args:
        pdf_bytes: Raw PDF content (already downloaded by the caller).
        ticker: Yahoo Finance symbol (used in the prompt only).
        report_type: One of the ReportType literal values; selects regional notes.
        period: ISO date string used as period context in the prompt.

    Returns the same dict shape as ``summarize_with_llm``, or None on API error.
    """
    logger.info(
        "Analyzing %s PDF for %s period %s (%d bytes)",
        report_type,
        ticker,
        period,
        len(pdf_bytes),
    )

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set in config.py or environment variables")
        return None

    try:
        client = _get_genai_client()

        # Reuse the shared prompt; the {text} slot is just a pointer to the attached
        # PDF since the document is sent as a separate Part.
        prompt = _BASE_PROMPT.format(
            ticker=ticker,
            report_type=report_type,
            period=period,
            regional_notes=_regional_notes_for(report_type),
            text="(See the attached PDF for the full report. Extract all metrics from it.)",
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt,
            ],
            config={
                "temperature": TEMPERATURE,
                "response_mime_type": "application/json",
                "response_json_schema": EarningsReportMetrics.model_json_schema(),
            },
        )

        result_dict = json.loads(response.text)
        validated_result = EarningsReportMetrics.model_validate(result_dict)
        return validated_result.model_dump()

    except Exception as e:
        logger.error("Error calling Gemini API: %s", e, exc_info=True)
        return None
