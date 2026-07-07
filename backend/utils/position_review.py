"""
LLM-driven position review for held instruments.

Builds structured context from portfolio holdings + user thesis, calls Gemini,
and returns a validated assessment payload. Persistence is handled by
scripts/run_position_review.py.
"""

import hashlib
import json
import logging
import math
from datetime import date, datetime
from typing import Any, Literal

from dateutil.relativedelta import relativedelta
from google import genai
from pydantic import BaseModel, Field

from backend.schemas.instrument_thesis import InstrumentThesisSchema
from config import GEMINI_API_KEY, TIMEZONE, logger
from models import EarningsReport, Instrument, MarketMetricsDaily

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# GA model — the gemini-3-flash-preview line graduated as 3.5 Flash; preview
# endpoints get retired without notice, so pin the stable id.
MODEL = "gemini-3.5-flash"
# Near-deterministic output: assessments should not vary run-to-run.
TEMPERATURE = 0.1
# A valid payload is ~1-2k tokens; the cap kills runaway generations (observed:
# a 64KB response with an invalid \uXXXX escape on PARRO.PA, 2026-07-06) while
# leaving headroom for the model's internal thinking tokens.
MAX_OUTPUT_TOKENS = 8192

_EARNINGS_SUMMARY_MAX_CHARS = 1500
_RECENT_EARNINGS_LIMIT = 3

PositionAction = Literal["no_action", "hold", "add", "trim", "exit"]
SignalStrength = Literal["none", "low", "medium", "high"]
ThesisStatus = Literal["intact", "eroding", "broken", "insufficient_data"]
TriggerStatusVerdict = Literal["met", "not_met", "pending", "insufficient_data"]
TrendDir = Literal["improving", "stable", "deteriorating", "unknown"]


class TriggerStatus(BaseModel):
    trigger_type: Literal["buy", "sell"]
    trigger_text: str
    status: TriggerStatusVerdict
    evidence: str


class QualityTrajectory(BaseModel):
    roic_trend: TrendDir
    margin_trend: TrendDir
    revenue_growth_trend: TrendDir
    notes: str


class ValuationContext(BaseModel):
    pe_vs_history: Literal["discount", "fair", "premium", "unknown"]
    dcf_note: str
    notes: str


class TimeHorizonAlignment(BaseModel):
    fits_horizon: bool
    notes: str


class PositionAssessmentSchema(BaseModel):
    """Structured Gemini output for a single instrument position review."""

    position_action: PositionAction
    signal_strength: SignalStrength
    thesis_status: ThesisStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    trigger_evaluations: list[TriggerStatus] = Field(default_factory=list)
    quality_trajectory: QualityTrajectory
    valuation_context: ValuationContext
    risk_flags: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    time_horizon_alignment: TimeHorizonAlignment
    rationale: str
    # Generated last so it summarises completed reasoning; shown in UI tooltips.
    headline: str = Field(
        ...,
        description="One line (<=140 chars): bottom-line action plus its primary driver, citing one specific number from the context",
    )


_PROMPT_HEADER = """\
You are a financial analyst supporting a retail investor with a UK ISA portfolio
and a 10+ year horizon. They are growth and quality investors — they want
businesses that compound capital over a decade. Your task is to assess whether
their existing position in {ticker} should be acted on, given the provided
context.

This assessment is NOT a buy/sell oracle. Default to silence. Five hard rules
override anything else you have been trained to do:

1. CITE ONLY PROVIDED NUMBERS. If a field is null in the context, say "null"
   or "unknown" explicitly. Never invent a number, never recall a number from
   your training data. If the holding dict says `roic_pct: null`, you cannot
   claim ROIC is 25% from memory.

2. DEFAULT TO SILENCE. `position_action` is one of `no_action`, `hold`, `add`,
   `trim`, `exit`. Most assessments are `hold`. Only emit `add` / `trim` /
   `exit` when (a) you would defend the call for 12+ months on a 10-year
   horizon, AND (b) `signal_strength` is `high`, AND (c) at least three
   independent signals in the context point in the same direction. If signals
   are mixed, contradict each other, or hinge on a fact you cannot verify in
   the context, the answer is `hold`. If there is not enough data to form a
   view, return `no_action` with `signal_strength: none`.

3. CONTRARIAN REWEIGHT. Read the `macro` section first.
     - If `vix` is low and `fear_greed_index` is in greed/extreme-greed
       (>= 60): apply extra skepticism to bullish actions, discount
       narrative-driven catalysts, raise the bar for `add`.
     - If `vix` is high and `fear_greed_index` is in fear/extreme-fear
       (<= 40): apply extra weight to durable fundamentals, discount
       narrative-driven concerns, raise the bar for `trim` / `exit`.
   Capitulation makes weak fundamentals look terminal; euphoria makes weak
   fundamentals look temporary. Resist the prevailing mood. Note the
   reweight you applied in `rationale`.

4. THESIS IS THE PRIMARY LENS. When `thesis` is non-null, weigh the data against the
   investor's own frame: `summary`, `horizon_years`, `conviction`, `buy_triggers`,
   and `sell_triggers` — not generic best practice.

   ALLOCATION BAND (required fields): `target_weight_min_pct` and `target_weight_max_pct`
   are the user's acceptable weight range as **percent of total portfolio value**, same
   unit and scale as `position.portfolio_pct` (0–100; e.g. 8 means 8%). The band is valid
   only when `target_weight_min_pct <= target_weight_max_pct` (validated upstream).

   TRIGGER EVALUATION: For each string in `buy_triggers` and `sell_triggers`, decide
   whether the context supports the condition:
     - condition clearly holds -> `met`
     - condition clearly fails -> `not_met`
     - cannot verify from context -> `insufficient_data`

   If ANY sell-side trigger is `met`, `thesis_status` MUST NOT be `intact`; choose
   `eroding` or `broken` by severity.

   BAND VS CURRENT SIZE (use when `position.portfolio_pct` is non-null): the band
   is the investor's advisory comfort range, NOT a hard rule. If `portfolio_pct`
   is **above** `target_weight_max_pct`, flag the breach in `risk_flags` and note
   it in `rationale`; recommend `trim` only when fundamentals or sell-side
   triggers corroborate (rule 2 applies — never `trim` on the breach alone). If
   `portfolio_pct` is **below** `target_weight_min_pct` AND `buy_triggers` are
   substantially `met`, consider an `add`. If `portfolio_pct` is null or unknown,
   do not rely on band-vs-size logic; use triggers and fundamentals only.

   When `thesis_rule_eval` is present, treat fired numeric rules (`buy_rules_met`,
   `sell_rules_met`) and `allocation_status` as factual inputs — do not re-derive them.

   If `thesis` is null in context, set `thesis_status` to `insufficient_data`.

5. NO EXTERNAL KNOWLEDGE. If a fact you would need is not in the context,
   say "insufficient data" rather than recall it from training. This includes
   news, peer comparisons, recent management changes, etc.

This investor uses ROIC over ROE (leverage-neutral). Quality matters more
than short-term beats: a weak quarter that does not damage the long-term
thesis is not a reason to trim. Deteriorating ROIC, moat erosion, or capital
misallocation IS a reason to trim or exit. DCF is a supporting signal only —
for high-multiple growth stocks the FCF-based DCF is unreliable; if the
context includes DCF figures for a high-multiple growth name, caveat them in
`valuation_context.dcf_note` rather than relying on them.

Generate the structured fields and a markdown `rationale` (~150-300 words)
covering: thesis status, quality trajectory, valuation in context, specific
risk flags grounded in the provided data, catalysts, the contrarian reweight
applied, and the bottom-line action with conviction. Use **bold** for key
numbers; cite specific values from the context (e.g. "**ROIC 28.5%**, up from
22.1% three years ago").

Finish with `headline`: ONE plain-text line (max 140 characters) stating the
action and its single primary driver, citing one specific number from the
context — e.g. "Add: fwd P/E 16 vs 5-yr avg 31 while revenue accelerates 18%".
No markdown, no hedging filler.

--- CONTEXT (JSON) ---
"""

_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _pe_history_summary(pes: dict | None) -> dict[str, Any]:
    out: dict[str, Any] = {"min_5y": None, "avg_5y": None, "max_5y": None, "samples": 0}
    if not pes:
        return out
    today = datetime.now(TIMEZONE).date()
    cutoff = today + relativedelta(years=-5)
    values: list[float] = []
    for k, v in pes.items():
        try:
            d = date.fromisoformat(k[:10])
        except (ValueError, TypeError):
            continue
        # d > today: forward estimate rows must not enter the 5y history stats
        if d < cutoff or d > today:
            continue
        try:
            pe = float(v.get("pe_ratio")) if isinstance(v, dict) else float(v)
        except (TypeError, ValueError):
            continue
        if pe > 0 and math.isfinite(pe):
            values.append(pe)
    if not values:
        return out
    out["min_5y"] = round(min(values), 2)
    out["max_5y"] = round(max(values), 2)
    out["avg_5y"] = round(sum(values) / len(values), 2)
    out["samples"] = len(values)
    return out


def _summarize_earnings(reports: list[EarningsReport]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in reports[:_RECENT_EARNINGS_LIMIT]:
        metrics = r.metrics or {}
        assessment = metrics.get("investment_assessment") or {}
        summary = (r.summary or "")[:_EARNINGS_SUMMARY_MAX_CHARS]
        out.append({
            "date": r.date.isoformat() if r.date else None,
            "report_type": metrics.get("report_type"),
            "investment_assessment": {
                "recommendation": assessment.get("recommendation"),
                "conviction": assessment.get("conviction"),
                "rationale": assessment.get("rationale"),
                "key_catalysts": assessment.get("key_catalysts"),
                "key_concerns": assessment.get("key_concerns"),
                "quality_signals": assessment.get("quality_signals"),
            },
            "summary_excerpt": summary,
        })
    return out


def _macro_context(row: MarketMetricsDaily | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        "date": row.date.isoformat() if row.date else None,
        "vix": row.vix,
        "fear_greed_index": row.fear_greed_index,
        "buffett_indicator": row.buffett_indicator,
        "yield_spread": row.yield_spread,
        "market_breadth_indicator": row.market_breadth_indicator,
        "sp500_above_sma200": row.sp500_above_sma200,
        "consumer_sentiment": row.consumer_sentiment,
        "real_yield_10y": row.real_yield_10y,
        "hy_oas": row.hy_oas,
    }


def build_context(
    *,
    holding: dict,
    instrument: Instrument,
    thesis: InstrumentThesisSchema | None,
    macro: MarketMetricsDaily | None,
) -> dict[str, Any]:
    """Build the canonical input dict the LLM sees for one held instrument."""
    info = (instrument.yahoo.info or {}) if instrument.yahoo else {}
    pe_history = _pe_history_summary((instrument.yahoo.pes or {}) if instrument.yahoo else None)

    sorted_earnings = sorted(
        instrument.earnings_reports or [],
        key=lambda r: r.date or date.min,
        reverse=True,
    )

    operating_margin = info.get("operatingMargins")
    operating_margin_pct = operating_margin * 100 if operating_margin is not None else None

    gross_margin = info.get("grossMargins")
    gross_margin_pct = gross_margin * 100 if gross_margin is not None else None

    return {
        "ticker": instrument.yahoo_symbol,
        "name": instrument.name,
        "sector": holding.get("sector") or info.get("sector"),
        "industry": info.get("industry"),
        "country": holding.get("country") or info.get("country"),
        "currency": holding.get("currency"),
        "quote_type": holding.get("quote_type"),
        "position": {
            "quantity": holding.get("quantity"),
            "avg_price": holding.get("avg_price"),
            "current_price": holding.get("current_price"),
            "market_value_gbp": round(holding.get("market_value") or 0.0, 2),
            "unrealised_ppl_pct": holding.get("return_pct"),
            "portfolio_pct": holding.get("portfolio_pct"),
        },
        "fundamentals": {
            "roic_pct": holding.get("roic"),
            "operating_margin_pct": operating_margin_pct,
            "profit_margin_pct": holding.get("profit_margins"),
            "gross_margin_pct": gross_margin_pct,
            "revenue_growth_yoy_pct": holding.get("revenue_growth"),
            "free_cashflow_yield_pct": holding.get("free_cashflow_yield"),
            "debt_to_equity": holding.get("debtToEquity"),
            "f_score": holding.get("f_score"),
            "rule_of_40_score": holding.get("rule_of_40_score"),
            "market_cap": holding.get("market_cap"),
            "total_revenue": holding.get("total_revenue"),
        },
        "valuation": {
            "pe_ratio": holding.get("pe_ratio"),
            "forward_pe_ratio": holding.get("forward_pe_ratio"),
            "peg_ratio": holding.get("peg_ratio"),
            "ps_ratio": holding.get("ps_ratio"),
            "pe_5y_avg": holding.get("avg_pe"),
            "pe_5y_min": pe_history["min_5y"],
            "pe_5y_max": pe_history["max_5y"],
            "pe_5y_samples": pe_history["samples"],
            "dcf_price": holding.get("dcf_price"),
            "dcf_diff": holding.get("dcf_diff"),
            "dcf_implied_growth": holding.get("dcf_implied_growth"),
            "fifty_two_week_high_distance": holding.get("fifty_two_week_high_distance"),
            "fifty_two_week_change": holding.get("fifty_two_week_change"),
        },
        "technical": {
            "rsi": holding.get("rsi"),
            "sma_20": holding.get("sma_20"),
            "sma_50": holding.get("sma_50"),
            "sma_200": holding.get("sma_200"),
            "rs_6m_vs_spy": holding.get("rs_6m_vs_spy"),
        },
        "sentiment": {
            "analyst_recommendation_mean": holding.get("recommendation_mean"),
            "analyst_recommendation_key": holding.get("recommendation_key"),
            "analyst_target_upside_pct": holding.get("prediction"),
            "number_of_analyst_opinions": holding.get("number_of_analyst_opinions"),
            "form13f_score": holding.get("form13f_score"),
            "insider_buy_count_90d": holding.get("insider_buy_count_90d"),
            "insider_sell_count_90d": holding.get("insider_sell_count_90d"),
            "insider_net_value_90d_usd": holding.get("insider_net_value_90d"),
            "institutional_ownership_pct": holding.get("institutional_ownership"),
            "short_percent_of_float": holding.get("short_percent_of_float"),
        },
        "screener": {
            "passed": holding.get("passedScreeners") or [],
            "score": holding.get("screener_score"),
        },
        "recent_earnings": _summarize_earnings(sorted_earnings),
        "macro": _macro_context(macro),
        "thesis": thesis.model_dump() if thesis else None,
        "thesis_rule_eval": holding["thesis_rule_eval"],
    }


def _round_floats(obj: Any, decimals: int = 2) -> Any:
    if isinstance(obj, float):
        if math.isfinite(obj):
            return round(obj, decimals)
        return None
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


# Bump to force portfolio-wide review regeneration on the next run (busts every
# stored inputs_hash) after schema or prompt changes that alter the payload.
_CONTEXT_VERSION = 2


def _slow_inputs(ctx: dict) -> dict:
    """Subset of the context whose change warrants a fresh review.

    Deliberately excludes price-driven noise — position value, technicals,
    valuation prints, raw macro — which would re-review every holding every
    day. Numeric conditions the investor cares about still bust the cache via
    the rule state the moment a rule fires or clears.
    """
    # market_cap and free_cashflow_yield move with the share price daily
    fundamentals = {
        k: v
        for k, v in ctx["fundamentals"].items()
        if k not in ("market_cap", "total_revenue", "free_cashflow_yield")
    }
    rule_eval = ctx["thesis_rule_eval"]
    macro = ctx["macro"]
    fg = macro.get("fear_greed_index")
    vix = macro.get("vix")
    return {
        "context_version": _CONTEXT_VERSION,
        "ticker": ctx["ticker"],
        "thesis": ctx["thesis"],
        "rule_state": {
            # descriptions only: eval reason strings embed live prices/percentages
            "buy_rules_met": sorted(r["description"] for r in rule_eval["buy_rules_met"]),
            "sell_rules_met": sorted(r["description"] for r in rule_eval["sell_rules_met"]),
            "allocation_status": rule_eval["allocation_status"],
        },
        "fundamentals": fundamentals,
        "recent_earnings": [
            {
                "date": r["date"],
                "report_type": r["report_type"],
                "recommendation": r["investment_assessment"]["recommendation"],
            }
            for r in ctx["recent_earnings"]
        ],
        # Regime buckets, not raw prints — the contrarian reweight reacts to
        # regime shifts (fear/neutral/greed, calm/elevated vol), not daily wiggles.
        "macro_regime": {
            "fear_greed": None if fg is None else ("fear" if fg <= 40 else "greed" if fg >= 60 else "neutral"),
            "vix_elevated": None if vix is None else bool(vix >= 25),
        },
    }


def hash_context(ctx: dict) -> str:
    """sha256 over the slow-moving input subset (see _slow_inputs)."""
    rounded = _round_floats(_slow_inputs(ctx), decimals=1)
    canonical = json.dumps(rounded, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_assessment(ctx: dict, *, ticker: str) -> dict[str, Any] | None:
    """Run Gemini for one instrument context. Returns parsed payload or None."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set in config.py or environment variables")
        return None

    try:
        client = _get_genai_client()
        prompt = _PROMPT_HEADER.format(ticker=ticker) + "\n" + json.dumps(ctx, indent=2, default=str)

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "response_mime_type": "application/json",
                "response_json_schema": PositionAssessmentSchema.model_json_schema(),
            },
        )

        try:
            result_dict = json.loads(response.text)
        except json.JSONDecodeError as e:
            # Structured output occasionally comes back malformed; one regeneration
            # usually fixes it. A second failure falls to the outer handler and the
            # nightly hash gate retries the instrument tomorrow.
            logger.warning("%s: invalid JSON from Gemini (%s) — retrying once", ticker, e)
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                    "response_json_schema": PositionAssessmentSchema.model_json_schema(),
                },
            )
            result_dict = json.loads(response.text)

        validated = PositionAssessmentSchema.model_validate(result_dict)
        return validated.model_dump()

    except Exception as e:
        logger.error("Error generating assessment for %s: %s", ticker, e, exc_info=True)
        return None
