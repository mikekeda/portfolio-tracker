"""
Shared 13F computation helpers: TypedDicts, constants, pure QoQ change
functions, and the DB query helper used across multiple views.
"""
from collections import defaultdict
from datetime import date
from typing import Optional, TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Form13FFiling, Form13FHolding, Form13FManager

# Thresholds (USD for value, % for change)
FORM13F_MIN_VALUE_NEW = 50_000       # Ignore "New" positions below this (pilot/noise)
FORM13F_INCREASE_EFFECTIVE_NEW = 1000        # +1000%+ = effectively new position
FORM13F_TRIM_EFFECTIVE_LIQUIDATION = -90     # -90%+ = effectively liquidated


class Form13FHolder(TypedDict):
    """13F holder with change and optional report data."""

    manager_id: int | None
    name: str
    change: str
    report_date: str | None
    shares: int | None
    shares_prev: int | None
    value: int | None
    scored: bool
    score_reason: str | None


class Form13FInstrumentResult(TypedDict):
    """13F score and holders for one instrument."""

    score: float
    holders: list[Form13FHolder]


class Form13FFilingRow(TypedDict):
    """Aggregated row per (manager, filing) for a single instrument in get_instrument."""

    manager_name: str
    manager_id: int
    manager_cik: str
    report_date: date
    accession_number: str
    value: int
    shares: int
    filing_total_value: int


def _safe_pct(value: int, value_prev: int | None, shares: int, shares_prev: int) -> float:
    """
    Compute QoQ % change using value-based calculation where possible.

    Value-based is preferred (split-adjusted), but DB unit inconsistencies can occur
    (e.g. one quarter stored in thousands, another in dollars).  When the share-based
    direction contradicts value-based by a large margin we fall back to share-based.
    """
    pct_shares = (shares - shares_prev) / shares_prev * 100
    if not (value and value_prev):
        return pct_shares
    pct_value = (value - value_prev) / value_prev * 100
    # Sanity-check: if both agree on direction, trust value-based (split-adjusted).
    # If they disagree on sign (unit mismatch in DB), trust share-based.
    if (pct_value >= 0) == (pct_shares >= 0):
        return pct_value
    return pct_shares


def _compute_form13f_change(
    shares: int,
    shares_prev: int | None,
    value: int = 0,
    value_prev: int | None = None,
) -> str:
    """
    Compute the quarter-over-quarter change label for a 13F position.

    Uses value-based % when both values are available (split-adjusted).
    Falls back to share-based % when value data is missing.

    Returns: "—" (no prior data), "New", "Closed", or "+X.X%".
    """
    if shares_prev is None:
        return "—"
    if shares == 0:
        return "Closed"
    if shares_prev == 0:
        return "New"
    pct = _safe_pct(value, value_prev, shares, shares_prev)
    return f"{pct:+.1f}%"


def _compute_form13f_signal_score(
    shares: int,
    shares_prev: int | None,
    value: int = 0,
    value_prev: int | None = None,
) -> int:
    """
    Compute per-holder 13F signal score (-2 to +2).

    Uses value-based % change when both values are available and directionally consistent
    (split-adjusted). Falls back to share-based % on DB unit inconsistency.

    Score rules (contiguous, no gaps):
      +2: New (value >= MIN); or increase ≥1000% with value >= MIN
      +1: Increase 10% to 999%; or increase ≥1000% with value < MIN
       0: Stable (-30% to +10%); no prior data; or tiny New (value < MIN)
      -1: Trimmed (-90% to -30%)
      -2: Closed; or effective liquidation (≤-90%)
    """
    value = value or 0
    if shares_prev is None:
        return 0
    if shares == 0:
        return -2
    if shares_prev == 0:
        return 2 if value >= FORM13F_MIN_VALUE_NEW else 0
    pct = _safe_pct(value, value_prev, shares, shares_prev)
    if pct >= FORM13F_INCREASE_EFFECTIVE_NEW:
        return 2 if value >= FORM13F_MIN_VALUE_NEW else 1
    if pct <= FORM13F_TRIM_EFFECTIVE_LIQUIDATION:
        return -2
    if pct >= 10:
        return 1
    if pct < -30:
        return -1
    return 0


def _score_reason(score: int, change: str, value: int) -> str | None:
    """Human-readable explanation for why a holder has score=0 (shown in tooltip)."""
    if score != 0:
        return None  # contributing to score — no explanation needed
    if change == "—":
        return "no prior quarter for comparison"
    if change == "New" and value < FORM13F_MIN_VALUE_NEW:
        return f"new position but value below ${FORM13F_MIN_VALUE_NEW:,} minimum"
    if change not in ("New", "Closed"):
        return "change within stable range (−30% to +10%)"
    return None


def _build_sec_13f_url(cik: Optional[str], accession: Optional[str]) -> Optional[str]:
    """Build SEC EDGAR URL for a 13F filing. Returns None if cik or accession missing."""
    if not cik or not accession:
        return None
    cik_num = str(cik).lstrip("0") or "0"
    accession_clean = str(accession).replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession_clean}/"


async def _get_form13f_for_instruments(
    session: AsyncSession, instrument_ids: list[int]
) -> dict[int, Form13FInstrumentResult]:
    """
    Get 13F score and holders for each instrument.

    Score aggregation:
    1. Per-holder score: -2 to +2 from raw share counts (see _compute_form13f_signal_score)
    2. Exclude score=0 holders (no prior data or stable — no directional signal)
    3. Conviction-weighted average: weight = value / filing_total_value (treats funds equally by commitment %)
    4. Clamp to [-2, 2], round to 1 decimal

    Returns {instrument_id: {score, holders}}.
    """
    if not instrument_ids:
        return {}

    rows = (
        await session.execute(
            select(Form13FHolding, Form13FFiling, Form13FManager)
            .join(Form13FFiling, Form13FHolding.filing_id == Form13FFiling.id)
            .join(Form13FManager, Form13FFiling.manager_id == Form13FManager.id)
            .where(Form13FHolding.instrument_id.in_(instrument_ids))
        )
    ).all()

    # Also fetch the two most recent filing dates per manager so we can detect
    # genuinely new positions (manager had a prior filing but no prior holding).
    all_manager_ids = {manager.id for _, _, manager in rows}
    manager_filing_dates: dict[int, list[date]] = defaultdict(list)
    if all_manager_ids:
        filing_rows = (
            await session.execute(
                select(Form13FFiling.manager_id, Form13FFiling.report_date)
                .where(Form13FFiling.manager_id.in_(all_manager_ids))
                .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
            )
        ).all()
        for mid, rdate in filing_rows:
            manager_filing_dates[mid].append(rdate)

    by_manager_filing: dict[tuple[int, int, int], dict[str, str | int | date | None]] = {}
    for holding, filing, manager in rows:
        key = (holding.instrument_id, manager.id, filing.id)
        if key not in by_manager_filing:
            by_manager_filing[key] = {
                "instrument_id": holding.instrument_id,
                "manager_name": manager.name,
                "manager_id": manager.id,
                "report_date": filing.report_date,
                "filing_total_value": filing.total_value,
                "value": 0,
                "shares": 0,
            }
        by_manager_filing[key]["value"] += holding.value
        by_manager_filing[key]["shares"] += holding.shares

    by_manager: dict[tuple[int, int], list[dict[str, str | int | date | None]]] = defaultdict(list)
    for (iid, mid, fid), data in by_manager_filing.items():
        by_manager[(iid, mid)].append(data)

    by_instrument: dict[int, list[dict[str, str | int | date | None]]] = defaultdict(list)
    for (iid, mid), filings_list in by_manager.items():
        filings_list.sort(key=lambda x: x["report_date"], reverse=True)
        latest = filings_list[0]
        prev = filings_list[1] if len(filings_list) > 1 else None

        if prev is not None:
            # Manager held this instrument in both quarters
            shares_prev: int | None = prev["shares"]
            value_prev: int | None = prev["value"]
        elif len(manager_filing_dates.get(mid, [])) >= 2:
            # Manager has a prior filing but didn't hold this instrument → new position
            shares_prev = 0
            value_prev = 0
        else:
            # Manager has only one filing total → no comparison possible
            shares_prev = None
            value_prev = None

        shares = latest["shares"]
        change = _compute_form13f_change(
            shares, shares_prev, value=latest["value"], value_prev=value_prev
        )
        score = _compute_form13f_signal_score(
            shares, shares_prev, value=latest["value"], value_prev=value_prev
        )
        filing_total = latest["filing_total_value"] or 0
        conviction = latest["value"] / filing_total if filing_total > 0 else 0.0
        by_instrument[iid].append({
            "manager_id": mid,
            "manager_name": latest["manager_name"],
            "change": change,
            "score": score,
            "value": latest["value"],
            "conviction": conviction,
            "report_date": latest["report_date"].isoformat() if latest.get("report_date") else None,
            "shares": latest["shares"],
            "shares_prev": shares_prev,
        })

    result: dict[int, Form13FInstrumentResult] = {}
    for iid, holders in by_instrument.items():
        # Only holders with a directional signal contribute to the score.
        # score == 0 means "no prior data" or "stable" — no useful information.
        scoring_holders = [h for h in holders if h["score"] != 0]

        if scoring_holders:
            # Weight by conviction (pct of manager's portfolio), not absolute dollars.
            # This treats a small fund putting 10% into a stock equally to a large fund doing the same.
            total_conviction = sum(h["conviction"] for h in scoring_holders)
            if total_conviction > 0:
                weighted_score = sum(h["score"] * h["conviction"] for h in scoring_holders) / total_conviction
            else:
                weighted_score = sum(h["score"] for h in scoring_holders) / len(scoring_holders)
        else:
            weighted_score = 0.0

        # Round only here (backend). Frontend displays as received.
        scoring_set = set(id(h) for h in scoring_holders)
        result[iid] = {
            "score": round(max(-2.0, min(2.0, weighted_score)), 1),
            "holders": [
                {
                    "manager_id": h.get("manager_id"),
                    "name": h["manager_name"],
                    "change": h["change"],
                    "report_date": h.get("report_date"),
                    "shares": h.get("shares"),
                    "shares_prev": h.get("shares_prev"),
                    "value": h.get("value"),
                    "scored": id(h) in scoring_set,
                    "score_reason": _score_reason(h["score"], h["change"], h.get("value") or 0),
                }
                for h in holders
            ],
        }
    return result
