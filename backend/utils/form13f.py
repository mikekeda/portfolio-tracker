"""
Shared 13F computation helpers: TypedDicts, constants, pure QoQ change
functions, and the DB query helper used across multiple views.
"""

from collections import defaultdict
from datetime import date
from math import isfinite
from typing import Optional, TypedDict
from types import SimpleNamespace

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Form13FFiling, Form13FHolding, Form13FManager, Instrument, InstrumentYahoo

from backend.utils.form13f_actions import HON_CUSIPS, quantity_split_history

# Thresholds (USD for value, % for change)
FORM13F_MIN_AUM_PCT_NEW = 0.1  # 0.1% of AUM minimum for new position signal
FORM13F_MIN_AUM_PCT_INCREASE = 0.05  # 0.05% of AUM minimum for increased position signal
FORM13F_INCREASE_EFFECTIVE_NEW = 1000  # +1000%+ = effectively new position
FORM13F_TRIM_EFFECTIVE_LIQUIDATION = -90  # -90%+ = effectively liquidated


def current_manager_ids(latest_by_manager: dict[int, date]) -> tuple[set[int], Optional[date]]:
    """
    Manager ids filed up to the consensus quarter, and that quarter (the newest any has filed).

    A manager that stopped filing still has two filings on record, so a cross-manager view that
    just takes each manager's two most recent compares its stale delta against everyone else's
    current one. Callers aggregating across managers must drop the ids this omits.
    """
    if not latest_by_manager:
        return set(), None
    quarter = max(latest_by_manager.values())
    return {mid for mid, report_date in latest_by_manager.items() if report_date >= quarter}, quarter


class Form13FHolder(TypedDict):
    """13F holder with change and optional report data."""

    manager_id: int | None
    name: str
    manager_name: str
    report_date_prev: str | None
    pct_of_portfolio: float | None
    sec_filing_url: str | None
    change: str
    report_date: str | None
    shares: int | None
    shares_prev: int | None
    shares_prev_adjusted: float | None
    value: int | None
    value_prev: int | None
    scored: bool
    score_reason: str | None


class Form13FInstrumentResult(TypedDict):
    """13F score and holders for one instrument."""

    score: Optional[float]
    holders: list[Form13FHolder]



def split_adjusted_shares(
    shares: float | None, splits: dict | None, previous_date: date | None, report_date: date
) -> float | None:
    """Restate shares using quantity_split_history's verified/common share factors."""
    if shares is None or shares == 0 or previous_date is None:
        return shares
    adjusted = shares
    for stamp, factor in (splits or {}).items():
        try:
            split_date = date.fromisoformat(str(stamp)[:10])
        except ValueError:
            continue
        if previous_date < split_date <= report_date:
            if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not isfinite(factor) or factor <= 0:
                return None  # Corporate action needs verification; withhold the comparison.
            adjusted *= factor
    return adjusted


async def load_split_history(session: AsyncSession, instrument_ids) -> dict[int, dict]:
    """One bulk read; never fetch Yahoo while serving a 13F view."""
    if not instrument_ids:
        return {}
    rows = (
        await session.execute(
            select(Instrument.id, InstrumentYahoo.splits, Instrument.yahoo_symbol)
            .outerjoin(InstrumentYahoo, Instrument.id == InstrumentYahoo.instrument_id).where(
                Instrument.id.in_(instrument_ids)
            )
        )
    ).all()
    return {iid: quantity_split_history(splits, symbol) for iid, splits, symbol in rows}


def resolved_holding_instrument_id():
    """Resolve the verified HON CUSIP transition without rewriting source filings."""
    hon_id = select(func.min(Instrument.id)).where(Instrument.yahoo_symbol == "HON").scalar_subquery()
    return func.coalesce(
        Form13FHolding.instrument_id,
        case((func.upper(Form13FHolding.cusip).in_(HON_CUSIPS), hon_id)),
    )


async def load_form13f_holdings(session: AsyncSession, filing_ids) -> list:
    rows = (await session.execute(
        select(Form13FHolding, resolved_holding_instrument_id().label("resolved_id"))
        .where(Form13FHolding.filing_id.in_(filing_ids))
    )).all()
    # Read-only projections: never dirty ORM entities while resolving aliases.
    return [SimpleNamespace(
        filing_id=h.filing_id, cusip=h.cusip, issuer=h.issuer,
        shares=h.shares, value=h.value, instrument_id=iid,
    ) for h, iid in rows]


def _safe_pct(shares: int, shares_prev: float) -> float:
    """Quantity change only; prior shares must use the current share basis."""
    return (shares - shares_prev) / shares_prev * 100


def _compute_form13f_change(
    shares: int,
    shares_prev: float | None,
) -> str:
    """
    Compute the quarter-over-quarter change label for a 13F position.

    Uses share-count change, with previous shares already split-adjusted by the caller.

    Returns: "—" (no prior data), "New", "Closed", or "+X.X%".
    """
    if shares_prev is None:
        return "—"
    if shares == 0:
        return "Closed"
    if shares_prev == 0:
        return "New"
    pct = _safe_pct(shares, shares_prev)
    return f"{pct:+.1f}%"


def _compute_form13f_signal_score(
    shares: int,
    shares_prev: float | None,
    value: int = 0,
    value_prev: int | None = None,
    filing_total_value: int | None = None,
) -> int:
    """
    Compute per-holder 13F signal score (-2 to +2).

    Uses split-adjusted share-count change; value only sets the conviction floor.

    Score rules (contiguous, no gaps):
      +2: New (pct AUM >= 0.1%); or increase ≥1000% with pct AUM >= 0.1%
      +1: Increase 10% to 999% and pct AUM >= 0.05%; or increase ≥1000% with pct AUM < 0.1% but >= 0.05%
       0: Stable (-30% to +10%); no prior data; or tiny New/Increase (pct AUM < floor)
      -1: Trimmed (-90% to -30%)
      -2: Closed; or effective liquidation (≤-90%)
    """
    value = value or 0
    pct_aum = (value / filing_total_value * 100) if (filing_total_value and filing_total_value > 0) else 0.0

    if shares_prev is None:
        return 0
    if shares == 0:
        return -2
    if shares_prev == 0:
        return 2 if pct_aum >= FORM13F_MIN_AUM_PCT_NEW else 0
    pct = _safe_pct(shares, shares_prev)
    if pct >= FORM13F_INCREASE_EFFECTIVE_NEW:
        return 2 if pct_aum >= FORM13F_MIN_AUM_PCT_NEW else (1 if pct_aum >= FORM13F_MIN_AUM_PCT_INCREASE else 0)
    if pct <= FORM13F_TRIM_EFFECTIVE_LIQUIDATION:
        return -2
    if pct >= 10:
        return 1 if pct_aum >= FORM13F_MIN_AUM_PCT_INCREASE else 0
    if pct < -30:
        return -1
    return 0


def aggregate_signal_score(scoring_pairs: list[tuple[float, float]]) -> Optional[float]:
    """Conviction-weighted mean of directional holder scores, clamped to [-2, 2].

    `scoring_pairs` is (score, conviction) for holders with a non-zero score, where
    conviction is pct of the manager's portfolio rather than absolute dollars — a
    small fund at 10% weighs the same as a large one at 10%.

    None when no holder cleared a directional threshold: that is the absence of a
    signal, not a neutral one, so consumers reweight rather than scoring it 0.
    """
    if not scoring_pairs:
        return None

    total_conviction = sum(c for _, c in scoring_pairs)
    if total_conviction > 0:
        weighted = sum(s * c for s, c in scoring_pairs) / total_conviction
    else:
        weighted = sum(s for s, _ in scoring_pairs) / len(scoring_pairs)

    return round(max(-2.0, min(2.0, weighted)), 1)


def _score_reason(
    score: int, change: str, value: int, filing_total_value: int | None = None
) -> str | None:
    """Human-readable explanation for why a holder does not contribute to the score (tooltip)."""
    if score != 0:
        return None  # contributing to score — no explanation needed
    if change == "—":
        return "no comparable prior shares (missing history or unverified corporate action)"

    pct_aum = (value / filing_total_value * 100) if (filing_total_value and filing_total_value > 0) else 0.0
    if change == "New" and pct_aum < FORM13F_MIN_AUM_PCT_NEW:
        return f"new position but conviction ({pct_aum:.2f}%) below {FORM13F_MIN_AUM_PCT_NEW}% threshold"
    if change not in ("New", "Closed"):
        if pct_aum < FORM13F_MIN_AUM_PCT_INCREASE and "+" in change:
            return f"increased position but conviction ({pct_aum:.2f}%) below {FORM13F_MIN_AUM_PCT_INCREASE}% threshold"
        return "change within stable range (−30% to +10%)"
    return None


def _build_sec_13f_url(cik: Optional[str], accession: Optional[str]) -> Optional[str]:
    """Build SEC EDGAR URL for a 13F filing. Returns None if cik or accession missing."""
    if not cik or not accession:
        return None
    cik_num = str(cik).lstrip("0") or "0"
    accession_clean = str(accession).replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession_clean}/"


def estimated_holder_flow(holder: dict) -> float | None:
    """Share-flow proxy: exits use prior value; not actual trading proceeds."""
    shares, value = holder.get("shares"), holder.get("value")
    prior_shares, prior_value = holder.get("shares_prev"), holder.get("value_prev")
    if shares == 0 and prior_shares is not None and prior_shares > 0:
        return -prior_value if prior_value is not None and isfinite(prior_value) and prior_value >= 0 else None
    adjusted = holder.get("shares_prev_adjusted", prior_shares)
    if any(v is None or not isfinite(v) for v in (shares, value, adjusted)) or shares <= 0:
        return None
    return (shares - adjusted) * value / shares


async def _get_form13f_for_instruments(
    session: AsyncSession, instrument_ids: list[int]
) -> dict[int, Form13FInstrumentResult]:
    """Compare each manager's latest two filings, including absent/closed positions.

    Holdings and Stock consume these same rows and score. Never select periods
    from the subset of filings that happens to contain the requested security.
    Source holdings remain untouched, including resolved corporate-action IDs.
    """
    if not instrument_ids:
        return {}
    splits_by_id = await load_split_history(session, instrument_ids)
    filing_rows = (await session.execute(
        select(Form13FFiling, Form13FManager)
        .join(Form13FManager, Form13FFiling.manager_id == Form13FManager.id)
        .order_by(Form13FFiling.report_date.desc(), Form13FFiling.id.desc())
    )).all()
    by_manager = defaultdict(list)
    managers = {}
    for filing, manager in filing_rows:
        managers[manager.id] = manager
        if len(by_manager[manager.id]) < 2:
            by_manager[manager.id].append(filing)
    current_ids, _ = current_manager_ids({mid: filings[0].report_date for mid, filings in by_manager.items()})
    filing_ids = [f.id for mid in current_ids for f in by_manager[mid]]
    if not filing_ids:
        return {}
    rows = (await session.execute(
        select(Form13FHolding, resolved_holding_instrument_id())
        .where(Form13FHolding.filing_id.in_(filing_ids), resolved_holding_instrument_id().in_(instrument_ids))
    )).all()
    holdings = defaultdict(lambda: {"shares": 0, "value": 0})
    instruments_by_manager = defaultdict(set)
    manager_by_filing = {f.id: mid for mid in current_ids for f in by_manager[mid]}
    for holding, iid in rows:
        item = holdings[(holding.filing_id, iid)]
        item["shares"] += holding.shares
        item["value"] += holding.value
        instruments_by_manager[manager_by_filing[holding.filing_id]].add(iid)

    result = {}
    scoring_pairs = defaultdict(list)
    for mid in sorted(current_ids):
        manager = managers[mid]
        filings = by_manager[mid]
        latest = filings[0]
        previous = filings[1] if len(filings) > 1 else None
        for iid in sorted(instruments_by_manager[mid]):
            current = holdings.get((latest.id, iid), {"shares": 0, "value": 0})
            prior = holdings.get((previous.id, iid), {"shares": 0, "value": 0}) if previous else None
            shares_prev = prior["shares"] if prior else None
            value_prev = prior["value"] if prior else None
            adjusted = split_adjusted_shares(
                shares_prev, splits_by_id.get(iid), previous.report_date if previous else None, latest.report_date
            )
            shares, value = current["shares"], current["value"]
            closed = shares == 0 and (shares_prev or 0) > 0
            # An unverified split prevents quantity comparisons, but cannot turn
            # absence in the latest filing into continued ownership.
            change = "Closed" if closed else _compute_form13f_change(shares, adjusted)
            score = -2 if closed else _compute_form13f_signal_score(
                shares, adjusted, value=value, value_prev=value_prev, filing_total_value=latest.total_value
            )
            # Exits have zero current value: weight their signal by prior conviction.
            conviction_value = value_prev if closed else value
            conviction_total = previous.total_value if closed else latest.total_value
            conviction = conviction_value / conviction_total if conviction_total > 0 else 0.0
            if score:
                scoring_pairs[iid].append((score, conviction))
            pct = value / latest.total_value * 100 if latest.total_value > 0 else None
            holder = {
                "manager_id": mid,
                "name": manager.name,
                "manager_name": manager.name,
                "change": change,
                "report_date": latest.report_date.isoformat(),
                "report_date_prev": previous.report_date.isoformat() if previous else None,
                "shares": shares,
                "shares_prev": shares_prev,
                "shares_prev_adjusted": adjusted,
                "value": value,
                "value_prev": value_prev,
                "pct_of_portfolio": round(pct, 2) if pct is not None else None,
                "scored": score != 0,
                "score_reason": _score_reason(score, change, value, latest.total_value),
                "sec_filing_url": _build_sec_13f_url(manager.cik, latest.accession_number),
            }
            result.setdefault(iid, {"score": None, "holders": []})["holders"].append(holder)
    for iid, item in result.items():
        item["score"] = aggregate_signal_score(scoring_pairs[iid])
    return result
