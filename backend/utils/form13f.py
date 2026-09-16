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
    score: int, change: str, value: int, filing_total_value: int | None = None, stale: bool = False
) -> str | None:
    """Human-readable explanation for why a holder does not contribute to the score (tooltip)."""
    if stale:
        return "has not filed for the latest quarter — change is from an older filing"
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


async def _get_form13f_for_instruments(
    session: AsyncSession, instrument_ids: list[int]
) -> dict[int, Form13FInstrumentResult]:
    """
    Get 13F score and holders for each instrument.

    Score aggregation:
    1. Per-holder score: -2 to +2 from split-adjusted share counts (see _compute_form13f_signal_score)
    2. Exclude score=0 holders (no prior data or stable — no directional signal)
    3. Conviction-weighted average: weight = value / filing_total_value (treats funds equally by commitment %)
    4. Clamp to [-2, 2], round to 1 decimal

    Returns {instrument_id: {score, holders}}.
    """
    if not instrument_ids:
        return {}

    splits_by_id = await load_split_history(session, instrument_ids)
    rows = (
        await session.execute(
            select(Form13FHolding, Form13FFiling, Form13FManager, resolved_holding_instrument_id())
            .join(Form13FFiling, Form13FHolding.filing_id == Form13FFiling.id)
            .join(Form13FManager, Form13FFiling.manager_id == Form13FManager.id)
            .where(resolved_holding_instrument_id().in_(instrument_ids))
        )
    ).all()

    # Every manager, not just holders of these instruments: the consensus quarter has to be the
    # newest across all managers, or a wholly stale subset would look current.
    manager_filing_dates: dict[int, list[date]] = defaultdict(list)
    filing_rows = (
        await session.execute(
            select(Form13FFiling.manager_id, Form13FFiling.report_date).order_by(
                Form13FFiling.manager_id, Form13FFiling.report_date.desc()
            )
        )
    ).all()
    for mid, rdate in filing_rows:
        manager_filing_dates[mid].append(rdate)

    current_ids, _ = current_manager_ids({mid: dates[0] for mid, dates in manager_filing_dates.items()})

    by_manager_filing: dict[tuple[int, int, int], dict[str, str | int | date | None]] = {}
    for holding, filing, manager, iid in rows:
        key = (iid, manager.id, filing.id)
        if key not in by_manager_filing:
            by_manager_filing[key] = {
                "instrument_id": iid,
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

        shares_prev_reported = shares_prev
        shares_prev = split_adjusted_shares(
            shares_prev, splits_by_id.get(iid), prev["report_date"] if prev else None, latest["report_date"]
        )
        shares = latest["shares"]
        filing_total = latest["filing_total_value"] or 0
        change = _compute_form13f_change(shares, shares_prev)
        score = _compute_form13f_signal_score(shares, shares_prev, value=latest["value"], value_prev=value_prev, filing_total_value=filing_total)
        conviction = latest["value"] / filing_total if filing_total > 0 else 0.0
        by_instrument[iid].append(
            {
                "manager_id": mid,
                "stale": mid not in current_ids,
                "manager_name": latest["manager_name"],
                "change": change,
                "score": score,
                "value": latest["value"],
                "value_prev": value_prev,
                "conviction": conviction,
                "filing_total_value": filing_total,
                "report_date": latest["report_date"].isoformat() if latest.get("report_date") else None,
                "shares": latest["shares"],
                "shares_prev": shares_prev_reported,
                "shares_prev_adjusted": shares_prev,
            }
        )

    result: dict[int, Form13FInstrumentResult] = {}
    for iid, holders in by_instrument.items():
        # Only holders with a directional signal contribute. score == 0 is mostly
        # a position below the AUM floors, sometimes stable or a first filing.
        scoring_holders = [h for h in holders if h["score"] != 0 and not h["stale"]]
        scoring_set = set(id(h) for h in scoring_holders)
        result[iid] = {
            "score": aggregate_signal_score([(h["score"], h["conviction"]) for h in scoring_holders]),
            "holders": [
                {
                    "manager_id": h.get("manager_id"),
                    "name": h["manager_name"],
                    "change": h["change"],
                    "report_date": h.get("report_date"),
                    "shares": h.get("shares"),
                    "shares_prev": h.get("shares_prev"),
                    "shares_prev_adjusted": h.get("shares_prev_adjusted"),
                    "value": h.get("value"),
                    "value_prev": h.get("value_prev"),
                    "scored": id(h) in scoring_set,
                    "score_reason": _score_reason(
                        h["score"],
                        h["change"],
                        h.get("value") or 0,
                        h.get("filing_total_value"),
                        stale=h["stale"],
                    ),
                }
                for h in holders
            ],
        }
    return result
