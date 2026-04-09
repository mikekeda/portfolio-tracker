from collections import defaultdict
from datetime import date as date_type
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app import get_db_session
from backend.utils.form13f import (
    _compute_form13f_change,
    _compute_form13f_signal_score,
    _safe_pct,
)
from models import Form13FFiling, Form13FHolding, Form13FManager, HoldingDaily, Instrument

router = APIRouter()

# Max activity names shown per category in manager card tooltips
_MAX_ACTIVITY_NAMES = 5


def _top_activity(items: list[dict]) -> list[dict]:
    """Sort by _sort key descending, take top N, strip the internal _sort key."""
    items.sort(key=lambda x: x["_sort"], reverse=True)
    return [{k: v for k, v in item.items() if k != "_sort"} for item in items[:_MAX_ACTIVITY_NAMES]]


def _aggregate_holdings_by_cusip(holdings: list) -> dict[str, dict]:
    """Aggregate Form13FHolding rows by CUSIP (sums value+shares across share classes)."""
    by_cusip: dict[str, dict] = {}
    for h in holdings:
        if h.cusip not in by_cusip:
            by_cusip[h.cusip] = {
                "cusip": h.cusip,
                "issuer": h.issuer,
                "value": 0,
                "shares": 0,
                "instrument_id": h.instrument_id,
            }
        else:
            # Prefer the first non-None instrument_id we find for this CUSIP
            if by_cusip[h.cusip]["instrument_id"] is None and h.instrument_id is not None:
                by_cusip[h.cusip]["instrument_id"] = h.instrument_id
        by_cusip[h.cusip]["value"] += h.value
        by_cusip[h.cusip]["shares"] += h.shares
    return by_cusip


def _change_sort_value(
    shares: int,
    shares_prev: Optional[int],
    value: int = 0,
    value_prev: Optional[int] = None,
) -> float:
    """Numeric sort key for QoQ change: New=+1e9, Closed=-1e9, else value-based %."""
    if shares_prev is None:
        return 0.0
    if shares == 0:
        return -1.0e9
    if shares_prev == 0:
        return 1.0e9
    return _safe_pct(value, value_prev, shares, shares_prev)


def _build_portfolio_and_moves(
    latest_by_cusip: dict,
    prev_by_cusip: dict,
    total_value: int,
    instruments: dict[int, Any],
) -> tuple[list[dict], dict]:
    """
    Compute per-position portfolio rows and the moves summary (new/closed/buys/sells).
    `instruments` is a dict of {instrument_id: Instrument}.
    Returns (portfolio_rows sorted by value desc, moves dict).
    """
    portfolio: list[dict] = []
    for cusip, item in latest_by_cusip.items():
        prev = prev_by_cusip.get(cusip)
        if prev is not None:
            shares_prev: Optional[int] = prev["shares"]
            value_prev: Optional[int] = prev["value"]
        elif prev_by_cusip:
            # Previous filing exists but this CUSIP wasn't in it → new position
            shares_prev = 0
            value_prev = 0
        else:
            shares_prev = None
            value_prev = None

        value_change = (item["value"] - value_prev) if value_prev is not None else None
        change = _compute_form13f_change(item["shares"], shares_prev, value=item["value"], value_prev=value_prev)

        instr = instruments.get(item["instrument_id"])
        pct = item["value"] / total_value * 100 if total_value else 0.0

        portfolio.append(
            {
                "cusip": cusip,
                "issuer": item["issuer"],
                "yahoo_symbol": instr.yahoo_symbol if instr else None,
                "name": instr.name if instr else item["issuer"],
                "value": item["value"],
                "shares": item["shares"],
                "pct_of_portfolio": round(pct, 2),
                "change": change,
                "change_sort": _change_sort_value(
                    item["shares"], shares_prev, value=item["value"], value_prev=value_prev
                ),
                "value_change": value_change,
                "value_prev": value_prev,
                "shares_prev": shares_prev,
            }
        )

    portfolio.sort(key=lambda x: x["value"], reverse=True)
    for i, p in enumerate(portfolio):
        p["rank"] = i + 1

    # Closed positions: in prev but missing from latest
    closed: list[dict] = []
    for cusip, prev in prev_by_cusip.items():
        if cusip not in latest_by_cusip:
            instr = instruments.get(prev["instrument_id"])
            closed.append(
                {
                    "cusip": cusip,
                    "issuer": prev["issuer"],
                    "yahoo_symbol": instr.yahoo_symbol if instr else None,
                    "name": instr.name if instr else prev["issuer"],
                    "value": 0,
                    "value_prev": prev["value"],
                    "shares": 0,
                    "shares_prev": prev["shares"],
                    "change": "Closed",
                    "change_sort": -1.0e9,
                    "value_change": -prev["value"],
                }
            )
    closed.sort(key=lambda x: (x["value_prev"] or 0), reverse=True)

    new_positions = sorted(
        [p for p in portfolio if p["change"] == "New"],
        key=lambda x: x["value"],
        reverse=True,
    )
    existing = [p for p in portfolio if p["change"] not in ("New", "Closed", "—")]
    all_buys = sorted(
        [p for p in existing if (p["value_change"] or 0) > 0],
        key=lambda x: x["value_change"],
        reverse=True,
    )
    all_sells = sorted(
        [p for p in existing if (p["value_change"] or 0) < 0],
        key=lambda x: x["value_change"],
    )

    moves = {
        "new_positions": new_positions,
        "closed_positions": closed,
        "top_buys": all_buys[:10],
        "top_sells": all_sells[:10],
        # Actual counts for stats bar (not capped at 10 like the display lists)
        "increased_count": len(all_buys),
        "trimmed_count": len(all_sells),
    }
    return portfolio, moves


@router.get("/api/13f/highlights")
async def get_form13f_highlights(session: AsyncSession = Depends(get_db_session)) -> dict:
    """
    Cross-manager consensus signals from the latest 13F filings.
    Returns the stocks being bought / sold by the most managers, and the most widely held.
    """
    managers_result = await session.execute(select(Form13FManager).order_by(Form13FManager.name))
    managers = managers_result.scalars().all()
    if not managers:
        return {"most_bought": [], "most_sold": [], "most_held": []}

    manager_ids = [m.id for m in managers]
    filings_result = await session.execute(
        select(Form13FFiling)
        .where(Form13FFiling.manager_id.in_(manager_ids))
        .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
    )
    all_filings = filings_result.scalars().all()

    filings_by_manager: dict[int, list] = defaultdict(list)
    for f in all_filings:
        filings_by_manager[f.manager_id].append(f)

    latest_ids: dict[int, int] = {}
    prev_ids: dict[int, int] = {}
    for mid, mf in filings_by_manager.items():
        latest_ids[mid] = mf[0].id
        if len(mf) >= 2:
            prev_ids[mid] = mf[1].id

    relevant_ids = list(set(latest_ids.values()) | set(prev_ids.values()))
    holdings_result = await session.execute(select(Form13FHolding).where(Form13FHolding.filing_id.in_(relevant_ids)))
    all_holdings = holdings_result.scalars().all()

    instrument_ids = {h.instrument_id for h in all_holdings if h.instrument_id is not None}
    instruments: dict[int, Any] = {}
    if instrument_ids:
        instr_result = await session.execute(select(Instrument).where(Instrument.id.in_(instrument_ids)))
        for instr in instr_result.scalars().all():
            instruments[instr.id] = instr

    holdings_by_filing: dict[int, list] = defaultdict(list)
    for h in all_holdings:
        holdings_by_filing[h.filing_id].append(h)

    manager_by_id = {m.id: m for m in managers}

    buying: dict[str, dict] = {}
    selling: dict[str, dict] = {}
    held: dict[str, dict] = {}

    for mid, mf in filings_by_manager.items():
        manager_name = manager_by_id[mid].name
        latest_filing_total = mf[0].total_value or 0
        latest_by_cusip = _aggregate_holdings_by_cusip(holdings_by_filing.get(latest_ids[mid], []))
        prev_by_cusip = (
            _aggregate_holdings_by_cusip(holdings_by_filing.get(prev_ids[mid], [])) if mid in prev_ids else {}
        )

        # Most held (count every manager that holds this stock currently)
        for cusip, item in latest_by_cusip.items():
            instr = instruments.get(item["instrument_id"])
            if cusip not in held:
                held[cusip] = {
                    "cusip": cusip,
                    "issuer": item["issuer"],
                    "yahoo_symbol": instr.yahoo_symbol if instr else None,
                    "name": instr.name if instr else item["issuer"],
                    "count": 0,
                    "managers": [],
                    "total_value": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    # Always present so the API schema is consistent even when there's no activity
                    "buy_managers": [],
                    "sell_managers": [],
                }
            held[cusip]["count"] += 1
            held[cusip]["managers"].append({"name": manager_name, "current_value": item["value"]})
            held[cusip]["total_value"] += item["value"]

        if not prev_by_cusip:
            continue  # No prev quarter → can't compute moves for this manager

        # Buying signals: new or increased positions
        for cusip, item in latest_by_cusip.items():
            prev = prev_by_cusip.get(cusip)
            # prev_by_cusip is non-empty here (guarded above), so None → new position
            shares_prev = prev["shares"] if prev is not None else 0
            value_prev_buy = prev["value"] if prev is not None else 0
            score = _compute_form13f_signal_score(
                item["shares"], shares_prev, item["value"], value_prev=value_prev_buy, filing_total_value=latest_filing_total
            )
            if score <= 0:
                continue
            # Skip positions with no reported dollar value (options, warrants, or missing data).
            # These produce misleading 0-value entries since price cannot be inferred.
            if item["value"] == 0:
                continue
            change = _compute_form13f_change(
                item["shares"], shares_prev, value=item["value"], value_prev=value_prev_buy
            )
            # Capital deployed = shares added × current quarter-end price.
            # This is independent of stock price movement and shows actual money spent.
            price_now = item["value"] / item["shares"] if item["shares"] else 0
            shares_added = item["shares"] - shares_prev
            transaction_value = round(shares_added * price_now)  # always positive for buys
            pct_impact = (transaction_value / latest_filing_total * 100) if latest_filing_total else 0.0

            instr = instruments.get(item["instrument_id"])
            if cusip not in buying:
                buying[cusip] = {
                    "cusip": cusip,
                    "issuer": item["issuer"],
                    "yahoo_symbol": instr.yahoo_symbol if instr else None,
                    "name": instr.name if instr else item["issuer"],
                    "count": 0,
                    "managers": [],
                    "total_value_added": 0,
                    "total_conviction_added": 0.0,
                }
            buying[cusip]["count"] += 1
            buying[cusip]["managers"].append(
                {"name": manager_name, "change": change, "value_change": transaction_value, "pct_impact": pct_impact}
            )
            buying[cusip]["total_value_added"] += transaction_value
            buying[cusip]["total_conviction_added"] += pct_impact

        # Selling signals: closed or significantly trimmed positions
        for cusip, prev_item in prev_by_cusip.items():
            latest_item = latest_by_cusip.get(cusip)
            if latest_item is None:
                # Skip closes if previous value was 0 (options / missing data)
                if prev_item["value"] == 0:
                    continue
                change = "Closed"
                instr_id = prev_item["instrument_id"]
                issuer = prev_item["issuer"]
                # Full exit: capital extracted = full previous position value
                transaction_value = -round(prev_item["value"])
            else:
                score = _compute_form13f_signal_score(
                    latest_item["shares"],
                    prev_item["shares"],
                    latest_item["value"],
                    value_prev=prev_item["value"],
                    filing_total_value=latest_filing_total,
                )
                if score >= 0:
                    continue
                change = _compute_form13f_change(
                    latest_item["shares"],
                    prev_item["shares"],
                    value=latest_item["value"],
                    value_prev=prev_item["value"],
                )
                instr_id = latest_item["instrument_id"] or prev_item["instrument_id"]
                issuer = latest_item["issuer"]
                # Capital extracted = shares sold × current quarter-end price.
                # Independent of price movement; shows actual proceeds received.
                price_now = latest_item["value"] / latest_item["shares"] if latest_item["shares"] else 0
                shares_sold = prev_item["shares"] - latest_item["shares"]
                transaction_value = -round(shares_sold * price_now)  # negative = extracted

            instr = instruments.get(instr_id)
            pct_impact = (abs(transaction_value) / latest_filing_total * 100) if latest_filing_total else 0.0
            if cusip not in selling:
                selling[cusip] = {
                    "cusip": cusip,
                    "issuer": issuer,
                    "yahoo_symbol": instr.yahoo_symbol if instr else None,
                    "name": instr.name if instr else issuer,
                    "count": 0,
                    "managers": [],
                    "total_value_removed": 0,
                    "total_conviction_removed": 0.0,
                }
            selling[cusip]["count"] += 1
            selling[cusip]["managers"].append(
                {"name": manager_name, "change": change, "value_change": transaction_value, "pct_impact": -pct_impact}
            )
            selling[cusip]["total_value_removed"] += abs(transaction_value)
            selling[cusip]["total_conviction_removed"] += pct_impact

    # Merge buy/sell counts per CUSIP and compute net signal
    # Propagate buy/sell trend counts + full manager objects (with change + value_change) into held items
    for cusip, h in held.items():
        if cusip in buying:
            h["buy_count"] += buying[cusip]["count"]
            h["buy_managers"] = buying[cusip]["managers"]  # keep full {name, change, value_change}
        if cusip in selling:
            h["sell_count"] += selling[cusip]["count"]
            h["sell_managers"] = selling[cusip]["managers"]  # keep full {name, change, value_change}

    all_cusips = set(buying) | set(selling)
    consensus_buys: list[dict] = []
    consensus_sells: list[dict] = []
    disputed: list[dict] = []

    for cusip in all_cusips:
        buy = buying.get(cusip)
        sell = selling.get(cusip)
        buy_count = buy["count"] if buy else 0
        sell_count = sell["count"] if sell else 0
        net = buy_count - sell_count

        buy_managers = buy["managers"] if buy else []
        sell_managers = sell["managers"] if sell else []
        new_count = sum(1 for m in buy_managers if m.get("change") == "New")
        closed_count = sum(1 for m in sell_managers if m.get("change") == "Closed")

        base = buy if buy else sell
        item: dict = {
            "cusip": cusip,
            "issuer": base["issuer"],
            "yahoo_symbol": base["yahoo_symbol"],
            "name": base["name"],
            "buy_count": buy_count,
            "sell_count": sell_count,
            "net_managers": net,
            "total_value_added": buy["total_value_added"] if buy else 0,
            "total_value_removed": sell["total_value_removed"] if sell else 0,
            "total_conviction_added": buy["total_conviction_added"] if buy else 0.0,
            "total_conviction_removed": sell["total_conviction_removed"] if sell else 0.0,
            "buy_managers": buy_managers,
            "sell_managers": sell_managers,
            "new_count": new_count,
            "closed_count": closed_count,
        }

        if buy_count > 0 and sell_count > 0 and net == 0:
            # Enrich with stable holders (from held dict) so the tooltip can show who's on the fence
            buy_names = {m["name"] for m in item["buy_managers"]}
            sell_names = {m["name"] for m in item["sell_managers"]}
            if cusip in held:
                item["managers"] = [
                    m for m in held[cusip]["managers"] if m["name"] not in buy_names and m["name"] not in sell_names
                ]
            else:
                item["managers"] = []
            disputed.append(item)
        elif net > 0:
            consensus_buys.append(item)
        elif net < 0:
            consensus_sells.append(item)

    most_bought = sorted(
        consensus_buys,
        key=lambda x: (x["net_managers"], x["new_count"], x["total_conviction_added"] - x["total_conviction_removed"]),
        reverse=True,
    )
    most_sold = sorted(
        consensus_sells,
        key=lambda x: (-x["net_managers"], x["closed_count"], x["total_conviction_removed"] - x["total_conviction_added"]),
        reverse=True,
    )
    most_held = sorted(held.values(), key=lambda x: (x["count"], x["total_value"]), reverse=True)[:20]
    top_disputed = sorted(
        disputed,
        key=lambda x: (x["buy_count"] + x["sell_count"]),
        reverse=True,
    )[:20]

    return {
        "most_bought": most_bought,
        "most_sold": most_sold,
        "most_held": most_held,
        "disputed": top_disputed,
    }


@router.get("/api/13f/managers")
async def get_form13f_managers_list(session: AsyncSession = Depends(get_db_session)) -> list:
    """List all 13F managers with their latest filing summary and QoQ activity counts."""
    managers_result = await session.execute(select(Form13FManager).order_by(Form13FManager.name))
    managers = managers_result.scalars().all()
    if not managers:
        return []

    manager_ids = [m.id for m in managers]

    filings_result = await session.execute(
        select(Form13FFiling)
        .where(Form13FFiling.manager_id.in_(manager_ids))
        .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
    )
    all_filings = filings_result.scalars().all()

    # Group filings per manager (already ordered desc by date)
    filings_by_manager: dict[int, list] = defaultdict(list)
    for f in all_filings:
        filings_by_manager[f.manager_id].append(f)

    # Collect the latest + prev filing IDs to load holdings in one query
    latest_ids: dict[int, int] = {}  # manager_id → latest filing_id
    prev_ids: dict[int, int] = {}
    for mid, mf in filings_by_manager.items():
        latest_ids[mid] = mf[0].id
        if len(mf) >= 2:
            prev_ids[mid] = mf[1].id

    relevant_filing_ids = list(set(latest_ids.values()) | set(prev_ids.values()))
    holdings_result = await session.execute(
        select(Form13FHolding).where(Form13FHolding.filing_id.in_(relevant_filing_ids))
    )
    all_holdings = holdings_result.scalars().all()

    # Group holdings by filing_id
    holdings_by_filing: dict[int, list] = defaultdict(list)
    for h in all_holdings:
        holdings_by_filing[h.filing_id].append(h)

    manager_by_id = {m.id: m for m in managers}
    result = []
    for mid in manager_ids:
        m = manager_by_id[mid]
        mf = filings_by_manager.get(mid, [])
        if not mf:
            continue

        latest_filing = mf[0]
        latest_cusips = _aggregate_holdings_by_cusip(holdings_by_filing.get(latest_ids[mid], []))
        prev_cusips = (
            _aggregate_holdings_by_cusip(holdings_by_filing.get(prev_ids.get(mid), [])) if mid in prev_ids else {}
        )

        # Count activity types and collect top stock names per category.
        # Sorting strategy (matches signal strength):
        #   new      → biggest current position first (size = conviction)
        #   increased → biggest current position first (highest-conviction adds)
        #   trimmed  → biggest % reduction first (strongest exit signal)
        #   closed   → biggest previous position first (impact of full exit)
        activity: dict[str, int] = {"new": 0, "closed": 0, "increased": 0, "trimmed": 0, "stable": 0}
        total_val = latest_filing.total_value or 0
        prev_total_val = (mf[1].total_value if len(mf) >= 2 else 0) or 0

        def _portfolio_pct(value: float, use_prev: bool = False) -> float | None:
            base = prev_total_val if use_prev else total_val
            if not base or not value:
                return None
            return round(value / base * 100, 1)

        # Collect all classified items first, then sort per category
        _all_new: list[dict] = []
        _all_increased: list[dict] = []
        _all_trimmed: list[dict] = []
        _all_closed: list[dict] = []

        if prev_cusips:
            for cusip, item in latest_cusips.items():
                prev = prev_cusips.get(cusip)
                shares_prev = prev["shares"] if prev is not None else 0
                value_prev_act = prev["value"] if prev is not None else 0
                change = _compute_form13f_change(
                    item["shares"], shares_prev, value=item["value"], value_prev=value_prev_act
                )
                name = item.get("issuer") or cusip
                if change == "New":
                    activity["new"] += 1
                    _all_new.append(
                        {
                            "name": name,
                            "pct": _portfolio_pct(item["value"]),
                            "_sort": item.get("value", 0),
                        }
                    )
                elif change == "—":
                    activity["stable"] += 1
                else:
                    m_obj = change.replace("+", "").replace("%", "")
                    try:
                        pct_val = float(m_obj)
                        if pct_val >= 10:
                            activity["increased"] += 1
                            _all_increased.append(
                                {
                                    "name": name,
                                    "pct": _portfolio_pct(item["value"]),
                                    "change": change,
                                    "_sort": item.get("value", 0),
                                }
                            )
                        elif pct_val <= -30:
                            activity["trimmed"] += 1
                            _all_trimmed.append(
                                {
                                    "name": name,
                                    "pct": _portfolio_pct(item["value"]),
                                    "change": change,
                                    "_sort": abs(pct_val),  # biggest % cut first
                                }
                            )
                        else:
                            activity["stable"] += 1
                    except ValueError:
                        activity["stable"] += 1

            for cusip, prev_item in prev_cusips.items():
                if cusip not in latest_cusips:
                    activity["closed"] += 1
                    _all_closed.append(
                        {
                            "name": prev_item.get("issuer") or cusip,
                            "pct": _portfolio_pct(prev_item.get("value", 0), use_prev=True),
                            "_sort": prev_item.get("value", 0),
                        }
                    )

        activity_names = {
            "new": _top_activity(_all_new),
            "increased": _top_activity(_all_increased),
            "trimmed": _top_activity(_all_trimmed),
            "closed": _top_activity(_all_closed),
        }

        result.append(
            {
                "id": m.id,
                "name": m.name,
                "cik": m.cik,
                "latest_report_date": latest_filing.report_date.isoformat(),
                "total_value": latest_filing.total_value,
                "num_positions": len(latest_cusips),
                "filing_count": len(mf),
                "activity": activity if prev_cusips else None,
                "activity_names": activity_names if prev_cusips else None,
            }
        )

    return result


@router.get("/api/13f/managers/{manager_id}")
async def get_form13f_manager_detail(
    manager_id: int,
    report_date: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Full portfolio + moves for a single manager for a given quarter (default: latest)."""
    manager_result = await session.execute(select(Form13FManager).where(Form13FManager.id == manager_id))
    manager = manager_result.scalar_one_or_none()
    if not manager:
        raise HTTPException(status_code=404, detail="Manager not found")

    filings_result = await session.execute(
        select(Form13FFiling).where(Form13FFiling.manager_id == manager_id).order_by(Form13FFiling.report_date.desc())
    )
    filings = filings_result.scalars().all()
    if not filings:
        raise HTTPException(status_code=404, detail="No filings found for this manager")

    available_quarters = [f.report_date.isoformat() for f in filings]

    # Resolve which filing to show
    target_filing = filings[0]
    if report_date:
        try:
            target_date = date_type.fromisoformat(report_date)
            target_filing = next((f for f in filings if f.report_date == target_date), filings[0])
        except ValueError:
            pass  # fall back to latest

    # Find the previous filing (the one just before the target)
    target_idx = filings.index(target_filing)
    prev_filing = filings[target_idx + 1] if target_idx + 1 < len(filings) else None

    filing_ids = [target_filing.id] + ([prev_filing.id] if prev_filing else [])
    holdings_result = await session.execute(select(Form13FHolding).where(Form13FHolding.filing_id.in_(filing_ids)))
    all_holdings = holdings_result.scalars().all()

    # Load instruments referenced by any of these holdings
    instrument_ids = {h.instrument_id for h in all_holdings if h.instrument_id is not None}
    instruments: dict[int, Any] = {}
    if instrument_ids:
        instr_result = await session.execute(select(Instrument).where(Instrument.id.in_(instrument_ids)))
        for instr in instr_result.scalars().all():
            instruments[instr.id] = instr

    holdings_by_filing: dict[int, list] = defaultdict(list)
    for h in all_holdings:
        holdings_by_filing[h.filing_id].append(h)

    latest_by_cusip = _aggregate_holdings_by_cusip(holdings_by_filing[target_filing.id])
    prev_by_cusip = _aggregate_holdings_by_cusip(holdings_by_filing[prev_filing.id]) if prev_filing else {}

    portfolio, moves = _build_portfolio_and_moves(
        latest_by_cusip, prev_by_cusip, target_filing.total_value, instruments
    )

    cik_num = str(manager.cik).lstrip("0") or "0"
    sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_num}&type=13F-HR&dateb=&owner=include&count=10"

    return {
        "manager": {"id": manager.id, "name": manager.name, "cik": manager.cik, "sec_url": sec_url},
        "available_quarters": available_quarters,
        "report_date": target_filing.report_date.isoformat(),
        "prev_report_date": prev_filing.report_date.isoformat() if prev_filing else None,
        "total_value": target_filing.total_value,
        "num_positions": len(portfolio),
        "portfolio": portfolio,
        "moves": moves,
    }


@router.get("/api/13f/not-in-portfolio")
async def get_13f_not_in_portfolio(
    min_managers: int = 2,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    """
    Stocks that tracked institutional managers hold but the user does not.
    Includes buy_count: how many managers opened or increased the position this quarter.
    Ranked by manager count, then total AUM.
    Only returns instruments matched in our DB (instrument_id IS NOT NULL).
    """
    # 1. Current portfolio instrument IDs
    latest_date_result = await session.execute(select(func.max(HoldingDaily.date)))
    latest_date = latest_date_result.scalar()
    held_instrument_ids: set[int] = set()
    if latest_date:
        held_result = await session.execute(
            select(HoldingDaily.instrument_id).where(
                HoldingDaily.date == latest_date,
                HoldingDaily.quantity > 0,
            )
        )
        held_instrument_ids = {row.instrument_id for row in held_result.all()}

    # 2. Latest and previous filing ID per manager (prev needed for buy_count)
    managers_result = await session.execute(select(Form13FManager))
    managers_objects = managers_result.scalars().all()
    manager_ids = [m.id for m in managers_objects]
    managers_by_id: dict[int, str] = {m.id: m.name for m in managers_objects}
    if not manager_ids:
        return []

    filings_result = await session.execute(
        select(Form13FFiling)
        .where(Form13FFiling.manager_id.in_(manager_ids))
        .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
    )
    latest_fid_by_mgr: dict[int, int] = {}
    prev_fid_by_mgr: dict[int, int] = {}
    for f in filings_result.scalars().all():
        if f.manager_id not in latest_fid_by_mgr:
            latest_fid_by_mgr[f.manager_id] = f.id
        elif f.manager_id not in prev_fid_by_mgr:
            prev_fid_by_mgr[f.manager_id] = f.id

    all_fids = list(set(latest_fid_by_mgr.values()) | set(prev_fid_by_mgr.values()))
    latest_fids_set = set(latest_fid_by_mgr.values())
    filing_to_manager = {fid: mid for mid, fid in latest_fid_by_mgr.items()}

    # 3. Holdings from latest + prev filings (aggregated by filing+instrument)
    rows_result = await session.execute(
        select(
            Form13FHolding.instrument_id,
            Form13FHolding.filing_id,
            Form13FHolding.value,
            Form13FHolding.shares,
        ).where(
            Form13FHolding.filing_id.in_(all_fids),
            Form13FHolding.instrument_id.is_not(None),
        )
    )
    # agg[(filing_id, instrument_id)] = {value, shares}  — sums multiple share classes
    agg: dict[tuple[int, int], dict] = {}
    for row in rows_result.all():
        k = (row.filing_id, row.instrument_id)
        if k not in agg:
            agg[k] = {"value": 0, "shares": 0}
        agg[k]["value"] += row.value
        agg[k]["shares"] += row.shares

    # 4. Per instrument: manager_count, buy_count, total_value
    by_instrument: dict[int, dict] = {}
    for (fid, inst_id), data in agg.items():
        if fid not in latest_fids_set:
            continue  # only process latest-filing rows here
        if inst_id in held_instrument_ids:
            continue

        manager_id = filing_to_manager[fid]
        prev_fid = prev_fid_by_mgr.get(manager_id)
        prev = agg.get((prev_fid, inst_id)) if prev_fid else None

        if inst_id not in by_instrument:
            by_instrument[inst_id] = {
                "manager_ids": {},  # manager_id -> current_value
                "buy_managers_data": {},  # manager_id -> {change, value_change}
                "sell_managers_data": {},  # manager_id -> {change, value_change}
                "total_value": 0,
            }

        by_instrument[inst_id]["manager_ids"][manager_id] = data["value"]
        by_instrument[inst_id]["total_value"] += data["value"]

        if prev_fid is not None:
            price_now = data["value"] / data["shares"] if data["shares"] else 0
            # Buying: new position OR shares increased by ≥10% (matches _compute_form13f_signal_score)
            if prev is None or data["shares"] > prev["shares"] * 1.10:
                shares_added = data["shares"] - (prev["shares"] if prev else 0)
                transaction_value = round(shares_added * price_now)  # capital deployed
                change = "New" if prev is None else f"+{round((data['shares'] / prev['shares'] - 1) * 100)}%"
                by_instrument[inst_id]["buy_managers_data"][manager_id] = {
                    "change": change,
                    "value_change": transaction_value,
                }
            # Selling: shares decreased by >10%
            elif prev is not None and data["shares"] < prev["shares"] * 0.90:
                shares_sold = prev["shares"] - data["shares"]
                transaction_value = -round(shares_sold * price_now)  # capital extracted, negative
                pct = round((data["shares"] / prev["shares"] - 1) * 100)
                by_instrument[inst_id]["sell_managers_data"][manager_id] = {
                    "change": f"{pct}%",
                    "value_change": transaction_value,
                }

    # 5. Filter, resolve names, sort
    candidates = [
        {"instrument_id": iid, **v} for iid, v in by_instrument.items() if len(v["manager_ids"]) >= min_managers
    ]
    if not candidates:
        return []

    inst_ids = [c["instrument_id"] for c in candidates]
    instr_result = await session.execute(select(Instrument).where(Instrument.id.in_(inst_ids)))
    instruments = {i.id: i for i in instr_result.scalars().all()}

    result = []
    for c in candidates:
        instr = instruments.get(c["instrument_id"])
        if not instr:
            continue
        # All managers sorted by position size descending (biggest holders first)
        all_mgrs = sorted(
            [
                {"name": managers_by_id[mid], "current_value": val}
                for mid, val in c["manager_ids"].items()
                if mid in managers_by_id
            ],
            key=lambda x: x["current_value"],
            reverse=True,
        )
        buy_mgrs = sorted(
            [
                {"name": managers_by_id[mid], **data}
                for mid, data in c["buy_managers_data"].items()
                if mid in managers_by_id
            ],
            key=lambda x: x["value_change"],
            reverse=True,
        )
        sell_mgrs = sorted(
            [
                {"name": managers_by_id[mid], **data}
                for mid, data in c["sell_managers_data"].items()
                if mid in managers_by_id
            ],
            key=lambda x: x["value_change"],  # most negative (biggest exit) first
        )
        result.append(
            {
                "yahoo_symbol": instr.yahoo_symbol,
                "name": instr.name,
                "manager_count": len(c["manager_ids"]),
                "buy_count": len(c["buy_managers_data"]),
                "sell_count": len(c["sell_managers_data"]),
                "total_value": c["total_value"],
                "managers": all_mgrs,  # [{name, current_value}] sorted by size
                "buying_managers": buy_mgrs,  # [{name, change, value_change}]
                "selling_managers": sell_mgrs,  # [{name, change, value_change}]
            }
        )

    result.sort(key=lambda x: (-x["manager_count"], -x["total_value"]))
    return result[:20]
