"""SEC companyfacts fetch, tag resolution, vintage selection, and statement ROIC.

Companyfacts JSON is the structured fundamentals source (not HTML filings).
Vintage rules: PIT = earliest `filed` per (concept, unit, start, end);
as-restated = latest `filed`. Join as-of uses strict `filed < as_of`.
SEC depth must not feed screener_score / roic_consistency until a cutover rebuild.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Literal, Optional, Sequence, Union

import requests

from backend.utils.roic import _roic_for_period, get_roic_ttm_series
from backend.utils.splits import adjust_share_count
from config import DEFAULT_SEC_USER_AGENT, SEC_USER_AGENT

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Prefer USD/EUR/… currency units over shares.
_CURRENCY_PREF = ("USD", "EUR", "GBP", "CAD", "DKK", "BRL", "JPY", "CHF", "CNY")

FY_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"})
Q_FORMS = frozenset({"10-Q", "10-Q/A"})
# Instant (balance sheet) vs duration (income / cash flow) heuristics.
_ANNUAL_DAYS = (300, 400)
_QUARTER_DAYS = (70, 100)

Vintage = Literal["pit", "as_restated"]


# Era-aware synonym lists: first matching concept present in facts wins.
# Tuples are (taxonomy, concept) or (taxonomy, concept, valid_from).
TagChoice = Union[tuple[str, str], tuple[str, str, Optional[date]]]

TAG_MAP: dict[str, list[TagChoice]] = {
    "operating_income": [
        ("us-gaap", "OperatingIncomeLoss"),
    ],
    "pretax_income": [
        (
            "us-gaap",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ),
        (
            "us-gaap",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
        ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"),
        ("us-gaap", "PretaxIncomeLoss"),
    ],
    "tax_expense": [
        ("us-gaap", "IncomeTaxExpenseBenefit"),
    ],
    "assets": [
        ("us-gaap", "Assets"),
    ],
    "assets_current": [
        ("us-gaap", "AssetsCurrent"),
    ],
    "liabilities_current": [
        ("us-gaap", "LiabilitiesCurrent"),
    ],
    "cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsAndShortTermInvestments"),
    ],
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "long_term_debt": [
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtNoncurrent"),
    ],
    "current_debt": [
        ("us-gaap", "LongTermDebtCurrent"),
        ("us-gaap", "ShortTermBorrowings"),
    ],
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", date(2018, 1, 1)),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "Revenues"),
    ],
    "gross_profit": [
        ("us-gaap", "GrossProfit"),
    ],
    "net_income": [
        ("us-gaap", "NetIncomeLoss"),
    ],
    "ocf": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ],
    "capex": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ],
    # Market-cap share count (point-in-time). DEI cover-date preferred.
    "shares_outstanding_dei": [
        ("dei", "EntityCommonStockSharesOutstanding"),
    ],
    "shares_outstanding_bs": [
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
    # F-Score dilution — match live Yahoo Ordinary Shares / Share Issued.
    "shares_issued": [
        ("us-gaap", "CommonStockSharesIssued"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
}


@dataclass(frozen=True)
class FactPoint:
    """One XBRL fact observation."""

    taxonomy: str
    concept: str
    end: date
    start: date | None
    filed: date
    form: str
    value: float
    unit: str
    accn: str
    fp: str | None


@dataclass(frozen=True)
class ShareCount:
    """Point-in-time share count with the date the count refers to."""

    value: float
    ref_date: date
    concept: str


def require_sec_user_agent() -> None:
    """Refuse to hit data.sec.gov unless SEC_USER_AGENT is overridden from the placeholder.

    Checks the resolved value (env / IMDS / GCP metadata via get_env_var), not the
    transport — production often sets T212_SEC_USER_AGENT only in instance metadata.
    """
    if not SEC_USER_AGENT.strip() or SEC_USER_AGENT.strip() == DEFAULT_SEC_USER_AGENT:
        raise SystemExit(
            "Set T212_SEC_USER_AGENT to an identifiable 'AppName contact@email' before calling data.sec.gov"
        )


def pad_cik(cik: str) -> str:
    """Zero-pad a CIK to 10 digits."""
    return str(cik).strip().zfill(10)


def fetch_companyfacts(cik: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    """GET companyfacts JSON for a CIK (User-Agent only — never Host: www.sec.gov)."""
    require_sec_user_agent()
    url = COMPANYFACTS_URL.format(cik=pad_cik(cik))
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    http = session or requests
    response = http.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _duration_days(start: date | None, end: date) -> int | None:
    if start is None:
        return None
    return (end - start).days


def _is_annual_duration(start: date | None, end: date, form: str, fp: str | None) -> bool:
    days = _duration_days(start, end)
    if days is not None:
        return _ANNUAL_DAYS[0] <= days <= _ANNUAL_DAYS[1]
    if form in FY_FORMS and (fp in (None, "FY") or fp == "FY"):
        return True
    return form in FY_FORMS and start is None


def _is_quarter_duration(start: date | None, end: date, form: str, fp: str | None) -> bool:
    days = _duration_days(start, end)
    if days is not None:
        return _QUARTER_DAYS[0] <= days <= _QUARTER_DAYS[1]
    return form in Q_FORMS and fp in ("Q1", "Q2", "Q3", "Q4")


def iter_concept_facts(facts: dict[str, Any], taxonomy: str, concept: str) -> list[FactPoint]:
    """All numeric observations for one concept."""
    block = (facts.get("facts") or {}).get(taxonomy, {}).get(concept)
    if not block:
        return []
    out: list[FactPoint] = []
    for unit, rows in (block.get("units") or {}).items():
        for row in rows:
            end = _parse_date(row.get("end"))
            filed = _parse_date(row.get("filed"))
            if end is None or filed is None:
                continue
            try:
                value = float(row["val"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                FactPoint(
                    taxonomy=taxonomy,
                    concept=concept,
                    end=end,
                    start=_parse_date(row.get("start")),
                    filed=filed,
                    form=str(row.get("form") or ""),
                    value=value,
                    unit=unit,
                    accn=str(row.get("accn") or ""),
                    fp=row.get("fp"),
                )
            )
    return out


def _preferred_unit(points: Sequence[FactPoint]) -> str | None:
    units = {p.unit for p in points}
    for pref in _CURRENCY_PREF:
        if pref in units:
            return pref
    return next(iter(units), None)


def resolve_tag(
    facts: dict[str, Any],
    tag_key: str,
    *,
    as_of_period: date | None = None,
) -> tuple[str, str] | None:
    """Pick the first era-valid synonym present in companyfacts."""
    for choice in TAG_MAP[tag_key]:
        taxonomy, concept = choice[0], choice[1]
        valid_from = choice[2] if len(choice) > 2 else None
        if valid_from is not None and as_of_period is not None and as_of_period < valid_from:
            continue
        if (facts.get("facts") or {}).get(taxonomy, {}).get(concept):
            return taxonomy, concept
    return None


def select_vintage(
    points: Iterable[FactPoint],
    vintage: Vintage,
) -> dict[tuple[date | None, date], FactPoint]:
    """One fact per (start, end): earliest filed (PIT) or latest filed (as-restated)."""
    best: dict[tuple[date | None, date], FactPoint] = {}
    for point in points:
        key = (point.start, point.end)
        prev = best.get(key)
        if prev is None:
            best[key] = point
            continue
        if vintage == "pit":
            if point.filed < prev.filed or (point.filed == prev.filed and point.accn < prev.accn):
                best[key] = point
        else:
            if point.filed > prev.filed or (point.filed == prev.filed and point.accn > prev.accn):
                best[key] = point
    return best


def _filter_period_type(
    points: Sequence[FactPoint],
    *,
    annual: bool,
    instant: bool,
) -> list[FactPoint]:
    """Keep annual or quarterly observations; instant BS facts use form + end only."""
    out: list[FactPoint] = []
    for p in points:
        if instant:
            if annual and (p.form in FY_FORMS or (p.form in FY_FORMS | Q_FORMS and p.fp == "FY")):
                out.append(p)
            elif not annual and p.form in Q_FORMS:
                out.append(p)
            continue
        if annual and _is_annual_duration(p.start, p.end, p.form, p.fp):
            out.append(p)
        elif not annual and _is_quarter_duration(p.start, p.end, p.form, p.fp):
            out.append(p)
    return out


def _pick_vintage(points: Sequence[FactPoint], vintage: Vintage) -> FactPoint:
    """Earliest or latest filed among candidates for one period."""
    best = points[0]
    for point in points[1:]:
        if vintage == "pit":
            if point.filed < best.filed or (point.filed == best.filed and point.accn < best.accn):
                best = point
        elif point.filed > best.filed or (point.filed == best.filed and point.accn > best.accn):
            best = point
    return best


def concept_series(
    facts: dict[str, Any],
    tag_key: str,
    *,
    vintage: Vintage,
    annual: bool,
    instant: bool = False,
    unit: str | None = None,
) -> dict[date, FactPoint]:
    """Map period-end → selected fact for a logical tag.

    Merges era-aware synonyms: for each period end, only synonyms whose
    `valid_from` is None or <= end compete, and the earliest TAG_MAP entry wins.
    Vintage selection then runs within that winning synonym's filings.
    """
    # (start, end) -> priority -> [FactPoint]
    by_period: dict[tuple[date | None, date], dict[int, list[FactPoint]]] = {}
    for priority, choice in enumerate(TAG_MAP[tag_key]):
        taxonomy, concept = choice[0], choice[1]
        valid_from = choice[2] if len(choice) > 2 else None
        raw = iter_concept_facts(facts, taxonomy, concept)
        if not raw:
            continue
        use_unit = unit or _preferred_unit(raw)
        if use_unit is None:
            continue
        typed = _filter_period_type([p for p in raw if p.unit == use_unit], annual=annual, instant=instant)
        for point in typed:
            if valid_from is not None and point.end < valid_from:
                continue
            key = (point.start, point.end)
            by_period.setdefault(key, {}).setdefault(priority, []).append(point)

    by_end: dict[date, FactPoint] = {}
    for (_start, end), pri_map in by_period.items():
        best_pri = min(pri_map)
        chosen = _pick_vintage(pri_map[best_pri], vintage)
        prev = by_end.get(end)
        if prev is None or (chosen.form in FY_FORMS and prev.form not in FY_FORMS):
            by_end[end] = chosen
        elif vintage == "as_restated" and chosen.filed > prev.filed:
            by_end[end] = chosen
        elif vintage == "pit" and chosen.filed < prev.filed:
            by_end[end] = chosen
    return by_end


def build_statement_pair(
    facts: dict[str, Any],
    *,
    vintage: Vintage,
    annual: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Yahoo-shaped balance_sheet / income_stmt dicts for `roic.py` helpers."""
    assets = concept_series(facts, "assets", vintage=vintage, annual=annual, instant=True)
    liab = concept_series(facts, "liabilities_current", vintage=vintage, annual=annual, instant=True)
    cash = concept_series(facts, "cash", vintage=vintage, annual=annual, instant=True)
    equity = concept_series(facts, "equity", vintage=vintage, annual=annual, instant=True)
    ltd = concept_series(facts, "long_term_debt", vintage=vintage, annual=annual, instant=True)
    cur_debt = concept_series(facts, "current_debt", vintage=vintage, annual=annual, instant=True)
    assets_cur = concept_series(facts, "assets_current", vintage=vintage, annual=annual, instant=True)

    op_inc = concept_series(facts, "operating_income", vintage=vintage, annual=annual)
    pretax = concept_series(facts, "pretax_income", vintage=vintage, annual=annual)
    tax = concept_series(facts, "tax_expense", vintage=vintage, annual=annual)
    revenue = concept_series(facts, "revenue", vintage=vintage, annual=annual)
    gross = concept_series(facts, "gross_profit", vintage=vintage, annual=annual)
    net_income = concept_series(facts, "net_income", vintage=vintage, annual=annual)

    if not annual:
        op_inc = _with_derived_q4(facts, "operating_income", op_inc, vintage=vintage)
        pretax = _with_derived_q4(facts, "pretax_income", pretax, vintage=vintage)
        tax = _with_derived_q4(facts, "tax_expense", tax, vintage=vintage)
        revenue = _with_derived_q4(facts, "revenue", revenue, vintage=vintage)
        gross = _with_derived_q4(facts, "gross_profit", gross, vintage=vintage)
        net_income = _with_derived_q4(facts, "net_income", net_income, vintage=vintage)

    balance_sheet: dict[str, dict[str, float]] = {}
    for end, point in assets.items():
        row: dict[str, float] = {"Total Assets": point.value}
        if end in liab:
            row["Current Liabilities"] = liab[end].value
        if end in cash:
            row["Cash And Cash Equivalents"] = cash[end].value
        if end in equity:
            row["Stockholders Equity"] = equity[end].value
        if end in ltd:
            row["Long Term Debt"] = ltd[end].value
        if end in cur_debt:
            row["Current Debt"] = cur_debt[end].value
        if "Long Term Debt" in row or "Current Debt" in row:
            row["Total Debt"] = row.get("Long Term Debt", 0.0) + row.get("Current Debt", 0.0)
        if end in assets_cur:
            row["Current Assets"] = assets_cur[end].value
        balance_sheet[end.isoformat()] = row

    income_stmt: dict[str, dict[str, float]] = {}
    ends = set(op_inc) | set(revenue) | set(gross) | set(net_income)
    for end in ends:
        row: dict[str, float] = {}
        if end in op_inc:
            row["Operating Income"] = op_inc[end].value
        if end in pretax:
            row["Pretax Income"] = pretax[end].value
        if end in tax:
            row["Tax Provision"] = tax[end].value
        if end in revenue:
            row["Total Revenue"] = revenue[end].value
        if end in gross:
            row["Gross Profit"] = gross[end].value
        if end in net_income:
            row["Net Income"] = net_income[end].value
        if row:
            income_stmt[end.isoformat()] = row

    return balance_sheet, income_stmt


def _with_derived_q4(
    facts: dict[str, Any],
    tag_key: str,
    quarterly: dict[date, FactPoint],
    *,
    vintage: Vintage,
) -> dict[date, FactPoint]:
    """Derive Q4 = FY − (Q1+Q2+Q3) when the 10-K only tags the annual duration."""
    annual = concept_series(facts, tag_key, vintage=vintage, annual=True, instant=False)
    if not annual:
        return quarterly
    out = dict(quarterly)
    for fy_end, fy_point in annual.items():
        q_ends = sorted(e for e in quarterly if fy_end - timedelta(days=370) < e <= fy_end)
        if len(q_ends) < 3:
            continue
        prior = [e for e in q_ends if e < fy_end][-3:]
        if len(prior) != 3:
            continue
        if fy_end in out:
            continue
        q_sum = sum(quarterly[e].value for e in prior)
        q4_val = fy_point.value - q_sum
        out[fy_end] = FactPoint(
            taxonomy=fy_point.taxonomy,
            concept=fy_point.concept,
            end=fy_end,
            start=prior[-1] + timedelta(days=1),
            filed=fy_point.filed,
            form=fy_point.form,
            value=q4_val,
            unit=fy_point.unit,
            accn=fy_point.accn,
            fp="Q4",
        )
    return out


def fy_roic_series(
    facts: dict[str, Any],
    *,
    vintage: Vintage = "as_restated",
) -> list[tuple[date, float, date]]:
    """(period_end, roic_pct, filed) newest-last for charting; filed is the fact's filed date."""
    balance_sheet, income_stmt = build_statement_pair(facts, vintage=vintage, annual=True)
    op_ann = concept_series(facts, "operating_income", vintage=vintage, annual=True)
    out: list[tuple[date, float, date]] = []
    for end_str in sorted(balance_sheet.keys()):
        end = date.fromisoformat(end_str)
        is_row = income_stmt.get(end_str)
        if is_row is None:
            prior = [d for d in income_stmt if d <= end_str]
            if not prior:
                continue
            is_row = income_stmt[max(prior)]
        roic = _roic_for_period(balance_sheet[end_str], is_row)
        if roic is None:
            continue
        filed_point = op_ann.get(end)
        filed = filed_point.filed if filed_point else end
        out.append((end, float(roic), filed))
    return out


def as_restated_roic_payload(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact FY ROIC series for caching on SecCompanyFacts / the stock page."""
    return [
        {"period_end": e.isoformat(), "roic": round(r, 2), "source": "sec_as_restated"}
        for e, r, _f in fy_roic_series(facts, vintage="as_restated")
    ]


def restatement_retention_ok(facts: dict[str, Any], concept: str = "OperatingIncomeLoss") -> dict[str, Any]:
    """Check whether companyfacts keeps multiple filed vintages for one period.

    Returns a small diagnostic dict; `ok` is True when at least one (start,end)
    has two distinct `filed` dates (superseded value retained).
    """
    points = iter_concept_facts(facts, "us-gaap", concept)
    by_period: dict[tuple[date | None, date], set[date]] = {}
    for p in points:
        if p.form not in FY_FORMS:
            continue
        by_period.setdefault((p.start, p.end), set()).add(p.filed)
    multi = {k: sorted(v) for k, v in by_period.items() if len(v) > 1}
    return {
        "ok": bool(multi),
        "multi_vintage_periods": len(multi),
        "example": (
            {
                "start": next(iter(multi))[0].isoformat() if next(iter(multi))[0] else None,
                "end": next(iter(multi))[1].isoformat(),
                "filed": [d.isoformat() for d in next(iter(multi.values()))],
            }
            if multi
            else None
        ),
    }


def share_count_for_market_cap(
    facts: dict[str, Any],
    *,
    vintage: Vintage,
    period_end: date,
    as_of: date,
) -> ShareCount | None:
    """Point-in-time shares for market cap; DEI cover-date preferred.

    Only counts with `filed < as_of` are eligible — the cover-date window must
    not pull a filing the market had not seen yet.
    """
    for tag_key in ("shares_outstanding_dei", "shares_outstanding_bs"):
        series = concept_series(facts, tag_key, vintage=vintage, annual=True, instant=True)
        candidates = [
            p
            for e, p in series.items()
            if e <= period_end + timedelta(days=150) and p.filed < as_of
        ]
        if not candidates:
            continue
        # Factor keys to the count's own end — a cover-date BS count can sit up to
        # 150d after period_end; using period_end would miss splits in between.
        point = min(candidates, key=lambda p: abs((p.end - period_end).days))
        return ShareCount(value=point.value, ref_date=point.end, concept=point.concept)
    return None


def share_count_for_fscore(
    facts: dict[str, Any],
    *,
    vintage: Vintage,
    period_end: date,
    as_of: date,
) -> ShareCount | None:
    """BS issued/outstanding for Piotroski dilution (match live Yahoo behaviour)."""
    series = concept_series(facts, "shares_issued", vintage=vintage, annual=True, instant=True)
    known = {e: p for e, p in series.items() if p.filed < as_of}
    point = known.get(period_end)
    if point is None:
        prior = [e for e in known if e <= period_end]
        if not prior:
            return None
        point = known[max(prior)]
    return ShareCount(value=point.value, ref_date=point.end, concept=point.concept)


def margin_pct(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom == 0:
        return None
    return float(numer / denom * 100.0)


def compute_sec_feature_snapshot(
    facts: dict[str, Any],
    *,
    as_of: date,
    splits: dict[str, Any] | None,
    adj_close: float | None,
    vintage: Vintage = "pit",
) -> dict[str, Any] | None:
    """Statement Tier B fields known strictly before `as_of` (`filed < as_of`)."""
    balance_sheet, income_stmt = build_statement_pair(facts, vintage=vintage, annual=True)
    q_bs, q_is = build_statement_pair(facts, vintage=vintage, annual=False)

    # Drop periods whose operating-income (or assets) filed date is not yet known.
    op_ann = concept_series(facts, "operating_income", vintage=vintage, annual=True)
    assets_ann = concept_series(facts, "assets", vintage=vintage, annual=True, instant=True)

    def known(end_iso: str) -> bool:
        end = date.fromisoformat(end_iso)
        filed_dates = []
        if end in op_ann:
            filed_dates.append(op_ann[end].filed)
        if end in assets_ann:
            filed_dates.append(assets_ann[end].filed)
        if not filed_dates:
            return False
        return max(filed_dates) < as_of

    balance_sheet = {k: v for k, v in balance_sheet.items() if known(k)}
    income_stmt = {k: v for k, v in income_stmt.items() if known(k)}
    if not balance_sheet or not income_stmt:
        return None

    # Latest ROIC
    roic = None
    for bs_date in sorted(balance_sheet.keys(), reverse=True):
        is_date = next((d for d in sorted(income_stmt.keys(), reverse=True) if d <= bs_date), None)
        if is_date is None:
            continue
        roic = _roic_for_period(balance_sheet[bs_date], income_stmt[is_date])
        if roic is not None:
            latest_end = date.fromisoformat(bs_date)
            latest_is = income_stmt[is_date]
            latest_bs = balance_sheet[bs_date]
            break
    else:
        return None

    # Quarterly TTM ROIC / trends — filter filed < as_of
    op_q = concept_series(facts, "operating_income", vintage=vintage, annual=False)
    q_bs = {k: v for k, v in q_bs.items() if k in {e.isoformat() for e in op_q if op_q[e].filed < as_of}}
    q_is = {k: v for k, v in q_is.items() if k in {e.isoformat() for e in op_q if op_q[e].filed < as_of}}

    ttm_series = get_roic_ttm_series(q_bs, q_is, max_windows=4)
    roic_ttm = ttm_series[0] if ttm_series else None
    roic_ttm_trend = (ttm_series[0] - ttm_series[-1]) if len(ttm_series) >= 2 else None

    rev = latest_is.get("Total Revenue")
    gross = latest_is.get("Gross Profit")
    op = latest_is.get("Operating Income")
    net = latest_is.get("Net Income")
    gross_margin = margin_pct(gross, rev)
    operating_margin = margin_pct(op, rev)
    profit_margin = margin_pct(net, rev)

    # YoY revenue growth from annual income
    rev_series = concept_series(facts, "revenue", vintage=vintage, annual=True)
    rev_known = {e: p for e, p in rev_series.items() if p.filed < as_of}
    revenue_growth = None
    ends = sorted(rev_known)
    if len(ends) >= 2 and rev_known[ends[-2]].value:
        revenue_growth = (rev_known[ends[-1]].value / rev_known[ends[-2]].value - 1.0) * 100.0

    debt = latest_bs.get("Total Debt")
    equity = latest_bs.get("Stockholders Equity")
    debt_to_equity = (debt / equity * 100.0) if debt is not None and equity not in (None, 0) else None

    rule_of_40 = None
    if revenue_growth is not None and profit_margin is not None:
        rule_of_40 = revenue_growth + profit_margin

    # fcf_yield
    ocf = concept_series(facts, "ocf", vintage=vintage, annual=True)
    capex = concept_series(facts, "capex", vintage=vintage, annual=True)
    fcf_yield = None
    if latest_end in ocf and ocf[latest_end].filed < as_of and adj_close is not None:
        capex_val = capex[latest_end].value if latest_end in capex else 0.0
        fcf = ocf[latest_end].value - abs(capex_val)
        shares = share_count_for_market_cap(
            facts, vintage=vintage, period_end=latest_end, as_of=as_of
        )
        if shares is not None:
            adj_shares = adjust_share_count(shares.value, splits, shares.ref_date)
            if adj_shares and adj_shares > 0:
                mkt_cap = adj_close * adj_shares
                if mkt_cap > 0:
                    fcf_yield = fcf / mkt_cap * 100.0

    # Margin / revenue quarterly trends
    gross_margin_trend_4q = None
    operating_margin_trend_4q = None
    g = concept_series(facts, "gross_profit", vintage=vintage, annual=False)
    o = concept_series(facts, "operating_income", vintage=vintage, annual=False)
    r = concept_series(facts, "revenue", vintage=vintage, annual=False)
    g_ends = sorted(e for e, p in g.items() if p.filed < as_of and e in r and r[e].filed < as_of)
    o_ends = sorted(e for e, p in o.items() if p.filed < as_of and e in r and r[e].filed < as_of)
    if len(g_ends) >= 5 and r[g_ends[-1]].value and r[g_ends[-5]].value:
        m0 = g[g_ends[-1]].value / r[g_ends[-1]].value * 100.0
        m1 = g[g_ends[-5]].value / r[g_ends[-5]].value * 100.0
        gross_margin_trend_4q = m0 - m1
    if len(o_ends) >= 5 and r[o_ends[-1]].value and r[o_ends[-5]].value:
        m0 = o[o_ends[-1]].value / r[o_ends[-1]].value * 100.0
        m1 = o[o_ends[-5]].value / r[o_ends[-5]].value * 100.0
        operating_margin_trend_4q = m0 - m1

    rev_q_ends = sorted(e for e, p in r.items() if p.filed < as_of)
    revenue_growth_4q_avg = None
    if len(rev_q_ends) >= 8:
        growths = []
        for i in range(-4, 0):
            cur, prev = rev_q_ends[i], rev_q_ends[i - 4]
            if r[prev].value:
                growths.append((r[cur].value / r[prev].value - 1.0) * 100.0)
        if growths:
            revenue_growth_4q_avg = sum(growths) / len(growths)

    return {
        "roic": float(roic) if roic is not None else None,
        "roic_ttm": roic_ttm,
        "roic_ttm_trend": roic_ttm_trend,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "profit_margin": profit_margin,
        "revenue_growth": revenue_growth,
        "fcf_yield": fcf_yield,
        "debt_to_equity": debt_to_equity,
        "rule_of_40": rule_of_40,
        "gross_margin_trend_4q": gross_margin_trend_4q,
        "operating_margin_trend_4q": operating_margin_trend_4q,
        "revenue_growth_4q_avg": revenue_growth_4q_avg,
        "period_end": latest_end.isoformat(),
    }


def rate_limited_get(cik: str, *, sleep_s: float = 0.12) -> dict[str, Any]:
    """Fetch companyfacts then sleep to stay under SEC's 10 req/s guidance."""
    data = fetch_companyfacts(cik)
    time.sleep(sleep_s)
    return data
