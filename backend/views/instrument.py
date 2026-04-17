from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app import get_db_session
from backend.utils.dcf import get_dcf_analyses, get_effective_betas
from backend.utils.form13f import (
    Form13FFilingRow,
    _build_sec_13f_url,
    _compute_form13f_change,
    _compute_form13f_signal_score,
    _score_reason,
)
from backend.views._shared import get_rates
from config import PRICE_FIELD, TIMEZONE
from models import (
    Form13FFiling,
    Form13FHolding,
    Form13FManager,
    HoldingDaily,
    Instrument,
    PricesDaily,
    TransactionHistory,
)

router = APIRouter()


async def _build_earnings_reports(instrument, yh, session: AsyncSession) -> list[dict]:
    """
    Build the earnings_reports list for the instrument response.
    For each report, finds the Yahoo announcement date (the date in InstrumentYahoo.earnings
    that falls 0–90 days after the SEC period-end date) and the closing price on that date.
    """
    reports = sorted(instrument.earnings_reports, key=lambda x: x.date, reverse=True)
    if not reports:
        return []

    yh_earnings: dict = (yh.earnings or {}) if yh else {}
    symbol = instrument.yahoo_symbol

    # Parse Yahoo earnings dates once
    yahoo_dates: list[date] = []
    for date_str in yh_earnings:
        try:
            yahoo_dates.append(date.fromisoformat(date_str[:10]))
        except (ValueError, TypeError):
            pass
    yahoo_dates.sort()

    # Collect all announcement dates we'll need prices for
    announcement_dates: list[date] = []
    for report in reports:
        best: date | None = None
        best_delta = float("inf")
        for yd in yahoo_dates:
            delta = (yd - report.date).days
            if 0 <= delta <= 90 and delta < best_delta:
                best_delta = delta
                best = yd
        announcement_dates.append(best)

    # Bulk price lookup for all needed dates (+ buffer for non-trading days)
    dates_to_fetch = {d for d in announcement_dates if d is not None}
    price_map: dict[date, float] = {}
    if dates_to_fetch and symbol:
        min_d = min(dates_to_fetch) - timedelta(days=4)
        max_d = max(dates_to_fetch) + timedelta(days=4)
        rows = await session.execute(
            select(PricesDaily.date, PricesDaily.close_price)
            .where(PricesDaily.symbol == symbol)
            .where(PricesDaily.date >= min_d)
            .where(PricesDaily.date <= max_d)
        )
        for pdate, close in rows.all():
            price_map[pdate] = close

    result = []
    for report, ann_date in zip(reports, announcement_dates):
        price_at_announcement = None
        if ann_date:
            for offset in range(5):
                p = price_map.get(ann_date + timedelta(days=offset))
                if p is not None:
                    price_at_announcement = round(p, 2)
                    break

        result.append(
            {
                "id": report.id,
                "date": report.date.isoformat(),
                "announcement_date": ann_date.isoformat() if ann_date else None,
                "price_at_announcement": price_at_announcement,
                "summary": report.summary,
                "metrics": report.metrics,
                "created_at": report.created_at.isoformat() if report.created_at else None,
            }
        )

    return result


@router.get("/api/instrument/{symbol}")
async def get_instrument(
    symbol: str, days: int = 30, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get detailed data for a specific stock by Yahoo symbol"""
    end_date = datetime.now(TIMEZONE).date()
    start_date = end_date - timedelta(days=days)

    instrument_result = await session.execute(
        select(Instrument)
        .filter(Instrument.yahoo_symbol == symbol)
        .options(selectinload(Instrument.yahoo), selectinload(Instrument.earnings_reports))
    )
    instrument = instrument_result.scalars().first()
    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")

    # Get price data for the period
    prices_result = await session.execute(
        select(PricesDaily)
        .where(PricesDaily.symbol == symbol, PricesDaily.date >= start_date)
        .order_by(PricesDaily.date.asc())
    )

    chart_price_data: dict[str, float] = {}
    prices = prices_result.scalars().all()
    for price in prices:
        chart_price_data[price.date.isoformat()] = getattr(price, PRICE_FIELD.lower().replace(" ", "_") + "_price")

    chart_orders_data: dict[str, dict[str, float | str]] = {}
    orders_result = await session.execute(
        select(TransactionHistory)
        .where(TransactionHistory.ticker == symbol, TransactionHistory.timestamp >= start_date)
        .order_by(TransactionHistory.timestamp)
    )
    orders = orders_result.scalars().all()
    for order in orders:
        chart_orders_data[order.timestamp.isoformat()] = {
            "action": order.action.value,
            "total": order.total,
        }

    yd = (instrument.yahoo.info or {}) if instrument.yahoo else {}

    fundamentals = {
        "marketCap": yd.get("marketCap"),
        "peRatio": yd.get("trailingPE"),
        "forwardPE": yd.get("forwardPE"),
        "pegRatio": yd.get("trailingPegRatio"),
        "beta": yd.get("beta"),
        "dividendYield": yd.get("dividendYield"),
        "totalDebt": yd.get("totalDebt"),
        "totalCash": yd.get("totalCash"),
        "sharesOutstanding": yd.get("sharesOutstanding") or yd.get("impliedSharesOutstanding"),
        "freeCashflow": yd.get("freeCashflow"),
        "operatingCashflow": yd.get("operatingCashflow"),
        "totalRevenue": yd.get("totalRevenue"),
        "revenuePerShare": yd.get("revenuePerShare"),
        "revenueGrowth": yd.get("revenueGrowth"),
        "profitMargins": yd.get("profitMargins"),
        "returnOnAssets": yd.get("returnOnAssets"),
        "returnOnEquity": yd.get("returnOnEquity"),
        # Additional valuation fields
        "enterpriseValue": yd.get("enterpriseValue"),
        "enterpriseToEbitda": yd.get("enterpriseToEbitda"),
        "enterpriseToRevenue": yd.get("enterpriseToRevenue"),
        "priceToSalesTtm": yd.get("priceToSalesTrailing12Months"),
        "priceToBook": yd.get("priceToBook"),
        "ebitda": yd.get("ebitda"),
        "recommendationMean": yd.get("recommendationMean"),
        "recommendationKey": yd.get("recommendationKey"),
        "numberOfAnalystOpinions": yd.get("numberOfAnalystOpinions"),
        "fiftyTwoWeekHighChangePercent": yd.get("fiftyTwoWeekHighChangePercent"),
        "fiftyTwoWeekHigh": yd.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": yd.get("fiftyTwoWeekLow"),
        "currentPrice": yd.get("currentPrice") or yd.get("regularMarketPrice"),
        "grossMargins": yd.get("grossMargins"),
        "operatingMargins": yd.get("operatingMargins"),
        "nextEarningsDate": yd.get("earningsTimestamp"),
        "_rawCurrency": yd.get("financialCurrency"),
    }

    # 13F institutional holders: match by instrument_id (set from Form13FHolding.cusip vs Instrument.isin)
    form13f_holdings: list[dict[str, str | int | float | None]] = []
    holdings_result = await session.execute(
        select(Form13FHolding, Form13FFiling, Form13FManager)
        .join(Form13FFiling, Form13FHolding.filing_id == Form13FFiling.id)
        .join(Form13FManager, Form13FFiling.manager_id == Form13FManager.id)
        .where(Form13FHolding.instrument_id == instrument.id)
    )
    rows = holdings_result.all()

    # Aggregate by (manager_id, filing_id): sum value + shares across share classes
    by_manager_filing: dict[tuple[int, int], Form13FFilingRow] = {}
    for holding, filing, manager in rows:
        key = (manager.id, filing.id)
        if key not in by_manager_filing:
            by_manager_filing[key] = {
                "manager_name": manager.name,
                "manager_id": manager.id,
                "manager_cik": manager.cik,
                "report_date": filing.report_date,
                "accession_number": filing.accession_number,
                "value": 0,
                "shares": 0,
                "filing_total_value": filing.total_value,
            }
        by_manager_filing[key]["value"] += holding.value
        by_manager_filing[key]["shares"] += holding.shares

    # Fetch two most recent filing dates per manager so we can distinguish
    # "manager's first filing ever" (no comparison) from "manager had a prior
    # filing but no prior holding of this instrument" (= genuine new position).
    # Without this, first-time buyers incorrectly show "—" instead of "New".
    all_manager_ids = {mid for (mid, _) in by_manager_filing}
    manager_filing_dates: dict[int, list[date]] = defaultdict(list)
    if all_manager_ids:
        filing_date_rows = await session.execute(
            select(Form13FFiling.manager_id, Form13FFiling.report_date)
            .where(Form13FFiling.manager_id.in_(all_manager_ids))
            .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
        )
        for mid, rdate in filing_date_rows.all():
            manager_filing_dates[mid].append(rdate)

    # Per manager: latest and prev filing for this instrument, compute change
    by_manager: dict[int, list[Form13FFilingRow]] = {}
    for (mid, fid), data in by_manager_filing.items():
        by_manager.setdefault(mid, []).append(data)

    for mid, filings_list in by_manager.items():
        filings_list.sort(key=lambda x: x["report_date"], reverse=True)
        latest = filings_list[0]
        prev = filings_list[1] if len(filings_list) > 1 else None

        if prev is not None:
            shares_prev: int | None = prev["shares"]
            value_prev_h: int | None = prev["value"]
        elif len(manager_filing_dates.get(mid, [])) >= 2:
            # Manager filed before but didn't hold this instrument → new position
            shares_prev = 0
            value_prev_h = 0
        else:
            # Manager has only one filing total → no comparison possible
            shares_prev = None
            value_prev_h = None

        change = _compute_form13f_change(latest["shares"], shares_prev, value=latest["value"], value_prev=value_prev_h)
        report_date_prev = prev["report_date"].isoformat() if prev else None
        filing_total = latest.get("filing_total_value") or 0
        pct_of_portfolio = (latest["value"] / filing_total * 100) if filing_total > 0 else None
        score = _compute_form13f_signal_score(
            latest["shares"],
            shares_prev,
            value=latest["value"],
            value_prev=value_prev_h,
            filing_total_value=filing_total,
        )

        form13f_holdings.append(
            {
                "manager_id": latest["manager_id"],
                "manager_name": latest["manager_name"],
                "value": latest["value"],
                "value_prev": value_prev_h,
                "pct_of_portfolio": round(pct_of_portfolio, 2) if pct_of_portfolio is not None else None,
                "shares": latest["shares"],
                "shares_prev": shares_prev,
                "report_date": latest["report_date"].isoformat(),
                "report_date_prev": report_date_prev,
                "change": change,
                "scored": score != 0,
                "score_reason": _score_reason(score, change, latest["value"], filing_total or None),
                "sec_filing_url": _build_sec_13f_url(latest.get("manager_cik"), latest.get("accession_number")),
            }
        )

    # Sort: scored holders first (by absolute flow to surface biggest moves),
    # then unscored holders by current value descending.
    def _abs_flow(h: dict) -> float:
        shares = h.get("shares") or 0
        shares_prev = h.get("shares_prev")
        value = h.get("value") or 0
        if shares_prev is None:
            return 0.0
        price = value / shares if shares > 0 else 0.0
        return abs((shares - shares_prev) * price)

    form13f_holdings.sort(key=lambda h: (0 if h.get("scored") else 1, -_abs_flow(h), -(h.get("value") or 0)))

    form13f_as_of = max((h["report_date"] for h in form13f_holdings), default=None)

    # User's position (if held): portfolio_pct, market_value, profit, return_pct
    my_position: dict[str, Any] | None = None
    latest_date_result = await session.execute(select(func.max(HoldingDaily.date)))
    latest_date = latest_date_result.scalar()
    if latest_date:
        holding_result = await session.execute(
            select(HoldingDaily)
            .where(
                HoldingDaily.instrument_id == instrument.id,
                HoldingDaily.date == latest_date,
            )
            .options(selectinload(HoldingDaily.instrument))
        )
        user_holding = holding_result.scalar_one_or_none()
        if user_holding and user_holding.quantity > 0:
            currency_rates = await get_rates(session)
            all_holdings_result = await session.execute(
                select(HoldingDaily)
                .join(Instrument)
                .where(HoldingDaily.date == latest_date)
                .options(selectinload(HoldingDaily.instrument))
            )
            all_holdings = all_holdings_result.scalars().all()
            total_portfolio_value = sum(
                h.quantity * h.current_price * currency_rates.get(h.instrument.currency, 1.0) for h in all_holdings
            )
            market_value_gbp = (
                user_holding.quantity * user_holding.current_price * currency_rates.get(instrument.currency, 1.0)
            )
            portfolio_pct = (market_value_gbp / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            profit = user_holding.ppl if user_holding.ppl is not None else 0
            cost_basis = (market_value_gbp - user_holding.ppl) if user_holding.ppl is not None else 0
            return_pct = (
                (user_holding.ppl / cost_basis * 100.0) if user_holding.ppl is not None and cost_basis > 0 else 0.0
            )
            my_position = {
                "portfolio_pct": round(portfolio_pct, 2),
                "market_value": round(market_value_gbp, 2),
                "profit": round(profit, 2),
                "return_pct": round(return_pct, 2),
            }

    effective_betas = await get_effective_betas([instrument], session)
    dcf_analyses = await get_dcf_analyses([instrument], effective_betas=effective_betas)
    dcf_analysis = (
        dcf_analyses[0] if dcf_analyses else {"price": None, "low": None, "high": None, "implied_growth": None}
    )
    dcf_price: float | None = dcf_analysis["price"]
    dcf_low: float | None = dcf_analysis["low"]
    dcf_high: float | None = dcf_analysis["high"]
    current_price = fundamentals.get("currentPrice")
    dcf_diff: float | None = (dcf_price / current_price - 1) if (dcf_price and current_price) else None

    yh = instrument.yahoo
    return {
        "instrument": {
            "id": instrument.id,
            "symbol": instrument.yahoo_symbol,
            "t212_code": instrument.t212_code,
            "name": instrument.name,
            "currency": instrument.currency,
            "sector": yd.get("sector"),
            "country": yd.get("country"),
            "business_summary": yd.get("longBusinessSummary"),
            "quote_type": yd.get("quoteType"),
        },
        "fundamentals": fundamentals,
        "earnings": (yh.earnings or {}) if yh else {},
        "cashflow": (yh.cashflow or {}) if yh else {},
        "prices": chart_price_data,
        "orders": chart_orders_data,
        "pe_history": {k: v["pe_ratio"] for k, v in (yh.pes or {}).items() if date.fromisoformat(k) >= start_date}
        if yh
        else {},
        "splits": {k: v for k, v in (yh.splits or {}).items() if date.fromisoformat(k) >= start_date} if yh else {},
        "recommendations": (yh.recommendations or {}) if yh else {},
        "news": yh.news if yh else [],
        "earnings_reports": await _build_earnings_reports(instrument, yh, session),
        "form13f_holdings": form13f_holdings,
        "form13f_as_of": form13f_as_of,
        "my_position": my_position,
        "analyst_price_targets": (yh.analyst_price_targets or {}) if yh else {},
        "dcf_price": dcf_price,
        "dcf_diff": dcf_diff,
        "dcf_low": dcf_low,
        "dcf_high": dcf_high,
        "dcf_implied_growth": dcf_analysis["implied_growth"],
    }
