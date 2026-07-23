from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.agent.rules_strategy import RulesStrategy
from backend.app import get_db_session
from backend.utils.dcf import get_dcf_analyses, get_effective_betas
from backend.utils.form13f import (
    Form13FFilingRow,
    _build_sec_13f_url,
    _compute_form13f_change,
    _compute_form13f_signal_score,
    _score_reason,
)
from backend.utils.piotroski import get_piotroski_f_score
from backend.utils.roic import get_roic
from backend.views._shared import get_rates
from config import BENCHES, PRICE_FIELD, TIMEZONE
from models import (
    FeaturesDaily,
    Form13FFiling,
    Form13FHolding,
    Form13FManager,
    HoldingDaily,
    Instrument,
    InstrumentMetricsDaily,
    PositionReview,
    PricesDaily,
    TradeSuggestion,
    TransactionAction,
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
        # PR rows carry the explicit release date in metrics["release_date"]
        if report.metrics.get("report_type") == "PR":
            announcement_dates.append(date.fromisoformat(report.metrics["release_date"]))
            continue
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


async def _chart_payload(session: AsyncSession, instrument: Instrument, start_date: date) -> dict[str, Any]:
    """Range-dependent chart data: prices, orders, splits, PE history/estimates.

    Served by both the full instrument endpoint (initial page load) and the
    lightweight /prices endpoint (chart range changes).
    """
    symbol = instrument.yahoo_symbol
    price_attr = PRICE_FIELD.lower().replace(" ", "_") + "_price"
    prices_result = await session.execute(
        select(PricesDaily)
        .where(PricesDaily.symbol == symbol, PricesDaily.date >= start_date)
        .order_by(PricesDaily.date.asc())
    )
    prices = {p.date.isoformat(): getattr(p, price_attr) for p in prices_result.scalars().all()}

    orders_result = await session.execute(
        select(TransactionHistory)
        .where(TransactionHistory.ticker == symbol, TransactionHistory.timestamp >= start_date)
        .order_by(TransactionHistory.timestamp)
    )
    orders = {
        o.timestamp.isoformat(): {"action": o.action.value, "total": o.total}
        for o in orders_result.scalars().all()
    }

    # Benchmark closes for the "% change vs index" overlay (skipped when the
    # instrument is itself a benchmark)
    bench_symbols = [b for b in BENCHES if b != symbol]
    benchmarks: dict[str, dict[str, float]] = {b: {} for b in bench_symbols}
    if bench_symbols:
        bench_result = await session.execute(
            select(PricesDaily)
            .where(PricesDaily.symbol.in_(bench_symbols), PricesDaily.date >= start_date)
            .order_by(PricesDaily.date.asc())
        )
        for p in bench_result.scalars().all():
            benchmarks[p.symbol][p.date.isoformat()] = getattr(p, price_attr)

    yh = instrument.yahoo
    pes = (yh.pes or {}) if yh else {}
    today = date.today()
    return {
        "prices": prices,
        "orders": orders,
        "benchmarks": benchmarks,
        "pe_history": {
            k: v["pe_ratio"] for k, v in pes.items() if start_date <= date.fromisoformat(k) <= today
        },
        # Wisesheets forward year-end estimates — charted as a dashed continuation
        "pe_estimates": {k: v["pe_ratio"] for k, v in pes.items() if date.fromisoformat(k) > today},
        "splits": {k: v for k, v in (yh.splits or {}).items() if date.fromisoformat(k) >= start_date}
        if yh
        else {},
    }


@router.get("/api/instrument/{symbol}/prices")
async def get_instrument_prices(
    symbol: str, days: int = 365, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Chart-window data only — cheap endpoint for range changes on the Stock page."""
    start_date = datetime.now(TIMEZONE).date() - timedelta(days=days)
    instrument = (
        await session.execute(
            select(Instrument)
            .filter(Instrument.yahoo_symbol == symbol)
            .options(selectinload(Instrument.yahoo))
        )
    ).scalars().first()
    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")
    return await _chart_payload(session, instrument, start_date)


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

    chart = await _chart_payload(session, instrument, start_date)

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
        # Rounded to match portfolio.py so the composite score is identical on both pages
        "recommendationMean": round(yd["recommendationMean"], 2) if yd.get("recommendationMean") else None,
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
        # Percent, statement-based with info-proxy fallback (same as Holdings)
        "roic": get_roic(yd, instrument.yahoo.balance_sheet, instrument.yahoo.income_stmt)
        if instrument.yahoo
        else None,
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

    # (score, conviction) pairs for the aggregate 13F score — same formula as
    # _get_form13f_for_instruments (conviction-weighted, zero scores excluded)
    scoring_pairs: list[tuple[float, float]] = []

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
        if score != 0:
            scoring_pairs.append((score, latest["value"] / filing_total if filing_total > 0 else 0.0))

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

    form13f_score: float | None = None
    if scoring_pairs:
        total_conviction = sum(c for _, c in scoring_pairs)
        if total_conviction > 0:
            weighted = sum(s * c for s, c in scoring_pairs) / total_conviction
        else:
            weighted = sum(s for s, _ in scoring_pairs) / len(scoring_pairs)
        form13f_score = round(max(-2.0, min(2.0, weighted)), 1)

    # User's position (if held): portfolio_pct, market_value, profit, return_pct
    my_position: dict[str, Any] | None = None
    latest_date_result = await session.execute(select(func.max(HoldingDaily.date)))
    latest_date = latest_date_result.scalar()
    if latest_date:
        holding_result = await session.execute(
            select(HoldingDaily).where(
                HoldingDaily.instrument_id == instrument.id,
                HoldingDaily.date == latest_date,
            )
        )
        user_holding = holding_result.scalar_one_or_none()
        if user_holding and user_holding.quantity > 0:
            currency_rates = await get_rates(session)
            # Same holdings-sum denominator as the Holdings page, aggregated in
            # SQL instead of loading every holding row + instrument into the ORM
            rate_expr = case(currency_rates, value=Instrument.currency, else_=1.0)
            total_portfolio_value = (
                await session.execute(
                    select(func.sum(HoldingDaily.quantity * HoldingDaily.current_price * rate_expr))
                    .join(Instrument, Instrument.id == HoldingDaily.instrument_id)
                    .where(HoldingDaily.date == latest_date)
                )
            ).scalar() or 0.0
            market_value_gbp = (
                user_holding.quantity * user_holding.current_price * currency_rates.get(instrument.currency, 1.0)
            )
            portfolio_pct = (market_value_gbp / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            profit = user_holding.ppl if user_holding.ppl is not None else 0
            cost_basis = (market_value_gbp - user_holding.ppl) if user_holding.ppl is not None else 0
            return_pct = (
                (user_holding.ppl / cost_basis * 100.0) if user_holding.ppl is not None and cost_basis > 0 else 0.0
            )
            buy_actions = [TransactionAction.MARKET_BUY, TransactionAction.LIMIT_BUY]
            dividend_actions = [
                TransactionAction.DIVIDEND,
                TransactionAction.DIVIDEND_PROPERTY,
                TransactionAction.DIVIDEND_TAX_EXEMPT,
            ]
            first_buy, dividends_gbp = (
                await session.execute(
                    select(
                        func.min(TransactionHistory.timestamp).filter(
                            TransactionHistory.action.in_(buy_actions)
                        ),
                        func.coalesce(
                            func.sum(TransactionHistory.total).filter(
                                TransactionHistory.action.in_(dividend_actions)
                            ),
                            0.0,
                        ),
                    ).where(TransactionHistory.ticker == instrument.yahoo_symbol)
                )
            ).one()
            my_position = {
                "portfolio_pct": round(portfolio_pct, 2),
                "market_value": round(market_value_gbp, 2),
                "profit": round(profit, 2),
                "return_pct": round(return_pct, 2),
                "quantity": user_holding.quantity,
                "avg_price": user_holding.avg_price,
                "first_buy_date": first_buy.date().isoformat() if first_buy else None,
                "dividends_gbp": round(dividends_gbp, 2),
            }

    effective_betas = await get_effective_betas([instrument], session)
    dcf_analyses = await get_dcf_analyses([instrument], session, effective_betas=effective_betas)
    dcf_analysis = (
        dcf_analyses[0]
        if dcf_analyses
        else {
            "price": None,
            "low": None,
            "high": None,
            "implied_growth": None,
            "implied_growth_status": "unavailable",
        }
    )
    dcf_price: float | None = dcf_analysis["price"]
    dcf_low: float | None = dcf_analysis["low"]
    dcf_high: float | None = dcf_analysis["high"]
    current_price = fundamentals.get("currentPrice")
    dcf_diff: float | None = (dcf_price / current_price - 1) if (dcf_price and current_price) else None

    review_row = (
        await session.execute(
            select(PositionReview)
            .where(PositionReview.instrument_id == instrument.id)
            .order_by(PositionReview.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    position_review = (
        {**review_row.payload, "reviewed_at": review_row.created_at.date().isoformat(), "model": review_row.model}
        if review_row
        else None
    )

    yh = instrument.yahoo
    f_score_result = get_piotroski_f_score(yh.cashflow, yh.balance_sheet, yh.income_stmt) if yh else None

    # Nightly snapshots (update_features.py) — the live screener needs
    # RSI/technicals for the whole portfolio, too heavy to recompute per stock.
    # A few extra rows beyond the newest feed the thesis sell-streak count.
    features_rows = (
        await session.execute(
            select(
                FeaturesDaily.date,
                FeaturesDaily.screener_score,
                FeaturesDaily.passed_screeners,
                FeaturesDaily.thesis_rule_eval,
            )
            .where(FeaturesDaily.instrument_id == instrument.id)
            .order_by(FeaturesDaily.date.desc())
            .limit(RulesStrategy.SELL_RULE_PERSISTENCE + 5)
        )
    ).all()
    features_row = features_rows[0] if features_rows else None
    # Consecutive newest snapshots with the sell leg firing — the same
    # backward-only streak the agent uses to escalate a sell to EXIT.
    thesis_sell_streak = 0
    for r in features_rows:
        if not (r.thesis_rule_eval or {}).get("sell_signal"):
            break
        thesis_sell_streak += 1

    # Quality trend (ROIC/margins/growth, percent units). FeaturesDaily only
    # started 2026-07-12, so the frontend hides the chart until enough points
    # accrete — no deploy needed when it lights up.
    quality_rows = (
        await session.execute(
            select(
                FeaturesDaily.date,
                FeaturesDaily.roic,
                FeaturesDaily.gross_margin,
                FeaturesDaily.operating_margin,
                FeaturesDaily.profit_margin,
                FeaturesDaily.revenue_growth,
            )
            .where(FeaturesDaily.instrument_id == instrument.id)
            .order_by(FeaturesDaily.date.desc())
            .limit(750)
        )
    ).all()
    quality_history = [
        {
            "date": r.date.isoformat(),
            "roic": r.roic,
            "gross_margin": r.gross_margin,
            "operating_margin": r.operating_margin,
            "profit_margin": r.profit_margin,
            "revenue_growth": r.revenue_growth,
        }
        for r in reversed(quality_rows)
    ]

    suggestion_row = (
        await session.execute(
            select(TradeSuggestion)
            .where(TradeSuggestion.instrument_id == instrument.id)
            .order_by(TradeSuggestion.date.desc(), TradeSuggestion.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    agent_suggestion = (
        {
            "id": suggestion_row.id,
            "date": suggestion_row.date.isoformat(),
            "strategy": suggestion_row.strategy,
            "action": suggestion_row.action,
            "quantity": suggestion_row.quantity,
            "value_gbp": suggestion_row.value_gbp,
            "weight_before": suggestion_row.weight_before,
            "weight_after": suggestion_row.weight_after,
            "score": suggestion_row.score,
            "rationale": suggestion_row.rationale or {},
            "constraint_adjustments": suggestion_row.constraint_adjustments or [],
            "status": suggestion_row.status,
        }
        if suggestion_row
        else None
    )

    # Latest insider aggregates from the most recent metrics row.
    insider_row = (
        await session.execute(
            select(
                InstrumentMetricsDaily.insider_buy_count_90d,
                InstrumentMetricsDaily.insider_sell_count_90d,
                InstrumentMetricsDaily.insider_net_value_90d,
            )
            .where(InstrumentMetricsDaily.instrument_id == instrument.id)
            .order_by(InstrumentMetricsDaily.date.desc())
            .limit(1)
        )
    ).first()
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
            "thesis": instrument.thesis,
            "tags": instrument.tags,
            "quote_type": yd.get("quoteType"),
        },
        "fundamentals": fundamentals,
        "earnings": (yh.earnings or {}) if yh else {},
        "cashflow": (yh.cashflow or {}) if yh else {},
        **chart,
        "recommendations": (yh.recommendations or {}) if yh else {},
        "news": yh.news if yh else [],
        "earnings_reports": await _build_earnings_reports(instrument, yh, session),
        "form13f_holdings": form13f_holdings,
        "form13f_as_of": form13f_as_of,
        "form13f_score": form13f_score,
        "screener_score": features_row[1] if features_row else None,
        "passed_screeners": (features_row[2] or []) if features_row else [],
        "screener_as_of": features_row[0].isoformat() if features_row else None,
        "thesis_rule_eval": features_row[3] if features_row else None,
        "thesis_sell_streak": thesis_sell_streak,
        "sell_rule_persistence": RulesStrategy.SELL_RULE_PERSISTENCE,
        "quality_history": quality_history,
        "agent_suggestion": agent_suggestion,
        "my_position": my_position,
        "analyst_price_targets": (yh.analyst_price_targets or {}) if yh else {},
        "dcf_price": dcf_price,
        "dcf_diff": dcf_diff,
        "dcf_low": dcf_low,
        "dcf_high": dcf_high,
        "dcf_implied_growth": dcf_analysis["implied_growth"],
        "dcf_implied_growth_status": dcf_analysis["implied_growth_status"],
        "position_review": position_review,
        "f_score": f_score_result["score"] if f_score_result else None,
        "f_score_details": f_score_result["details"] if f_score_result else None,
        "insider_buy_count_90d": insider_row[0] if insider_row else None,
        "insider_sell_count_90d": insider_row[1] if insider_row else None,
        "insider_net_value_90d": insider_row[2] if insider_row else None,
    }
