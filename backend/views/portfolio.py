import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, Depends
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer, selectinload

from backend.app import get_db_session
from backend.utils.dcf import get_dcf_analyses, get_effective_betas
from backend.utils.drawdown import compute_max_drawdown, underwater_series
from backend.utils.form13f import _get_form13f_for_instruments
from backend.utils.market_breadth import compute_live_rolling_breadth
from utils.market_data import (
    gen_buffett_indicator,
    gen_fear_greed_index,
    gen_market_breadth_indicator,
    gen_sp500_above_sma200,
    get_consumer_sentiment,
    get_hy_oas,
    get_real_yield_10y,
    get_yield_spread,
)
from backend.utils.piotroski import get_piotroski_f_score
from backend.utils.roic import get_roic, get_roic_history, get_roic_ttm_series
from backend.utils.statement_trends import (
    eps_next_quarter_growth,
    eps_revision_ratio,
    gross_margin_trend,
    operating_margin_trend,
    revenue_growth_yoy_avg,
    trend,
)
from backend.utils.screener import calculate_screener_results
from backend.utils.thesis_rules import attach_thesis_rule_eval
from backend.utils.technical import calculate_technical_indicators_for_symbols
from backend.views._shared import (
    PRICE_COLUMN,
    calculate_historical_trends,
    get_rates,
    statement_to_quote_factor,
)
from config import BENCHES, DAYS_PER_YEAR, TAG_CLUSTER_SOFT_CAPS, TIMEZONE, VIX
from data import QUICK_RATIO_THRESHOLDS
from models import (
    EarningsReport,
    HoldingDaily,
    Instrument,
    InstrumentMetricsDaily,
    InstrumentThesisTypedDict,
    InstrumentYahoo,
    PortfolioDaily,
    PositionReview,
    PricesDaily,
    TransactionAction,
    TransactionHistory,
)

router = APIRouter()

# Deposits below this amount are treated as admin/residual entries (e.g.
# "interest on cash" miscategorised upstream) and hidden from chart overlays.
_MIN_DEPOSIT_AMOUNT_GBP = 50.0

# A third of the Yahoo JSONB payload that nothing on this path reads. Deferred
# rather than dropped: still lazy-loadable if a caller ever needs one.
_UNUSED_YAHOO_BLOBS = (
    defer(InstrumentYahoo.news),
    defer(InstrumentYahoo.splits),
    defer(InstrumentYahoo.quarterly_cashflow),
)


def _twrr_wealth_index(
    rows: list[tuple[date, Optional[float]]],
) -> list[Optional[float]]:
    """
    Convert stored annualised TWRR percentages into a cumulative wealth index.

    ``PortfolioDaily.twrr`` is the *annualised* time-weighted return computed
    over calendar days since the first portfolio snapshot (see
    ``scripts/update_returns.py``). Inverting that annualisation recovers the
    cumulative growth factor on each day:

        wealth_index[t] = (1 + twrr[t] / 100) ** (days_since_first / DAYS_PER_YEAR)

    This index is anchored at 1.0 on the first row and is unaffected by
    deposits or withdrawals — the correct series for drawdown analysis.
    Callers must pass the full history so the anchor is inception.
    """
    if not rows:
        return []
    first_date = rows[0][0]
    out: list[Optional[float]] = []
    for d, twrr in rows:
        if twrr is None:
            out.append(None)
            continue
        days = (d - first_date).days
        if days <= 0:
            out.append(1.0)
            continue
        factor = 1.0 + twrr / 100.0
        if factor <= 0:
            out.append(None)
            continue
        out.append(factor ** (days / DAYS_PER_YEAR))
    return out


async def _get_insider_signals(session: AsyncSession, instrument_ids: list[int]) -> dict[int, dict]:
    """Return latest 90d insider aggregates per instrument as {instrument_id: {...}}.

    Pulls the most recent ``InstrumentMetricsDaily`` row (across any date) per
    instrument so the signal survives a brief gap in daily ingestion.
    """
    if not instrument_ids:
        return {}

    latest_sq = (
        select(
            InstrumentMetricsDaily.instrument_id,
            func.max(InstrumentMetricsDaily.date).label("max_date"),
        )
        .where(InstrumentMetricsDaily.instrument_id.in_(instrument_ids))
        .group_by(InstrumentMetricsDaily.instrument_id)
        .subquery()
    )
    rows = await session.execute(
        select(
            InstrumentMetricsDaily.instrument_id,
            InstrumentMetricsDaily.insider_buy_count_90d,
            InstrumentMetricsDaily.insider_sell_count_90d,
            InstrumentMetricsDaily.insider_net_value_90d,
        ).join(
            latest_sq,
            and_(
                InstrumentMetricsDaily.instrument_id == latest_sq.c.instrument_id,
                InstrumentMetricsDaily.date == latest_sq.c.max_date,
            ),
        )
    )
    return {
        inst_id: {
            "insider_buy_count_90d": buy,
            "insider_sell_count_90d": sell,
            "insider_net_value_90d": net,
        }
        for inst_id, buy, sell, net in rows.all()
    }


async def _get_earnings_signals(session: AsyncSession, items: list) -> dict[int, dict]:
    """
    Returns { instrument_id: { earnings_signal, earnings_conviction,
                               since_earnings_pct, earnings_announcement_date } }
    using the most recent EarningsReport for each instrument.
    """
    inst_ids = [h.instrument.id for h in items]
    if not inst_ids:
        return {}

    # Most recent EarningsReport per instrument (one row per instrument)
    latest_sq = (
        select(
            EarningsReport.instrument_id,
            func.max(EarningsReport.date).label("max_date"),
        )
        .where(EarningsReport.instrument_id.in_(inst_ids))
        .group_by(EarningsReport.instrument_id)
        .subquery()
    )
    reports_q = await session.execute(
        select(EarningsReport.instrument_id, EarningsReport.date, EarningsReport.metrics).join(
            latest_sq,
            and_(
                EarningsReport.instrument_id == latest_sq.c.instrument_id,
                EarningsReport.date == latest_sq.c.max_date,
            ),
        )
    )
    latest_reports: dict[int, tuple] = {inst_id: (rdate, metrics) for inst_id, rdate, metrics in reports_q.all()}
    if not latest_reports:
        return {}

    # Per-instrument context: symbol, current_price, yahoo earnings dates (already in memory)
    inst_ctx: dict[int, dict] = {}
    for h in items:
        inst = h.instrument
        yh = inst.yahoo
        info = (yh.info or {}) if yh else {}
        inst_ctx[inst.id] = {
            "symbol": inst.yahoo_symbol,
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "yahoo_earnings": (yh.earnings or {}) if yh else {},
        }

    # SEC saves period-end as EarningsReport.date (map forward to Yahoo's announcement);
    # UK RNS saves the announcement date itself and yfinance often lacks historical LSE
    # earnings, so use it directly.
    announcement_map: dict[int, tuple] = {}  # inst_id -> (symbol, ann_date | None)
    for inst_id, (rdate, _) in latest_reports.items():
        ctx = inst_ctx.get(inst_id, {})
        symbol = ctx.get("symbol")
        best: date | None = None
        if symbol and symbol.endswith(".L"):
            best = rdate
        else:
            best_delta = float("inf")
            for date_str in ctx.get("yahoo_earnings", {}):
                try:
                    yd = date.fromisoformat(date_str[:10])
                    delta = (yd - rdate).days
                    if 0 <= delta <= 90 and delta < best_delta:
                        best_delta = delta
                        best = yd
                except (ValueError, TypeError):
                    pass
        announcement_map[inst_id] = (symbol, best)

    # Bulk price query for all announcement dates
    price_map: dict[tuple, float] = {}
    needed_pairs = {(sym, ann) for _, (sym, ann) in announcement_map.items() if sym and ann}
    if needed_pairs:
        syms = {sym for sym, _ in needed_pairs}
        all_dates = {d for _, d in needed_pairs}
        min_date = min(all_dates) - timedelta(days=4)
        max_date = max(all_dates) + timedelta(days=4)
        rows = await session.execute(
            select(PricesDaily.symbol, PricesDaily.date, PricesDaily.close_price)
            .where(PricesDaily.symbol.in_(syms))
            .where(PricesDaily.date >= min_date)
            .where(PricesDaily.date <= max_date)
        )
        for sym, pdate, close in rows.all():
            price_map[(sym, pdate)] = close

    result: dict[int, dict] = {}
    for inst_id, (report_date, metrics) in latest_reports.items():
        ctx = inst_ctx.get(inst_id, {})
        sym, ann_date = announcement_map.get(inst_id, (None, None))
        current_price = ctx.get("current_price")
        assessment = (metrics or {}).get("investment_assessment") or {}

        price_at_ann = None
        if ann_date and sym:
            for offset in range(5):
                p = price_map.get((sym, ann_date + timedelta(days=offset)))
                if p is not None:
                    price_at_ann = round(p, 2)
                    break

        since_pct = None
        if price_at_ann is not None and current_price is not None:
            since_pct = round(((current_price - price_at_ann) / price_at_ann) * 100, 1)

        result[inst_id] = {
            "earnings_signal": assessment.get("recommendation"),
            "earnings_conviction": assessment.get("conviction"),
            "since_earnings_pct": since_pct,
            "earnings_announcement_date": ann_date.isoformat() if ann_date else None,
            # Fallback for age-decay when no announcement date could be matched.
            "earnings_report_date": report_date.isoformat(),
        }

    return result


async def _get_position_reviews(session: AsyncSession, inst_ids: list[int]) -> dict[int, dict]:
    """
    Returns { instrument_id: { review_action, review_thesis_status,
                               review_signal_strength, review_confidence, reviewed_at } }
    from the most recent PositionReview per instrument.
    """
    if not inst_ids:
        return {}

    latest_sq = (
        select(
            PositionReview.instrument_id,
            func.max(PositionReview.created_at).label("max_created"),
        )
        .where(PositionReview.instrument_id.in_(inst_ids))
        .group_by(PositionReview.instrument_id)
        .subquery()
    )
    rows = await session.execute(
        select(PositionReview.instrument_id, PositionReview.payload, PositionReview.created_at).join(
            latest_sq,
            and_(
                PositionReview.instrument_id == latest_sq.c.instrument_id,
                PositionReview.created_at == latest_sq.c.max_created,
            ),
        )
    )
    return {
        inst_id: {
            "review_action": payload["position_action"],
            "review_thesis_status": payload["thesis_status"],
            "review_signal_strength": payload["signal_strength"],
            "review_confidence": payload["confidence"],
            # .get: rows generated before the headline field was added lack it
            "review_headline": payload.get("headline"),
            "reviewed_at": created_at.date().isoformat(),
        }
        for inst_id, payload, created_at in rows.all()
    }


@router.get("/api/portfolio/current")
async def get_current_portfolio(
    session: AsyncSession = Depends(get_db_session), show_all: bool = False
) -> dict[str, Any]:
    """Get current portfolio holdings with detailed information. show_all=true: all monitored instruments."""
    # Get the latest snapshot date
    result = await session.execute(select(func.max(HoldingDaily.date)))
    latest_date = result.scalar()

    if show_all:
        # All monitored instruments: real holdings where held, mock (quantity=0) for others
        instruments_result = await session.execute(
            select(Instrument)
            .where(Instrument.yahoo_symbol.isnot(None))
            .order_by(Instrument.name)
            .options(selectinload(Instrument.yahoo).options(*_UNUSED_YAHOO_BLOBS))
        )
        instruments = list(instruments_result.scalars().all())

        holdings_result = await session.execute(
            select(HoldingDaily)
            .join(Instrument)
            .filter(HoldingDaily.date == latest_date)
            .options(
                selectinload(HoldingDaily.instrument)
                .selectinload(Instrument.yahoo)
                .options(*_UNUSED_YAHOO_BLOBS)
            )
        )
        holdings_list = holdings_result.scalars().all()
        holding_by_symbol = {h.instrument.yahoo_symbol: h for h in holdings_list}
        symbols_held = set(holding_by_symbol.keys())
        currency_rates = await get_rates(session)
        total_portfolio_value = sum(
            h.quantity * h.current_price * currency_rates[h.instrument.currency] for h in holdings_list
        )
        items = []
        for inst in instruments:
            sym = inst.yahoo_symbol
            if not sym:
                continue
            if sym in symbols_held:
                items.append(holding_by_symbol[sym])
            else:
                info = (inst.yahoo.info or {}) if inst.yahoo else {}
                price = (info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")) or 0
                if not isinstance(price, (int, float)):
                    price = 0
                items.append(
                    SimpleNamespace(
                        instrument=inst,
                        quantity=0,
                        avg_price=None,
                        current_price=float(price) if price else 0,
                        ppl=None,
                        fx_ppl=None,
                        date=latest_date or datetime.now(TIMEZONE).date(),
                    )
                )
        symbols_for_technical = [i.yahoo_symbol for i in instruments if i.yahoo_symbol]
        instruments_for_dcf = instruments
    else:
        # Query holdings with instrument data in the same session
        holdings_result = await session.execute(
            select(HoldingDaily)
            .join(Instrument)
            .filter(HoldingDaily.date == latest_date)
            .order_by(Instrument.name)
            .options(
                selectinload(HoldingDaily.instrument)
                .selectinload(Instrument.yahoo)
                .options(*_UNUSED_YAHOO_BLOBS)
            )
        )
        holdings = holdings_result.scalars().all()
        # Get currency rates
        currency_rates = await get_rates(session)
        total_portfolio_value = 0.0
        for holding in holdings:
            market_value_native = holding.quantity * holding.current_price
            market_value_gbp = market_value_native * currency_rates[holding.instrument.currency]
            total_portfolio_value += market_value_gbp
        items = holdings
        symbols_for_technical = [h.instrument.yahoo_symbol for h in holdings if h.instrument.yahoo_symbol]
        instruments_for_dcf = [h.instrument for h in holdings]

    # Calculate technical indicators using centralized function
    currencies_for_technical = {i.yahoo_symbol: i.currency for i in instruments_for_dcf if i.yahoo_symbol}
    rsi_data, technical_data = await calculate_technical_indicators_for_symbols(
        symbols_for_technical, session, currencies=currencies_for_technical
    )
    effective_betas = await get_effective_betas(instruments_for_dcf, session)
    dcf_analyses = await get_dcf_analyses(instruments_for_dcf, session, effective_betas=effective_betas)
    # Key off the instruments the analyses were computed from: symbols_for_technical
    # is filtered, so zipping against it would shift every result onto the next stock.
    dcf_analyses_dict = {
        inst.yahoo_symbol: analysis
        for inst, analysis in zip(instruments_for_dcf, dcf_analyses)
        if inst.yahoo_symbol
    }

    instrument_ids = [h.instrument.id for h in items]
    form13f = await _get_form13f_for_instruments(session, instrument_ids)
    earnings_signals = await _get_earnings_signals(session, items)
    insider_signals = await _get_insider_signals(session, instrument_ids)
    position_reviews = await _get_position_reviews(session, instrument_ids)

    portfolio_data: list[dict[str, Any]] = []
    thesis_by_symbol: dict[str, InstrumentThesisTypedDict] = {}
    for holding in items:
        market_value_native = holding.quantity * holding.current_price
        market_value_gbp = market_value_native * currency_rates[holding.instrument.currency]
        portfolio_pct = (market_value_gbp / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        dcf_analysis = dcf_analyses_dict.get(holding.instrument.yahoo_symbol) or {
            "price": None,
            "low": None,
            "high": None,
            "implied_growth": None,
            "implied_growth_status": "unavailable",
        }
        dcf_price = dcf_analysis["price"]
        dcf_low = dcf_analysis["low"]
        dcf_high = dcf_analysis["high"]

        # Yahoo Finance info for this instrument
        yh = holding.instrument.yahoo
        info = (yh.info or {}) if yh else {}
        f_score_result = get_piotroski_f_score(yh.cashflow, yh.balance_sheet, yh.income_stmt) if yh else None
        trends = calculate_historical_trends(holding)
        profit = holding.ppl if holding.ppl is not None else 0
        cost_basis = (market_value_gbp - holding.ppl) if holding.ppl is not None else 0
        return_pct = (holding.ppl / cost_basis * 100.0) if holding.ppl is not None and cost_basis > 0 else 0.0

        if holding.instrument.thesis:
            thesis_by_symbol[holding.instrument.yahoo_symbol] = holding.instrument.thesis

        # marketCap is quoted in `currency` but the statements are in
        # `financialCurrency`; mixing the two is wrong for ~22% of the universe.
        fx = statement_to_quote_factor(info, currency_rates)
        market_cap = info.get("marketCap")
        free_cashflow = info.get("freeCashflow")
        total_revenue = info.get("totalRevenue")
        ebitda = info.get("ebitda")
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        fcf_quote = free_cashflow * fx if (free_cashflow is not None and fx) else None
        revenue_quote = total_revenue * fx if (total_revenue is not None and fx) else None

        # Quarterly statements are the only multi-period fundamentals available.
        # roic_ttm sums four quarters, so it is comparable with `roic` above.
        quarterly_income_stmt = yh.quarterly_income_stmt if yh else None
        roic_ttm_series = (
            get_roic_ttm_series(yh.quarterly_balance_sheet, quarterly_income_stmt) if yh else []
        )

        portfolio_data.append(
            {
                "t212_code": holding.instrument.t212_code,
                "name": holding.instrument.name,
                "yahoo_symbol": holding.instrument.yahoo_symbol,
                "currency": holding.instrument.currency,
                "sector": info.get("sector"),
                "country": info.get("country"),
                "tags": holding.instrument.tags or [],
                "quantity": holding.quantity,
                "avg_price": holding.avg_price,
                "current_price": holding.current_price,
                "analyst_price_targets": holding.instrument.yahoo.analyst_price_targets
                if holding.instrument.yahoo
                else None,
                "dcf_price": dcf_price,
                "dcf_diff": (dcf_price / holding.current_price - 1) if (dcf_price and holding.current_price) else None,
                "dcf_low": dcf_low,
                "dcf_high": dcf_high,
                "dcf_implied_growth": dcf_analysis["implied_growth"],
                "dcf_implied_growth_status": dcf_analysis["implied_growth_status"],
                "ppl": holding.ppl,
                "fx_ppl": holding.fx_ppl,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe_ratio": info.get("forwardPE"),
                # Yahoo's own priceToSalesTrailing12Months carries the currency
                # mismatch, so recompute it rather than reading the field.
                "ps_ratio": market_cap / revenue_quote if (market_cap and revenue_quote) else None,
                "beta": info.get("beta"),
                "date": holding.date.isoformat() if hasattr(holding.date, "isoformat") else str(holding.date),
                "market_value": market_value_gbp,  # Now in GBP
                "profit": profit,  # Total profit (same as terminal - ppl already includes FX)
                "return_pct": return_pct,
                "portfolio_pct": portfolio_pct,
                "dividend_yield": info.get("dividendYield"),
                "business_summary": info.get("longBusinessSummary"),
                "prediction": (info["targetMedianPrice"] / holding.current_price - 1) * 100.0
                if info.get("targetMedianPrice")
                else None,
                "institutional_ownership": round(info.get("heldPercentInstitutions") * 100.0)
                if (info.get("heldPercentInstitutions") is not None)
                else None,
                # Full precision throughout — these feed screener evaluation.
                "peg_ratio": info["trailingPegRatio"] if info.get("trailingPegRatio") is not None else None,
                "profit_margins": info["profitMargins"] * 100.0 if info.get("profitMargins") is not None else None,
                "gross_margin": info["grossMargins"] * 100.0 if info.get("grossMargins") is not None else None,
                "operating_margin": info["operatingMargins"] * 100.0
                if info.get("operatingMargins") is not None
                else None,
                "revenue_growth": info["revenueGrowth"] * 100.0 if info.get("revenueGrowth") is not None else None,
                "total_revenue": revenue_quote,
                "return_on_assets": info["returnOnAssets"] * 100.0 if info.get("returnOnAssets") is not None else None,
                "return_on_equity": info["returnOnEquity"] * 100.0 if info.get("returnOnEquity") is not None else None,
                "roic": get_roic(info, yh.balance_sheet, yh.income_stmt) if yh else None,
                "roic_3y_min": min(get_roic_history(yh.balance_sheet, yh.income_stmt, periods=3), default=None)
                if yh
                else None,
                "roic_ttm": roic_ttm_series[0] if roic_ttm_series else None,
                "roic_ttm_trend": trend(roic_ttm_series),
                "gross_margin_trend_4q": gross_margin_trend(quarterly_income_stmt),
                "operating_margin_trend_4q": operating_margin_trend(quarterly_income_stmt),
                "revenue_growth_4q_avg": revenue_growth_yoy_avg(quarterly_income_stmt),
                "eps_revision_ratio_30d": eps_revision_ratio(yh.estimates) if yh else None,
                "eps_next_q_growth": eps_next_quarter_growth(yh.estimates) if yh else None,
                "f_score": f_score_result["score"] if f_score_result else None,
                "f_score_details": f_score_result["details"] if f_score_result else None,
                "free_cashflow_yield": fcf_quote / market_cap * 100 if (fcf_quote is not None and market_cap) else None,
                # Both terms are in the reporting currency, so this is FX-immune.
                "fcf_margin": free_cashflow / total_revenue * 100
                if (free_cashflow is not None and total_revenue)
                else None,
                "quickRatio": info.get("quickRatio") if info.get("sector") != "Financial Services" else None,
                "debtToEquity": info.get("debtToEquity"),
                "net_cash": total_cash > total_debt if (total_cash is not None and total_debt is not None) else None,
                "net_debt_to_ebitda": (total_debt - total_cash) / ebitda
                if (total_debt is not None and total_cash is not None and ebitda and ebitda > 0)
                else None,
                "recommendation_mean": round(info["recommendationMean"], 2) if info.get("recommendationMean") else None,
                "recommendation_key": info.get("recommendationKey"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                "fifty_two_week_high_distance": round(info["fiftyTwoWeekHighChangePercent"] * 100, 2)
                if info.get("fiftyTwoWeekHighChangePercent") is not None
                else None,  # Distance from 52-week high (negative = below high, 0 = at the high)
                "fifty_two_week_change": round(info.get("52WeekChange", 0) * 100)
                if info.get("52WeekChange") is not None
                else None,
                "short_percent_of_float": info["shortPercentOfFloat"] * 100
                if info.get("shortPercentOfFloat")
                else None,
                "rsi": rsi_data.get(holding.instrument.yahoo_symbol),
                "rule_of_40_score": (info.get("revenueGrowth", 0) * 100) + (info.get("profitMargins", 0) * 100)
                if (info.get("revenueGrowth") is not None and info.get("profitMargins") is not None)
                else None,  # Keep full precision
                # Technical indicators calculated from price history
                "sma_20": technical_data.get(holding.instrument.yahoo_symbol, {}).get("sma_20"),
                "sma_50": technical_data.get(holding.instrument.yahoo_symbol, {}).get("sma_50"),
                "sma_200": technical_data.get(holding.instrument.yahoo_symbol, {}).get("sma_200"),
                "rs_6m_vs_spy": technical_data.get(holding.instrument.yahoo_symbol, {}).get("rs_6m_vs_spy"),
                "gc_days_since": technical_data.get(holding.instrument.yahoo_symbol, {}).get("gc_days_since"),
                "gc_within_sma50_frac": technical_data.get(holding.instrument.yahoo_symbol, {}).get(
                    "gc_within_sma50_frac"
                ),
                "bb_width_20": technical_data.get(holding.instrument.yahoo_symbol, {}).get("bb_width_20"),
                "bb_width_20_p30_6m": technical_data.get(holding.instrument.yahoo_symbol, {}).get("bb_width_20_p30_6m"),
                "vol20_lt_vol60": technical_data.get(holding.instrument.yahoo_symbol, {}).get("vol20_lt_vol60"),
                "volume_ratio": technical_data.get(holding.instrument.yahoo_symbol, {}).get("volume_ratio"),
                # Supplies avg_pe alongside the PE/recommendation trends
                **trends,
                "quote_type": info.get("quoteType", "Unknown"),
                "passedScreeners": [],  # will be populated below
                "screener_score": 0,  # will be populated below
                "screener_score_max": None,  # will be populated below
                "form13f_score": form13f.get(holding.instrument.id, {}).get("score"),
                "form13f_holders": form13f.get(holding.instrument.id, {}).get("holders", []),
                **(
                    earnings_signals.get(holding.instrument.id)
                    or {
                        "earnings_signal": None,
                        "earnings_conviction": None,
                        "since_earnings_pct": None,
                        "earnings_announcement_date": None,
                        "earnings_report_date": None,
                    }
                ),
                **(
                    insider_signals.get(holding.instrument.id)
                    or {
                        "insider_buy_count_90d": None,
                        "insider_sell_count_90d": None,
                        "insider_net_value_90d": None,
                    }
                ),
                **(
                    position_reviews.get(holding.instrument.id)
                    or {
                        "review_action": None,
                        "review_thesis_status": None,
                        "review_signal_strength": None,
                        "review_confidence": None,
                        "review_headline": None,
                        "reviewed_at": None,
                    }
                ),
            }
        )

    # Calculate screener results for each holding
    calculate_screener_results(portfolio_data)

    rule_recommendations = attach_thesis_rule_eval(portfolio_data, thesis_by_symbol)

    last_updated = latest_date.isoformat() if latest_date else datetime.now(TIMEZONE).date().isoformat()
    return {
        "holdings": portfolio_data,
        "total_holdings": len(portfolio_data),
        "quick_ratio_thresholds": QUICK_RATIO_THRESHOLDS,
        "last_updated": last_updated,
        "rule_recommendations": rule_recommendations,
    }


@router.get("/api/portfolio/summary")
async def get_portfolio_summary(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get portfolio summary statistics including total value, profit, and win rate."""
    # Get the latest portfolio snapshot
    result = await session.execute(select(PortfolioDaily).order_by(PortfolioDaily.date.desc()))
    latest_snapshot = result.scalars().first()

    if not latest_snapshot:
        return {"error": "No portfolio data available"}

    # Max drawdown computed on a TWRR-derived wealth index so deposits and
    # withdrawals don't artificially shrink (or deepen) drawdowns.
    pf_twrr_res = await session.execute(select(PortfolioDaily.date, PortfolioDaily.twrr).order_by(PortfolioDaily.date))
    pf_twrr_rows = pf_twrr_res.all()
    pf_wealth = _twrr_wealth_index([(d, t) for d, t in pf_twrr_rows])
    dd = compute_max_drawdown(pf_wealth)

    # Use the pre-computed trailing 365-day Jensen's Alpha stored by update_returns.py.
    alpha = latest_snapshot.jensens_alpha

    # Get holdings for the same date to calculate win rate
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False), raise_for_status=True
    ) as aiohttp_session:
        (
            holdings_result,
            vix_result,
            fear_greed_index,
            yield_spread,
            buffett_indicator,
            market_breadth_indicator,
            sp500_above_sma200,
            consumer_sentiment,
            real_yield_10y,
            hy_oas,
        ) = await asyncio.gather(
            session.execute(select(HoldingDaily).filter(HoldingDaily.date == latest_snapshot.date)),
            session.execute(
                select(PricesDaily.close_price)
                .where(
                    PricesDaily.symbol == VIX,
                )
                .order_by(PricesDaily.date.desc())
                .limit(1)
            ),
            gen_fear_greed_index(aiohttp_session),
            get_yield_spread(aiohttp_session),
            gen_buffett_indicator(aiohttp_session),
            gen_market_breadth_indicator(session),
            gen_sp500_above_sma200(session),
            get_consumer_sentiment(aiohttp_session),
            get_real_yield_10y(aiohttp_session),
            get_hy_oas(aiohttp_session),
        )

    market_breadth_indicator = await compute_live_rolling_breadth(session, market_breadth_indicator)

    holdings = holdings_result.scalars().all()

    profitable_count = 0
    losing_count = 0

    for holding in holdings:
        profit = holding.ppl  # Total profit (same as terminal - ppl already includes FX)
        if profit > 0:
            profitable_count += 1
        else:
            losing_count += 1

    return {
        "total_value": latest_snapshot.value,
        "total_profit": latest_snapshot.unrealised_profit,
        "total_return_pct": latest_snapshot.return_pct,
        "total_holdings": len(holdings),
        "profitable_holdings": profitable_count,
        "losing_holdings": losing_count,
        "beta": latest_snapshot.beta,
        "sortino_ratio": latest_snapshot.sortino_ratio,
        "sharpe_ratio": latest_snapshot.sharpe_ratio,
        "mwrr": latest_snapshot.mwrr,
        "twrr": latest_snapshot.twrr,
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "drawdown_duration_days": dd["drawdown_duration_days"],
        "market_breadth_indicator": market_breadth_indicator,
        "sp500_above_sma200": sp500_above_sma200,
        "last_updated": latest_snapshot.updated_at.isoformat(),
        "vix": vix_result.scalar(),
        "fear_greed_index": fear_greed_index,
        "yield_spread": yield_spread,
        "buffett_indicator": buffett_indicator,
        "consumer_sentiment": consumer_sentiment,
        "real_yield_10y": real_yield_10y,
        "hy_oas": hy_oas,
        "alpha": alpha,
    }


@router.get("/api/portfolio/allocations")
async def get_portfolio_allocations(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get portfolio allocations by sector and country."""
    # Get the latest portfolio snapshot
    result = await session.execute(select(PortfolioDaily).order_by(PortfolioDaily.date.desc()))
    latest_snapshot = result.scalars().first()

    if not latest_snapshot:
        return {"error": "No portfolio data available"}

    return {
        "sector_allocation": latest_snapshot.sector_allocation,
        "country_allocation": latest_snapshot.country_allocation,
        "currency_allocation": latest_snapshot.currency_allocation,
        "etf_equity_split": latest_snapshot.etf_equity_split,
        "total_value": latest_snapshot.value,
        "tag_soft_caps": TAG_CLUSTER_SOFT_CAPS,
    }


@router.get("/api/portfolio/history")
async def get_portfolio_history(
    days: Optional[int] = None, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get portfolio history for the last N days."""
    if days is not None:
        cutoff_date = datetime.now(TIMEZONE).date() - timedelta(days=days)
        query = select(PortfolioDaily).filter(PortfolioDaily.date >= cutoff_date).order_by(PortfolioDaily.date)
    else:
        query = select(PortfolioDaily).order_by(PortfolioDaily.date)

    snap_res = await session.execute(query)
    snapshots = snap_res.scalars().all()

    bench_res = await session.execute(
        select(PricesDaily.date, PRICE_COLUMN, PricesDaily.symbol)
        .where(
            PricesDaily.symbol.in_(BENCHES),
            PricesDaily.date >= snapshots[0].date,  # latest date in the snapshots
        )
        .order_by(PricesDaily.date)
    )
    bench_rows = bench_res.all()
    daily_bench: dict[str, dict[date, float]] = defaultdict(lambda: defaultdict(float))
    bench_prices = defaultdict(list)
    for bench_row in bench_rows:
        daily_bench[bench_row.symbol][bench_row.date] = bench_row.price
        bench_prices[bench_row.symbol].append(bench_row.price)
    benches_base_price = {symbol: bench_price[0] for symbol, bench_price in bench_prices.items()}

    # Zero-index every line at the window start so the chart answers
    # "return within this time window" for portfolio and benchmarks alike.
    #
    # The portfolio line comes from the TWRR wealth index rather than differencing
    # `return_pct`. `return_pct` is lifetime P&L over the cost basis of *currently
    # held* positions, so a deposit inflates its denominator and drags the line
    # down with no price move at all: the April 2026 deposit alone put the 6-month
    # window 11pp below the true time-weighted return, flipping it from +3.1% to
    # -8.0%. Differencing two such values is also arithmetic where the benchmark
    # lines are geometric, so the two were never comparable. `twrr` is annualised
    # from inception, so the index must be built over the full history (see
    # `_twrr_wealth_index`) before being re-based to this window.
    twrr_res = await session.execute(
        select(PortfolioDaily.date, PortfolioDaily.twrr).order_by(PortfolioDaily.date)
    )
    wealth_rows = twrr_res.all()
    wealth_by_date = dict(
        zip(
            (row.date for row in wealth_rows),
            _twrr_wealth_index([(row.date, row.twrr) for row in wealth_rows]),
        )
    )
    window_base = next((wealth_by_date[s.date] for s in snapshots if wealth_by_date.get(s.date)), None)

    bench = [0.0 for _bench_symbol in BENCHES]
    portfolio_return = 0.0

    history_data = []
    for snapshot in snapshots:
        bench = [
            (daily_bench[bench_symbol][snapshot.date] / benches_base_price[bench_symbol] - 1) * 100
            if daily_bench[bench_symbol].get(snapshot.date)
            else bench[i]
            for i, bench_symbol in enumerate(BENCHES)
        ]
        wealth = wealth_by_date.get(snapshot.date)
        if wealth and window_base:
            portfolio_return = (wealth / window_base - 1) * 100
        history_data.append(
            {
                "date": snapshot.date.isoformat(),
                "total_value": snapshot.value,
                "total_profit": snapshot.unrealised_profit,
                "total_return_pct": portfolio_return,
                "country_allocation": snapshot.country_allocation,
                "sector_allocation": snapshot.sector_allocation,
                "currency_allocation": snapshot.currency_allocation,
                "etf_equity_split": snapshot.etf_equity_split,
                "benchmark_return_pct": bench,
            }
        )

    return {"history": history_data, "days": days, "benchmark": BENCHES}


@router.get("/api/portfolio/underwater")
async def get_portfolio_underwater(
    days: Optional[int] = None, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """
    Underwater (drawdown from running peak) series for the portfolio and
    benchmark indices. Every value is <= 0; 0 means at a new all-time high
    within the window.

    The portfolio series uses a TWRR-derived wealth index so deposits and
    withdrawals do not distort the curve (an inflow would otherwise push the
    line back toward 0). Benchmarks use their own price series — they have no
    cash flows by construction.

    days=None returns the full history.
    """
    # Always query full TWRR history so the wealth index is anchored at
    # inception; otherwise the annualisation inversion would reset each window
    # to 1.0 and lose the true starting drawdown state.
    snap_res = await session.execute(select(PortfolioDaily.date, PortfolioDaily.twrr).order_by(PortfolioDaily.date))
    snap_rows = snap_res.all()
    if not snap_rows:
        return {"history": [], "days": days, "benchmark": list(BENCHES)}

    full_wealth = _twrr_wealth_index([(r.date, r.twrr) for r in snap_rows])
    full_portfolio_uw = underwater_series(full_wealth)

    if days is not None:
        cutoff_date = datetime.now(TIMEZONE).date() - timedelta(days=days)
        sliced_data = [(r.date, uw) for r, uw in zip(snap_rows, full_portfolio_uw) if r.date >= cutoff_date]
    else:
        sliced_data = [(r.date, uw) for r, uw in zip(snap_rows, full_portfolio_uw)]

    if not sliced_data:
        return {"history": [], "days": days, "benchmark": list(BENCHES)}

    portfolio_dates = [d for d, _ in sliced_data]
    portfolio_uw = [uw for _, uw in sliced_data]

    # Benchmark underwater is computed over each benchmark's own trading-day
    # series from portfolio inception (so the peak does not reset for shorter windows),
    # then aligned to portfolio dates via forward-fill.
    bench_res = await session.execute(
        select(PricesDaily.date, PRICE_COLUMN, PricesDaily.symbol)
        .where(
            PricesDaily.symbol.in_(BENCHES),
            PricesDaily.date >= snap_rows[0].date,  # Start from inception
            PricesDaily.date <= portfolio_dates[-1],
        )
        .order_by(PricesDaily.date)
    )
    bench_by_symbol: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for row in bench_res.all():
        bench_by_symbol[row.symbol].append((row.date, row.price))

    bench_uw_by_date: dict[str, dict[date, float]] = {}
    for symbol in BENCHES:
        rows = bench_by_symbol.get(symbol, [])
        if not rows:
            bench_uw_by_date[symbol] = {}
            continue
        prices = [p for _, p in rows]
        uw = underwater_series(prices)
        bench_uw_by_date[symbol] = {d: uw_val for (d, _), uw_val in zip(rows, uw) if uw_val is not None}

    last_bench_uw = {symbol: 0.0 for symbol in BENCHES}
    history: list[dict[str, Any]] = []
    for i, pdate in enumerate(portfolio_dates):
        for symbol in BENCHES:
            v = bench_uw_by_date[symbol].get(pdate)
            if v is not None:
                last_bench_uw[symbol] = v
        history.append(
            {
                "date": pdate.isoformat(),
                "portfolio_dd_pct": portfolio_uw[i],
                "benchmark_dd_pct": [last_bench_uw[sym] for sym in BENCHES],
            }
        )

    return {"history": history, "days": days, "benchmark": list(BENCHES)}


@router.get("/api/portfolio/deposits")
async def get_portfolio_deposits(
    days: Optional[int] = None, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """
    DEPOSIT cash flows within the requested window, aggregated by calendar date.

    Used to overlay deposit events on the Total Return chart. Withdrawals are
    not included (this portfolio doesn't produce any meaningful ones), and
    same-day totals below ``_MIN_DEPOSIT_AMOUNT_GBP`` are dropped as admin /
    residual noise.
    """
    q = select(
        func.date(TransactionHistory.timestamp).label("day"),
        func.sum(TransactionHistory.total).label("amount"),
    ).where(TransactionHistory.action == TransactionAction.DEPOSIT)

    if days is not None:
        cutoff = datetime.now(TIMEZONE).date() - timedelta(days=days)
        q = q.where(func.date(TransactionHistory.timestamp) >= cutoff)

    q = q.group_by(func.date(TransactionHistory.timestamp)).order_by(func.date(TransactionHistory.timestamp))

    rows = (await session.execute(q)).all()
    deposits = [
        {"date": r.day.isoformat(), "amount": float(r.amount)}
        for r in rows
        if float(r.amount) >= _MIN_DEPOSIT_AMOUNT_GBP
    ]
    return {"deposits": deposits, "days": days}


@router.get("/api/portfolio/indicators/history")
async def get_portfolio_indicators_history(
    days: int = 365,
    session: AsyncSession = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """
    Historical time-series of portfolio risk/return metrics from PortfolioDaily.
    days=0 returns all available history.

    Selects only the metric columns to avoid loading the JSONB allocation blobs
    (country_allocation, sector_allocation, etc.) that are not needed here.
    """
    cols = (
        PortfolioDaily.date,
        PortfolioDaily.value,
        PortfolioDaily.unrealised_profit,
        PortfolioDaily.realised_profit,
        PortfolioDaily.invested,
        PortfolioDaily.sortino_ratio,
        PortfolioDaily.beta,
        PortfolioDaily.jensens_alpha,
        PortfolioDaily.positions_total,
        PortfolioDaily.positions_winning,
        PortfolioDaily.mwrr,
        PortfolioDaily.twrr,
    )
    if days == 0:
        query = select(*cols).order_by(PortfolioDaily.date)
    else:
        cutoff = datetime.now(TIMEZONE).date() - timedelta(days=days)
        query = select(*cols).where(PortfolioDaily.date >= cutoff).order_by(PortfolioDaily.date)

    result = await session.execute(query)
    rows = result.all()
    return [
        {
            "date": row.date.isoformat(),
            "value": row.value,
            "profit": row.unrealised_profit,
            "return_pct": (
                round((row.unrealised_profit + row.realised_profit) / row.invested * 100, 4) if row.invested else None
            ),
            "sortino_ratio": row.sortino_ratio,
            "beta": row.beta,
            "jensens_alpha": row.jensens_alpha,
            "positions_total": row.positions_total,
            "positions_winning": row.positions_winning,
            "mwrr": row.mwrr,
            "twrr": row.twrr,
        }
        for row in rows
    ]


@router.get("/api/tickers")
async def get_instruments(session: AsyncSession = Depends(get_db_session)) -> dict[str, list[dict[str, Any]]]:
    """Get all instruments in the database for autocomplete."""
    name_col = func.coalesce(Instrument.name, PricesDaily.symbol)
    result = await session.execute(
        select(PricesDaily.symbol, name_col.label("name"))
        .outerjoin(Instrument, and_(Instrument.yahoo_symbol == PricesDaily.symbol, Instrument.yahoo_symbol.isnot(None)))
        .distinct()
        .order_by(name_col)
    )

    instruments = [{"symbol": row.symbol, "name": row.name} for row in result.all()]

    return {"instruments": instruments}
