"""
scripts/run_trade_agent.py
==========================
Generate today's trade suggestions (suggest-only — nothing is executed).

Builds the portfolio state from the latest holdings snapshot, runs the rules
strategy through the same data pipeline and constraint layer as the backtest,
and upserts the resulting orders (including vetoed ones, for transparency)
into trade_suggestions. Run from project root:

    python scripts/run_trade_agent.py
"""

import asyncio
from datetime import datetime, timedelta, timezone
from math import isfinite

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.agent.backtest.data import features_for_date, load_market_data, risk_columns, tradable_universe
from backend.agent.constraints import apply_constraints
from backend.agent.rules_strategy import RulesStrategy
from backend.agent.types import AgentLimits, PortfolioState
from backend.app import get_session
from config import logger
from models import HoldingDaily, Instrument, PortfolioDaily, TradeAgentRun, TradeSuggestion

# Price history window: enough for 252d technicals + 504d risk window with slack.
LOOKBACK_DAYS = 900
# Conservative calendar-day bounds allow weekends and ordinary market holidays.
# Incomplete inputs skip the run; we never silently shrink the account or
# forward-fill prices indefinitely to make a recommendation possible.
MAX_INPUT_AGE_DAYS = 4
MAX_VALUATION_PRICE_AGE_DAYS = 5
MAX_VALUATION_DRIFT = 0.10


def _skip(run: dict, reason: str) -> None:
    logger.warning("Trade-agent skipped: %s", reason)
    run.update(status="skipped", reason=reason[:200])


def _positive(value) -> bool:
    return value is not None and isfinite(value) and value > 0


def _valuation_prices(prices, symbols, decision_date):
    """Latest valid GBP valuation per holding, bounded in calendar days.

    This does not manufacture candles or technical signals. Holdings with no
    observed decision-day price remain outside the strategy's tradable universe.
    """
    window = prices.loc[
        (prices.index <= decision_date)
        & (prices.index >= decision_date - timedelta(days=MAX_VALUATION_PRICE_AGE_DAYS))
    ]
    values, carried = {}, {}
    for symbol in symbols:
        if not symbol or symbol not in window:
            continue
        valid = window[symbol].dropna()
        valid = valid[valid.map(_positive)]
        if valid.empty:
            continue
        values[symbol] = float(valid.iloc[-1])
        if valid.index[-1] != decision_date:
            carried[symbol] = valid.index[-1]
    return values, carried


async def run_trade_agent() -> None:
    run = {"strategy": RulesStrategy.name, "ran_at": datetime.now(timezone.utc)}
    try:
        async with get_session() as session:
            await _generate_suggestions(session, AgentLimits.from_config(), run)
            # The run result and its proposals commit together, including zero intents.
            session.add(TradeAgentRun(**run))
    except Exception:
        # The proposal transaction has rolled back. Persist the failed attempt in
        # a separate transaction, without exposing exception contents in the UI.
        run.update(status="failed", reason="Agent run failed; see worker logs")
        try:
            async with get_session() as session:
                session.add(TradeAgentRun(**run))
        except Exception:
            logger.exception("Trade-agent: failed to record unsuccessful run")
        raise


async def _generate_suggestions(session, limits: AgentLimits, run: dict) -> None:
    today = run["ran_at"].astimezone(timezone.utc).date()
    md = await load_market_data(session, today - timedelta(days=LOOKBACK_DAYS))
    if md.gbp_prices.empty:
        run.update(status="skipped", reason="No price data loaded")
        logger.warning("Trade-agent: no price data loaded")
        return
    # During a manual/intraday run, today's sparse bars must not select the
    # portfolio or the strategy universe. Use the latest prior UTC date only;
    # do not fall back to an older, more convenient date when coverage fails.
    dates = md.gbp_prices.index[md.gbp_prices.index < today]
    if dates.empty:
        _skip(run, "No prior-day price data; current-day bars are provisional")
        return
    d = dates[-1]
    run["as_of_date"] = d
    if (today - d).days > MAX_INPUT_AGE_DAYS:
        _skip(run, f"Price data is stale: {d}")
        return

    latest = (await session.execute(select(HoldingDaily.date).order_by(HoldingDaily.date.desc()).limit(1))).scalar()
    holding_rows = (
        await session.execute(
            select(Instrument.id, Instrument.yahoo_symbol, HoldingDaily.quantity)
            .join(Instrument, Instrument.id == HoldingDaily.instrument_id)
            .where(HoldingDaily.date == latest, HoldingDaily.quantity > 0)
        )
    ).all()
    snapshot = (
        await session.execute(select(PortfolioDaily).order_by(PortfolioDaily.date.desc()).limit(1))
    ).scalar()
    if snapshot is None or latest is None or snapshot.date != latest:
        _skip(run, "Holdings and account snapshots are missing or have different dates")
        return
    if not 0 <= (today - latest).days <= MAX_INPUT_AGE_DAYS:
        _skip(run, f"Holdings/account snapshot is stale or future-dated: {latest}")
        return
    if not _positive(snapshot.value) or snapshot.cash is None or not isfinite(snapshot.cash):
        _skip(run, "Account value or cash is invalid")
        return
    cash = snapshot.cash
    prices_today, carried = _valuation_prices(md.gbp_prices, [r.yahoo_symbol for r in holding_rows], d)
    quantities = {}
    missing = []
    for row in holding_rows:
        symbol = row.yahoo_symbol
        if not symbol or not _positive(row.quantity) or not _positive(prices_today.get(symbol)):
            missing.append(symbol or f"instrument {row.id}")
            continue
        quantities[symbol] = quantities.get(symbol, 0.0) + row.quantity
    if missing:
        _skip(
            run,
            f"Incomplete valuation for {d}: {len(missing)}/{len(holding_rows)} holdings lack valid price/FX/quantity: "
            + ", ".join(sorted(missing)),
        )
        return
    values = {s: q * prices_today[s] for s, q in quantities.items()}
    total_value = cash + sum(values.values())
    if not _positive(total_value):
        _skip(run, "Portfolio value is zero or invalid")
        return
    # A complete set of prices can still have wrong currency/share units.
    # This is a sanity bound, not exact reconciliation of asynchronously sampled
    # broker quotes and the prior day's adjusted closes.
    if abs(total_value / snapshot.value - 1) > MAX_VALUATION_DRIFT:
        _skip(run, f"Valuation differs from broker snapshot by more than {MAX_VALUATION_DRIFT:.0%}")
        return
    if carried:
        details = ", ".join(f"{symbol} ({stamp})" for symbol, stamp in sorted(carried.items()))
        reason = f"Carried valuation prices for {len(carried)} holding(s): {details}"
        logger.warning("Trade-agent: %s", reason)
        run["reason"] = reason[:200]
    weights = {s: v / total_value for s, v in values.items()}

    universe = tradable_universe(md, d)
    features = features_for_date(md, d, universe, fundamentals_as_of=today).join(risk_columns(md, d, weights))
    state = PortfolioState(
        date=d,
        total_value_gbp=total_value,
        cash_gbp=cash,
        weights=weights,
        quantities=quantities,
        currencies=md.currencies,
        tags=md.tags,
        etf_symbols=md.etf_symbols,
    )

    strategy = RulesStrategy(limits)
    intents = strategy.propose(d, features, state)
    run["intent_count"] = len(intents)
    # A rerun replaces this batch: drop rows the new run no longer proposes,
    # but keep anything the user already accepted or dismissed.
    await session.execute(
        delete(TradeSuggestion).where(
            TradeSuggestion.date == d,
            TradeSuggestion.strategy == strategy.name,
            TradeSuggestion.status == "proposed",
        )
    )
    if not intents:
        run.update(status="success", order_count=0, executable_count=0)
        logger.info("Trade-agent: no intents for %s — nothing to suggest", d)
        return
    decision_prices = {s: float(md.gbp_prices.loc[d, s]) for s in {i.symbol for i in intents}}
    orders = apply_constraints(intents, state, decision_prices, limits)
    run.update(order_count=len(orders), executable_count=sum(1 for o in orders if o.executable))

    id_by_symbol = dict((await session.execute(select(Instrument.yahoo_symbol, Instrument.id))).all())
    rows = [
        {
            "date": d,
            "instrument_id": id_by_symbol[o.symbol],
            "strategy": strategy.name,
            "action": o.action,
            "quantity": o.quantity,
            "value_gbp": o.value_gbp,
            "weight_before": o.weight_before,
            "weight_after": o.weight_after,
            "score": o.score,
            "fee_gbp": o.fee_gbp,
            "rationale": o.rationale,
            "constraint_adjustments": list(o.adjustments),
            "status": "proposed",
        }
        for o in orders
    ]
    stmt = pg_insert(TradeSuggestion).values(rows)
    update_cols = {
        c.name: getattr(stmt.excluded, c.name)
        for c in TradeSuggestion.__table__.columns
        if c.name not in ("id", "date", "instrument_id", "strategy", "status", "created_at")
    }
    stmt = stmt.on_conflict_do_update(constraint="uq_suggestion_date_instrument_strategy", set_=update_cols)
    await session.execute(stmt)
    run["status"] = "success"

    executable = sum(1 for o in orders if o.executable)
    logger.info(
        "Trade-agent: %s — %d suggestions saved (%d executable, %d vetoed)",
        d,
        len(orders),
        executable,
        len(orders) - executable,
    )
    for o in orders:
        logger.info(
            "  %s %s £%.0f (score %.2f)%s",
            o.action.upper(),
            o.symbol,
            o.value_gbp,
            o.score,
            f" [{'; '.join(o.adjustments)}]" if o.adjustments else "",
        )


def main() -> None:
    asyncio.run(run_trade_agent())


if __name__ == "__main__":
    main()
