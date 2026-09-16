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
from datetime import date, datetime, timedelta, timezone

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
    md = await load_market_data(session, date.today() - timedelta(days=LOOKBACK_DAYS))
    if md.gbp_prices.empty:
        run.update(status="skipped", reason="No price data loaded")
        logger.warning("Trade-agent: no price data loaded")
        return
    d = md.gbp_prices.index[-1]
    run["as_of_date"] = d

    latest = (await session.execute(select(HoldingDaily.date).order_by(HoldingDaily.date.desc()).limit(1))).scalar()
    holding_rows = (
        await session.execute(
            select(Instrument.id, Instrument.yahoo_symbol, HoldingDaily.quantity)
            .join(Instrument, Instrument.id == HoldingDaily.instrument_id)
            .where(HoldingDaily.date == latest, HoldingDaily.quantity > 0)
        )
    ).all()
    cash = (
        await session.execute(select(PortfolioDaily.cash).order_by(PortfolioDaily.date.desc()).limit(1))
    ).scalar() or 0.0

    prices_today = md.gbp_prices.loc[d]
    quantities = {r.yahoo_symbol: r.quantity for r in holding_rows if r.yahoo_symbol in md.gbp_prices.columns}
    values = {s: q * prices_today.get(s) for s, q in quantities.items()}
    values = {s: v for s, v in values.items() if v == v}  # drop NaN-priced
    total_value = cash + sum(values.values())
    if total_value <= 0:
        run.update(status="skipped", reason="Portfolio value is zero")
        logger.warning("Trade-agent: portfolio value is zero")
        return
    weights = {s: v / total_value for s, v in values.items()}

    universe = tradable_universe(md, d)
    features = features_for_date(md, d, universe, fundamentals_as_of=date.today()).join(risk_columns(md, d, weights))
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
