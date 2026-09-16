"""Trade-agent suggestion endpoints (read-only — the API never mutates suggestions)."""

from datetime import date, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app import get_db_session
from models import Instrument, TradeAgentRun, TradeSuggestion

router = APIRouter()


def _row(s: TradeSuggestion, inst: Instrument) -> dict[str, Any]:
    return {
        "id": s.id,
        "date": s.date.isoformat(),
        "symbol": inst.yahoo_symbol,
        "name": inst.name,
        "strategy": s.strategy,
        "action": s.action,
        "quantity": s.quantity,
        "value_gbp": s.value_gbp,
        "weight_before": s.weight_before,
        "weight_after": s.weight_after,
        "score": s.score,
        "fee_gbp": s.fee_gbp,
        "rationale": s.rationale or {},
        "constraint_adjustments": s.constraint_adjustments or [],
        "status": s.status,
    }


def _run_row(run: TradeAgentRun | None) -> dict | None:
    if run is None:
        return None
    return {
        "as_of_date": run.as_of_date.isoformat() if run.as_of_date else None,
        "strategy": run.strategy,
        "ran_at": run.ran_at.isoformat(),
        "status": run.status,
        "intent_count": run.intent_count,
        "order_count": run.order_count,
        "executable_count": run.executable_count,
        "reason": run.reason,
    }


@router.get("/api/agent/suggestions")
async def get_suggestions(
    session: AsyncSession = Depends(get_db_session), for_date: Optional[date] = None
) -> dict[str, Any]:
    """Latest successful evaluation, including no-action runs; prices do not select the batch."""
    strategy = "rules"
    query = select(TradeAgentRun).where(TradeAgentRun.strategy == strategy)
    if for_date is not None:
        query = query.where(TradeAgentRun.as_of_date == for_date)
    query = query.order_by(TradeAgentRun.ran_at.desc(), TradeAgentRun.id.desc()).limit(1)
    latest_run = (await session.execute(query)).scalar()
    if latest_run is not None and latest_run.status == "success":
        run = latest_run
    else:
        run = (await session.execute(query.where(TradeAgentRun.status == "success"))).scalar()
    target = for_date if for_date is not None else run.as_of_date if run else None
    if target is None:
        # Existing installations have proposals predating the run log.
        target = (
            await session.execute(select(func.max(TradeSuggestion.date)).where(TradeSuggestion.strategy == strategy))
        ).scalar()
    rows = []
    if target is not None:
        rows = (
            await session.execute(
                select(TradeSuggestion, Instrument)
                .join(Instrument, Instrument.id == TradeSuggestion.instrument_id)
                .where(TradeSuggestion.date == target, TradeSuggestion.strategy == strategy)
                .order_by(TradeSuggestion.value_gbp.desc())
            )
        ).all()
    return {
        "date": target.isoformat() if target else None,
        "suggestions": [_row(s, inst) for s, inst in rows],
        "run": _run_row(run),
        "latest_run": _run_row(latest_run),
    }


@router.get("/api/agent/suggestions/history")
async def get_suggestions_history(session: AsyncSession = Depends(get_db_session), days: int = 30) -> dict[str, Any]:
    """Suggestions over the trailing N days, newest first."""
    since = date.today() - timedelta(days=days)
    rows = (
        await session.execute(
            select(TradeSuggestion, Instrument)
            .join(Instrument, Instrument.id == TradeSuggestion.instrument_id)
            .where(TradeSuggestion.date >= since)
            .order_by(TradeSuggestion.date.desc(), TradeSuggestion.value_gbp.desc())
        )
    ).all()
    return {"since": since.isoformat(), "suggestions": [_row(s, inst) for s, inst in rows]}
