"""Trade-agent suggestion endpoints (suggest-only — nothing executes trades)."""

from datetime import date, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app import get_db_session
from models import Instrument, TradeSuggestion

router = APIRouter()

VALID_STATUSES = ("proposed", "accepted", "dismissed")


class StatusUpdate(BaseModel):
    status: str


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


@router.get("/api/agent/suggestions")
async def get_suggestions(
    session: AsyncSession = Depends(get_db_session), for_date: Optional[str] = None
) -> dict[str, Any]:
    """Suggestions for one day (default: the most recent day that has any)."""
    if for_date:
        target = date.fromisoformat(for_date)
    else:
        target = (await session.execute(select(func.max(TradeSuggestion.date)))).scalar()
    if target is None:
        return {"date": None, "suggestions": []}

    rows = (
        await session.execute(
            select(TradeSuggestion, Instrument)
            .join(Instrument, Instrument.id == TradeSuggestion.instrument_id)
            .where(TradeSuggestion.date == target)
            .order_by(TradeSuggestion.value_gbp.desc())
        )
    ).all()
    return {"date": target.isoformat(), "suggestions": [_row(s, inst) for s, inst in rows]}


@router.get("/api/agent/suggestions/history")
async def get_suggestions_history(
    session: AsyncSession = Depends(get_db_session), days: int = 30
) -> dict[str, Any]:
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


@router.post("/api/agent/suggestions/{suggestion_id}/status")
async def set_suggestion_status(
    suggestion_id: int, body: StatusUpdate, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Record the user's decision on a suggestion (future ML training label)."""
    if body.status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"status must be one of {VALID_STATUSES}")
    suggestion = (
        await session.execute(select(TradeSuggestion).where(TradeSuggestion.id == suggestion_id))
    ).scalar_one_or_none()
    if suggestion is None:
        raise HTTPException(status_code=404, detail="suggestion not found")
    suggestion.status = body.status
    await session.commit()
    return {"id": suggestion.id, "status": suggestion.status}
