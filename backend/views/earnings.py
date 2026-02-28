from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app import get_db_session
from config import TIMEZONE
from models import EarningsReport, HoldingDaily, Instrument

router = APIRouter()


@router.get("/api/earnings/calendar")
async def get_earnings_calendar(
    portfolio_only: bool = False,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """
    Return earnings events for all (or portfolio-only) instruments.
    Sources:
      - Past: InstrumentYahoo.earnings JSONB keys (EPS data from Yahoo)
      - Future: InstrumentYahoo.info.earningsTimestamp
      - Enriched with has_report flag from EarningsReport table
    """
    today = datetime.now(TIMEZONE).date()
    window_start = today - timedelta(days=365)
    window_end = today + timedelta(days=365)

    # Load all instruments with their Yahoo data
    instruments_q = (
        select(Instrument)
        .where(Instrument.yahoo_symbol.isnot(None))
        .options(selectinload(Instrument.yahoo))
    )
    instruments = list((await session.execute(instruments_q)).scalars().all())

    if portfolio_only:
        # Keep only instruments present in the latest holdings snapshot
        latest_date_result = await session.execute(select(func.max(HoldingDaily.date)))
        latest_date = latest_date_result.scalar()
        if latest_date:
            holdings_result = await session.execute(
                select(HoldingDaily.instrument_id)
                .where(HoldingDaily.date == latest_date)
                .where(HoldingDaily.quantity > 0)
            )
            held_ids = {row[0] for row in holdings_result.all()}
            instruments = [i for i in instruments if i.id in held_ids]

    # Build a lookup: instrument_id -> set of dates that have a full EarningsReport
    report_dates_result = await session.execute(
        select(EarningsReport.instrument_id, EarningsReport.date)
    )
    report_dates: dict[int, set] = {}
    for inst_id, rdate in report_dates_result.all():
        report_dates.setdefault(inst_id, set()).add(rdate)

    events: list[dict] = []

    for inst in instruments:
        yh = inst.yahoo
        if not yh:
            continue

        symbol = inst.yahoo_symbol
        name = inst.name

        # ── Past events from Yahoo earnings JSONB ─────────────────────────
        yahoo_earnings: dict = yh.earnings or {}
        for date_str, eps_data in yahoo_earnings.items():
            try:
                event_date = date.fromisoformat(date_str[:10])
            except (ValueError, TypeError):
                continue
            if not (window_start <= event_date <= today):
                continue

            has_report = event_date in report_dates.get(inst.id, set())
            eps_est = eps_data.get("EPS Estimate") if isinstance(eps_data, dict) else None
            eps_act = eps_data.get("Reported EPS") if isinstance(eps_data, dict) else None
            surprise = eps_data.get("Surprise(%)") if isinstance(eps_data, dict) else None

            events.append({
                "date": event_date.isoformat(),
                "symbol": symbol,
                "name": name,
                "type": "past",
                "has_report": has_report,
                "eps_estimate": float(eps_est) if eps_est is not None else None,
                "eps_actual": float(eps_act) if eps_act is not None else None,
                "surprise_pct": float(surprise) if surprise is not None else None,
            })

        # ── Upcoming event from Yahoo info.earningsTimestamp ──────────────
        ts = (yh.info or {}).get("earningsTimestamp")
        if ts:
            try:
                upcoming_date = datetime.fromtimestamp(int(ts), tz=TIMEZONE).date()
            except (ValueError, OSError):
                upcoming_date = None
            if upcoming_date and today < upcoming_date <= window_end:
                events.append({
                    "date": upcoming_date.isoformat(),
                    "symbol": symbol,
                    "name": name,
                    "type": "upcoming",
                    "has_report": False,
                    "eps_estimate": None,
                    "eps_actual": None,
                    "surprise_pct": None,
                })

    # Deduplicate: if a past Yahoo event and an upcoming event share same symbol+date, keep past
    seen: dict[tuple, dict] = {}
    for ev in events:
        key = (ev["symbol"], ev["date"])
        existing = seen.get(key)
        if existing is None or ev["type"] == "past":
            seen[key] = ev
    events = sorted(seen.values(), key=lambda e: e["date"])

    return {"events": events}


@router.get("/api/earnings-report/{symbol}/{report_date}", response_class=HTMLResponse)
async def get_earnings_report_html(symbol: str, report_date: str) -> HTMLResponse:
    """
    Get the full HTML earnings report file for a given symbol and date.
    For local dev server only - production should use nginx to serve static files directly.
    """
    project_root = Path(__file__).parent.parent.parent
    filings_dir = project_root / "data" / "filings" / symbol

    if not filings_dir.exists():
        raise HTTPException(status_code=404, detail=f"Report directory not found for {symbol}")

    matching_files = sorted(filings_dir.glob(f"{report_date}_*.html"))

    if not matching_files:
        raise HTTPException(status_code=404, detail=f"Report file not found for {symbol} on {report_date}")

    html_content = matching_files[0].read_text(encoding="utf-8")

    return HTMLResponse(content=html_content)
