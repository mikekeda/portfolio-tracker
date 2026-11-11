"""Refresh historical quarterly P/E ratios from public Wisesheets pages."""

import json
import math
from datetime import date, datetime
from time import sleep
from typing import Any

import requests
from bs4 import BeautifulSoup  # type: ignore[import-untyped]
from sqlalchemy import func, or_, select, update
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from config import TIMEZONE, logger
from data import STOCKS_DELISTED, WISESHEETS_NO_PE
from models import HoldingDaily, Instrument, InstrumentYahoo
from scripts.update_data import get_session


def build_url(ticker: str) -> str:
    return f"https://www.wisesheets.io/pe-ratio/{ticker.strip().upper()}"


def parse_pe_data(html: str, ticker: str) -> dict[str, float]:
    """Read the quarterly series the public page uses to render its table."""
    script = BeautifulSoup(html, "html.parser").select_one("script#__NEXT_DATA__")
    if script is None or not script.string:
        raise ValueError("Wisesheets page is missing its embedded data")
    try:
        page = json.loads(script.string)["props"]["pageProps"]
        if page["symbol"].upper() != ticker.strip().upper():
            raise ValueError("Wisesheets returned data for a different ticker")
        quarterly = page["data"]["keyMetricsQuarterly"]
        if not isinstance(quarterly, list):
            raise ValueError("Wisesheets quarterly P/E history is not a list")

        records: dict[str, float] = {}
        for row in quarterly:
            period_end = date.fromisoformat(row["date"]).isoformat()
            ratio = row["peRatio"]
            if ratio is None:
                continue
            if type(ratio) not in (int, float) or not math.isfinite(ratio):
                raise ValueError("Wisesheets returned an invalid quarterly P/E ratio")
            # The public table displays nonpositive P/E as not meaningful.
            if ratio > 0:
                records[period_end] = float(ratio)
        return records
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Wisesheets quarterly P/E payload has an unexpected structure") from exc


def fetch_pe_data(http: requests.Session, ticker: str) -> dict[str, float]:
    response = http.get(build_url(ticker), timeout=(5, 20))
    response.raise_for_status()
    return parse_pe_data(response.text, ticker)


def _quarter(period_end: str) -> tuple[int, int]:
    day = date.fromisoformat(period_end)
    return day.year, (day.month - 1) // 3


def merge_pe_history(cached: dict[str, Any], latest: dict[str, float], today: date) -> dict[str, Any]:
    """Keep history while replacing old Wisesheets dates for refreshed quarters."""
    today_iso = today.isoformat()
    latest = {day: ratio for day, ratio in latest.items() if day <= today_iso}
    refreshed_quarters = {_quarter(day) for day in latest}
    merged = {}
    for day, entry in cached.items():
        if day > today_iso:
            continue  # Previous estimates are not part of the new history.
        # The old Wisesheets parser assigned calendar dates by row position.
        # Remove its aliases in refreshed quarters, but keep other providers' data.
        is_wisesheets = (
            "stock_price" in entry and entry["stock_price"] is None
            and "ttm_eps" in entry and entry["ttm_eps"] is None
        )
        if is_wisesheets and _quarter(day) in refreshed_quarters:
            continue
        merged[day] = entry
    merged.update({
        day: {"stock_price": None, "ttm_eps": None, "pe_ratio": ratio}
        for day, ratio in latest.items()
    })
    return merged


def update_pe_data(limit: int = 100, onboard_limit: int = 40) -> None:
    """Update PE ratio historical data from Wisesheets for instruments with oldest PE data.

    Two queues: refresh (existing histories, oldest last-PE-key first) and
    onboarding (reserved slots for equities with no PE history, held first).
    """
    # WISESHEETS_NO_PE gaps are often temporary (new IPOs Wisesheets hasn't onboarded);
    # probing once a week auto-detects coverage arriving without nightly error noise.
    skip = STOCKS_DELISTED if datetime.now(TIMEZONE).weekday() == 6 else STOCKS_DELISTED | WISESHEETS_NO_PE

    with get_session() as session:
        # Sort tickers by pe date, old first, get pe for the first N tickers
        last_pe_date_expr = (
            select(func.max(sql_text("key")))
            .select_from(func.jsonb_object_keys(InstrumentYahoo.pes).alias("key"))
            .correlate(InstrumentYahoo)
            .scalar_subquery()
        )

        refresh_query = (
            select(InstrumentYahoo)
            .join(InstrumentYahoo.instrument)
            .where(
                InstrumentYahoo.pes != {},
                Instrument.yahoo_symbol.notin_(skip),
            )
            .order_by(last_pe_date_expr.asc(), InstrumentYahoo.updated_at.nulls_first())
            .limit(limit)
            .options(selectinload(InstrumentYahoo.instrument))
        )
        rows = session.execute(refresh_query).scalars().all()

        # Onboarding gets its own budget: sharing one queue meant seeding only
        # ran with capacity left after every refresh (limit − populated ≈ 1/night,
        # reaching 0 as coverage grows) — NFLX/SPOT/DUOL sat uncovered for months.
        # Dot-free equities only (seeding sticks to Wisesheets' safest coverage);
        # currently-held instruments are seeded first. 40 keeps the nightly run
        # (~20 s/ticker, 150 total) inside the 02:00→03:00 slot before pies sync.
        held_ids = select(HoldingDaily.instrument_id).where(
            HoldingDaily.date == select(func.max(HoldingDaily.date)).scalar_subquery(),
            HoldingDaily.quantity > 0,
        )
        onboarding_query = (
            select(InstrumentYahoo)
            .join(InstrumentYahoo.instrument)
            .where(
                or_(InstrumentYahoo.pes.is_(None), InstrumentYahoo.pes == {}),
                func.coalesce(InstrumentYahoo.info["quoteType"].astext, "EQUITY") == "EQUITY",
                Instrument.yahoo_symbol.is_not(None),
                Instrument.yahoo_symbol.not_like("%.%"),
                Instrument.yahoo_symbol.notin_(skip),
            )
            .order_by(Instrument.id.in_(held_ids).desc(), InstrumentYahoo.updated_at.nulls_first())
            .limit(onboard_limit)
            .options(selectinload(InstrumentYahoo.instrument))
        )
        rows += session.execute(onboarding_query).scalars().all()
        tickers = [row.instrument.yahoo_symbol for row in rows][:10]
        logger.info("Found %s instruments to update: %s,...", len(rows), tickers)

        updated = no_data = failed = 0
        with requests.Session() as http:
            for row in rows:
                ticker = row.instrument.yahoo_symbol
                try:
                    latest = fetch_pe_data(http, ticker)
                    now = datetime.now(TIMEZONE)
                    if not any(day <= now.date().isoformat() for day in latest):
                        no_data += 1
                        logger.info("No usable quarterly P/E for %s; keeping cached history", ticker)
                    else:
                        merged = merge_pe_history(row.pes or {}, latest, now.date())
                        session.execute(
                            update(InstrumentYahoo)
                            .where(InstrumentYahoo.instrument_id == row.instrument_id)
                            .values(pes=merged, updated_at=now)
                        )
                        session.commit()
                        updated += 1
                        logger.info("Updated %s: %d quarterly P/E observations", ticker, len(latest))
                except (requests.RequestException, SQLAlchemyError, ValueError) as exc:
                    failed += 1
                    logger.error("Failed to refresh quarterly P/E for %s: %s", ticker, exc)
                    session.rollback()
                    sleep(20)
                    continue

                sleep(15)

        logger.info("Wisesheets P/E refresh: %d updated, %d without data, %d failed", updated, no_data, failed)
        if failed and not updated:
            raise RuntimeError(f"Wisesheets P/E refresh updated no instruments ({failed} failures)")


if __name__ == "__main__":
    update_pe_data(limit=10)
