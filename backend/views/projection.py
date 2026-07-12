"""
Projection endpoint — feeds the /projection page with assumption inputs.

Returns the values that would otherwise be guessed on the frontend:
  * portfolio starting value + annualised TWRR (as of the latest snapshot)
  * track-record length in years
  * UK CPI trailing 12m and trailing 10y CAGR (from the ONS MM23 dataset)
  * MSCI-World-style long-run benchmarks (static constants, documented inline)

Monte Carlo itself runs client-side (see frontend/src/components/Projection.js);
this endpoint only returns the inputs.
"""

import csv
import ssl
import time
from typing import Any, Optional

import aiohttp
import certifi
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app import get_db_session
from config import (
    DAYS_PER_YEAR,
    PROJECTION_BENCHMARK_NOMINAL_RETURN,
    PROJECTION_BENCHMARK_REAL_RETURN,
    PROJECTION_BENCHMARK_VOLATILITY,
    logger,
)
from models import PortfolioDaily
from utils.market_data import UA

router = APIRouter()

# ONS CPI index (series D7BT, all items, 2015=100), monthly. The former FRED
# source (GBRCPIALLMINMEI) froze at March 2025 when OECD discontinued the MEI
# dataset, so we go straight to the ONS.
ONS_CPI_URL = (
    "https://www.ons.gov.uk/generator?format=csv"
    "&uri=/economy/inflationandpriceindices/timeseries/d7bt/mm23"
)

# CPI is monthly; refetch at most once a day and serve stale data on failure.
CPI_CACHE_TTL_SECONDS = 24 * 3600

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    )
}

_cpi_cache: dict[str, Any] = {"obs": None, "fetched_at": 0.0}

# macOS framework Pythons ship without system CA certs, which makes aiohttp's
# default SSL context reject ons.gov.uk; certifi's bundle works everywhere.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _parse_ons_csv(text: str) -> Optional[list[dict[str, Any]]]:
    """Parse an ONS timeseries CSV into monthly observations, newest first.

    Monthly rows look like ("2026 MAY", "142.4"); header, annual ("2026") and
    quarterly ("2026 Q1") rows are skipped. Dates are returned as ISO strings
    (first of the month).
    """
    out = []
    for row in csv.reader(text.splitlines()):
        if len(row) != 2:
            continue
        parts = row[0].split()
        if len(parts) != 2 or parts[1] not in _MONTHS:
            continue
        try:
            value = float(row[1])
        except ValueError:
            continue
        out.append({"date": f"{parts[0]}-{_MONTHS[parts[1]]:02d}-01", "value": value})
    out.sort(key=lambda o: o["date"], reverse=True)
    return out or None


async def _get_cpi_observations() -> Optional[list[dict[str, Any]]]:
    """UK CPI observations (newest first) with a daily in-process cache.

    A failed refresh falls back to the last good result so a transient ONS
    outage doesn't blank the projection page's inflation inputs.
    """
    now = time.time()
    if _cpi_cache["obs"] is not None and now - _cpi_cache["fetched_at"] < CPI_CACHE_TTL_SECONDS:
        return _cpi_cache["obs"]

    obs = None
    try:
        connector = aiohttp.TCPConnector(ssl=_SSL_CONTEXT)
        async with aiohttp.ClientSession(connector=connector) as http_session:
            async with http_session.get(ONS_CPI_URL, headers={"User-Agent": UA}) as resp:
                if resp.status == 200:
                    obs = _parse_ons_csv(await resp.text())
                else:
                    logger.warning("ONS CPI fetch failed: HTTP %s", resp.status)
    except Exception as e:
        logger.warning("ONS CPI fetch failed: %s", e)

    if obs:
        _cpi_cache["obs"] = obs
        _cpi_cache["fetched_at"] = now
        return obs
    return _cpi_cache["obs"]


def _cpi_metrics(obs: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    """From an observation list (newest first), derive trailing 12m YoY, trailing
    10y CAGR, and the latest observation date. Rates are decimals (0.024 = 2.4%)."""
    if not obs or len(obs) < 13:
        return {"cpi_12m": None, "cpi_10y": None, "as_of": None}
    latest = obs[0]["value"]
    twelve_back = obs[12]["value"]
    cpi_12m = (latest / twelve_back - 1.0) if twelve_back > 0 else None

    # 10y CAGR: use newest and the observation closest to 120 months ago.
    ten_year_idx = min(120, len(obs) - 1)
    old = obs[ten_year_idx]["value"]
    years = ten_year_idx / 12.0
    cpi_10y = ((latest / old) ** (1.0 / years) - 1.0) if old > 0 and years > 0 else None
    return {"cpi_12m": cpi_12m, "cpi_10y": cpi_10y, "as_of": obs[0]["date"]}


@router.get("/api/projection/inputs")
async def get_projection_inputs(
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Assumption inputs for the projection page. All returns/volatility are
    annualized decimals (0.08 = 8%). Fields can be None when data is missing;
    the frontend falls back to benchmark defaults."""
    # Fetch earliest and latest snapshots only — enough to derive starting value,
    # TWRR, and track-record length without reading the full daily series.
    earliest_row = (
        await session.execute(
            select(PortfolioDaily.date).order_by(PortfolioDaily.date).limit(1)
        )
    ).first()
    latest_row = (
        await session.execute(
            select(PortfolioDaily.date, PortfolioDaily.value)
            .order_by(PortfolioDaily.date.desc())
            .limit(1)
        )
    ).first()
    # Intraday snapshots are written before update_returns scores them, so take
    # TWRR from the newest row that has one rather than the newest row outright.
    twrr_row = (
        await session.execute(
            select(PortfolioDaily.twrr)
            .where(PortfolioDaily.twrr.is_not(None))
            .order_by(PortfolioDaily.date.desc())
            .limit(1)
        )
    ).first()

    twrr: Optional[float] = None
    track_record_years: Optional[float] = None
    starting_value: Optional[float] = None
    if earliest_row and latest_row:
        track_record_years = (latest_row.date - earliest_row.date).days / DAYS_PER_YEAR
        starting_value = latest_row.value
    if twrr_row:
        # PortfolioDaily.twrr is stored as a percentage (16.17 = 16.17% annualised,
        # see scripts/update_returns.py). The frontend expects a decimal.
        twrr = twrr_row.twrr / 100.0

    cpi = _cpi_metrics(await _get_cpi_observations())

    return {
        "portfolio": {
            "starting_value": starting_value,
            "twrr": twrr,
            "track_record_years": track_record_years,
        },
        "inflation": {
            "uk_cpi_12m": cpi["cpi_12m"],
            "uk_cpi_10y_avg": cpi["cpi_10y"],
            "uk_cpi_as_of": cpi["as_of"],
        },
        "benchmark": {
            "nominal_return": PROJECTION_BENCHMARK_NOMINAL_RETURN,
            "real_return": PROJECTION_BENCHMARK_REAL_RETURN,
            "volatility": PROJECTION_BENCHMARK_VOLATILITY,
            "label": "MSCI World long-run (1970–present)",
        },
    }
