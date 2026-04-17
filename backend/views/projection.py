"""
Projection endpoint — feeds the /projection page with assumption inputs.

Returns the values that would otherwise be guessed on the frontend:
  * portfolio starting value + annualised TWRR (as of the latest snapshot)
  * track-record length in years
  * UK CPI trailing 12m and trailing 10y CAGR (from FRED)
  * MSCI-World-style long-run benchmarks (static constants, documented inline)

Monte Carlo itself runs client-side (see frontend/src/components/Projection.js);
this endpoint only returns the inputs.
"""

from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app import get_db_session
from backend.utils.market_data import gen_fred_latest
from models import PortfolioDaily

router = APIRouter()

# MSCI World long-run reference figures (1970–present, rough long-term expectations).
# Deliberately modest vs historical US-only to avoid over-optimism on the projection page.
# Nominal 8.5% / inflation ~2.5% ≈ real 6.5%. σ ~16% p.a. from monthly returns.
BENCHMARK_NOMINAL_RETURN = 0.085
BENCHMARK_REAL_RETURN = 0.065
BENCHMARK_VOLATILITY = 0.16

# FRED series for UK CPI (All Items, monthly index, seasonally adjusted).
UK_CPI_SERIES = "GBRCPIALLMINMEI"

# We want ~10 years of monthly observations. Pull a bit more to be safe.
CPI_OBS_LIMIT = 130


def _cpi_metrics(obs: Optional[list[dict[str, Any]]]) -> dict[str, Optional[float]]:
    """From a FRED observation list (newest first), derive trailing 12m YoY and
    trailing 10y CAGR. Returns decimals (0.024 = 2.4%)."""
    if not obs or len(obs) < 13:
        return {"cpi_12m": None, "cpi_10y": None}
    # obs is newest-first per gen_fred_latest.
    latest = obs[0]["value"]
    twelve_back = obs[12]["value"]
    cpi_12m = (latest / twelve_back - 1.0) if twelve_back > 0 else None

    # 10y CAGR: use newest and the observation closest to 120 months ago.
    ten_year_idx = min(120, len(obs) - 1)
    old = obs[ten_year_idx]["value"]
    years = ten_year_idx / 12.0
    cpi_10y = ((latest / old) ** (1.0 / years) - 1.0) if old > 0 and years > 0 else None
    return {"cpi_12m": cpi_12m, "cpi_10y": cpi_10y}


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
            select(PortfolioDaily.date, PortfolioDaily.value, PortfolioDaily.twrr)
            .order_by(PortfolioDaily.date.desc())
            .limit(1)
        )
    ).first()

    twrr: Optional[float] = None
    track_record_years: Optional[float] = None
    starting_value: Optional[float] = None
    if earliest_row and latest_row:
        track_record_years = (latest_row.date - earliest_row.date).days / 365.25
        # TWRR on PortfolioDaily is stored as a decimal (annualised) already.
        twrr = latest_row.twrr
        starting_value = latest_row.value

    # CPI from FRED (best-effort — failure is non-fatal, frontend uses the benchmark default).
    async with aiohttp.ClientSession() as http_session:
        obs = await gen_fred_latest(http_session, UK_CPI_SERIES, limit=CPI_OBS_LIMIT)
    cpi = _cpi_metrics(obs)

    return {
        "portfolio": {
            "starting_value": starting_value,
            "twrr": twrr,
            "track_record_years": track_record_years,
        },
        "inflation": {
            "uk_cpi_12m": cpi["cpi_12m"],
            "uk_cpi_10y_avg": cpi["cpi_10y"],
        },
        "benchmark": {
            "nominal_return": BENCHMARK_NOMINAL_RETURN,
            "real_return": BENCHMARK_REAL_RETURN,
            "volatility": BENCHMARK_VOLATILITY,
            "label": "MSCI World long-run (1970–present)",
        },
    }
