import asyncio
import logging
import os

from typing import Optional, Union

import aiohttp

FRED_API_KEY = os.getenv("FRED_API_KEY")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


async def gen_fear_greed_index(session: aiohttp.ClientSession) -> Optional[dict[str, Union[str, float]]]:
    """Scrape Fear & Greed Index from CNN"""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    headers = {"User-Agent": UA}

    try:
        async with session.get(url, headers=headers) as response:
            data = (await response.json())["fear_and_greed"]

            # Extract the index value
            fear_greed_value = data["score"]
            fear_greed_label = data["rating"]  # "Extreme Fear", "Greed", etc.

            return {"value": fear_greed_value, "label": fear_greed_label, "timestamp": data["timestamp"]}
    except Exception as e:
        logging.warning(f"Error fetching Fear & Greed Index: {e}")
        return None


async def gen_fred_latest(
    session: aiohttp.ClientSession, series_id: str, limit: int = 1
) -> Optional[list[dict[str, float]]]:
    """Fetch latest N observations for a FRED series (async). Returns list of dicts [{date, value}], newest first."""
    if not FRED_API_KEY:
        logging.warning("FRED_API_KEY is not set")
        return None

    params: dict[str, str | int] = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",
    }
    headers = {"User-Agent": UA}

    try:
        async with session.get(
            "https://api.stlouisfed.org/fred/series/observations", params=params, headers=headers
        ) as resp:
            data = await resp.json()
            obs = data.get("observations", [])
            out = []
            for o in obs:
                v = o.get("value")
                if v is None or v == ".":
                    continue
                out.append({"date": o["date"], "value": float(v)})
            return out or None
    except Exception as e:
        logging.warning(f"FRED fetch error for {series_id}: {e}")
        return None


async def get_yield_spread(session: aiohttp.ClientSession) -> Optional[float]:
    """
    Async 10Y–2Y spread via FRED.
    Series: T10Y2Y (percentage points)
    """
    obs = await gen_fred_latest(session, "T10Y2Y", limit=1)
    if not obs:
        return None
    return obs[0]["value"]


async def gen_buffett_indicator(session: aiohttp.ClientSession) -> Optional[float]:
    """
    Buffett-like ratio:
      Numerator (M USD, quarterly): NCBEILQ027S  → convert to B USD
      Denominator (B USD SAAR): GDP
    """
    # Get last few quarters to ensure a common date
    num_obs, gdp_obs = await asyncio.gather(
        gen_fred_latest(session, "NCBEILQ027S", limit=6),
        gen_fred_latest(session, "GDP", limit=6),
    )
    if not num_obs or not gdp_obs:
        return None

    num_map = {o["date"]: o["value"] for o in num_obs}
    gdp_map = {o["date"]: o["value"] for o in gdp_obs}
    common_dates = sorted(set(num_map.keys()) & set(gdp_map.keys()), reverse=True)
    if not common_dates:
        logging.warning("No common quarter between NCBEILQ027S and GDP")
        return None

    d = common_dates[0]
    numerator_millions = num_map[d]  # millions USD
    denominator_billions = gdp_map[d]  # billions USD (SAAR)
    if denominator_billions == 0:
        return None

    numerator_billions = numerator_millions / 1000.0  # <-- convert M → B
    ratio_pct = (numerator_billions / denominator_billions) * 100.0
    return ratio_pct
