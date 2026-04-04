import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional, Union

import aiohttp
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import FRED_API_KEY, logger, TIMEZONE, PRICE_FIELD
from data import SP500
from models import PricesDaily

PRICE_COLUMN = getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


async def gen_sp500_above_sma200(session: AsyncSession) -> float:
    """Calculate percentage of S&P 500 stocks above their 200-day SMA."""
    # We need enough history for 200-day SMA. 300 days (calendar) is ~215 trading days.
    # Let's fetch 400 days to be safe and ensure we have 200 trading days.
    start_date = datetime.now(TIMEZONE).date() - timedelta(days=400)

    # Fetch prices for SP500 symbols
    result = await session.execute(
        select(PricesDaily.symbol, PricesDaily.date, PRICE_COLUMN)
        .where(PricesDaily.symbol.in_(SP500), PricesDaily.date >= start_date)
        .order_by(PricesDaily.symbol, PricesDaily.date)
    )

    prices_by_symbol = defaultdict(list)
    for row in result:
        prices_by_symbol[row.symbol].append(row.price)

    above_count = 0
    total_count = 0
    missing = []

    for symbol in SP500:
        prices = prices_by_symbol.get(symbol, [])
        if not prices:
            missing.append(symbol)
            continue

        # We need at least 200 days of data
        if len(prices) < 200:
            missing.append(symbol)
            continue

        sma_200 = sum(prices[-200:]) / 200

        if sma_200 is not None:
            if prices[-1] > sma_200:
                above_count += 1
            total_count += 1
        else:
            missing.append(symbol)

    if missing:
        logger.info("%d stocks are missing: %s", len(missing), missing)

    if total_count == 0:
        return 0.0

    return (above_count / total_count) * 100.0


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
        logger.warning(f"Error fetching Fear & Greed Index: {e}")
        return None


async def gen_fred_latest(
    session: aiohttp.ClientSession, series_id: str, limit: int = 1
) -> Optional[list[dict[str, float]]]:
    """Fetch latest N observations for a FRED series (async). Returns list of dicts [{date, value}], newest first."""
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY is not set")
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
            if resp.status != 200:
                # Avoid logging full URL/query string to prevent API key leakage.
                logger.warning("FRED fetch error for %s: HTTP %s", series_id, resp.status)
                return None

            # Parse JSON without relying on server Content-Type; FRED occasionally returns
            # incorrect headers via upstream edge/CDN layers.
            data = await resp.json(content_type=None)
            obs = data.get("observations", [])
            out = []
            for o in obs:
                v = o.get("value")
                if v is None or v == ".":
                    continue
                out.append({"date": o["date"], "value": float(v)})
            return out or None
    except aiohttp.ContentTypeError:
        logger.warning("FRED fetch error for %s: unexpected content type", series_id)
        return None
    except Exception as e:
        logger.warning("FRED fetch error for %s: %s", series_id, e.__class__.__name__)
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


async def get_consumer_sentiment(session: aiohttp.ClientSession) -> Optional[float]:
    """
    University of Michigan Consumer Sentiment Index via FRED.
    Series: UMCSENT (index, base 1966:Q1 = 100). Monthly.
    Low values often coincide with fear/recession; high with optimism.
    Useful for long-term: depressed sentiment can signal contrarian entry.
    """
    obs = await gen_fred_latest(session, "UMCSENT", limit=1)
    if not obs:
        return None
    return obs[0]["value"]


async def get_risk_free_rate(session: aiohttp.ClientSession) -> float:
    """
    Fetches the 10-Year Treasury Constant Maturity Rate (DGS10) from FRED.
    Returns the value as a decimal (e.g., 0.045 for 4.5%).
    """
    obs = await gen_fred_latest(session, "DGS10", limit=1)
    if not obs:
        return 0.04  # Fallback to 4% if API fails

    # FRED returns percent (e.g., 4.5), we want decimal (0.045)
    return obs[0]["value"] / 100.0


# FRED series IDs for 10-year government bond yields by instrument currency.
# EUR uses the German Bund as the euro-area benchmark; all others are direct.
_CURRENCY_FRED_SERIES: dict[str, str] = {
    "USD": "DGS10",              # US 10-year T-note (daily)
    "GBP": "IRLTLT01GBM156N",   # UK 10-year Gilt (monthly, OECD)
    "EUR": "IRLTLT01DEM156N",   # Germany 10-year Bund (monthly, OECD)
    "CAD": "IRLTLT01CAM156N",   # Canada 10-year (monthly, OECD)
}

# Conservative fallbacks if FRED is unavailable (approximate 2025/2026 levels)
_RFR_FALLBACKS: dict[str, float] = {
    "USD": 0.043,
    "GBP": 0.045,
    "EUR": 0.026,
    "CAD": 0.032,
}


async def get_risk_free_rates(session: aiohttp.ClientSession) -> dict[str, float]:
    """
    Fetches 10-year government bond yields for USD, GBP, EUR, and CAD
    concurrently from FRED. Returns {currency: rate_as_decimal}.

    Discounting a company's cash flows at the local-currency risk-free rate
    avoids mixing currency environments (e.g., using US T-note yields to
    discount EUR cash flows over-penalises European companies).
    """
    currencies = list(_CURRENCY_FRED_SERIES.keys())
    results = await asyncio.gather(
        *[gen_fred_latest(session, _CURRENCY_FRED_SERIES[c], limit=1) for c in currencies],
        return_exceptions=True,
    )
    rates: dict[str, float] = {}
    for currency, obs in zip(currencies, results):
        if isinstance(obs, Exception) or not obs:
            rates[currency] = _RFR_FALLBACKS[currency]
            logger.warning("FRED risk-free rate fetch failed for %s, using fallback %.3f", currency, _RFR_FALLBACKS[currency])
        else:
            rates[currency] = obs[0]["value"] / 100.0
    return rates


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
        logger.warning("No common quarter between NCBEILQ027S and GDP")
        return None

    d = common_dates[0]
    numerator_millions = num_map[d]  # millions USD
    denominator_billions = gdp_map[d]  # billions USD (SAAR)
    if denominator_billions == 0:
        return None

    numerator_billions = numerator_millions / 1000.0  # <-- convert M → B
    ratio_pct = (numerator_billions / denominator_billions) * 100.0
    return ratio_pct


async def gen_sp500_symbols(session: aiohttp.ClientSession) -> list[str]:
    """Fetch current S&P500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": UA}
    async with session.get(url, headers=headers) as r:
        html = await r.text()
    tables = pd.read_html(StringIO(html))
    df = tables[1]
    results = [str(row["Symbol"]) for _, row in df.iterrows()]

    return results


async def gen_market_breadth_indicator(db_session: AsyncSession, session: aiohttp.ClientSession) -> float:
    """Calculate a simple market breadth indicator for S&P 500 constituents."""
    start_date = datetime.now(TIMEZONE).date() - timedelta(days=4)  # 4 to cover weekends (2 days) + 2 days

    prices_result = await db_session.execute(
        select(PricesDaily.symbol, PRICE_COLUMN)
        .where(PricesDaily.symbol.in_(SP500), PricesDaily.date >= start_date)
        .order_by(PricesDaily.date.desc())
    )
    prices = defaultdict(list)
    for row in prices_result:
        prices[row.symbol].append(row.price)

    advance = 0
    decline = 0
    missing = []

    for ticker in SP500:
        if not prices.get(ticker) or len(prices[ticker]) < 2:
            missing.append(ticker)
            continue

        # prices[ticker][0] is latest (desc order), prices[ticker][1] is previous
        if prices[ticker][0] > prices[ticker][1]:
            advance += 1
        elif prices[ticker][0] < prices[ticker][1]:
            decline += 1

    if missing:
        logger.info("%d stocks are missing: %s", len(missing), missing)

    return (advance - decline) / len(SP500)
