"""
Market Metrics Update Script
===========================
Updates daily market metrics (Buffett Indicator, Yield Spread, Fear & Greed, etc.)
into the MarketMetricsDaily table.
"""

from datetime import datetime, timedelta
from typing import Optional

import requests
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session

from config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    FRED_API_KEY,
    TIMEZONE,
    PRICE_FIELD,
    VIX,
    logger,
)
from data import SP500
from models import MarketMetricsDaily, PricesDaily

# Setup Database Connection
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
PRICE_COLUMN = getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price")


def get_fred_latest(series_id: str, limit: int = 1) -> Optional[list[dict]]:
    """Fetch latest N observations for a FRED series (sync)."""
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY is not set")
        return None

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",
    }
    headers = {"User-Agent": UA}

    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params,
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        out = []
        for o in obs:
            v = o.get("value")
            if v is None or v == ".":
                continue
            out.append({"date": o["date"], "value": float(v)})
        return out or None
    except Exception as e:
        logger.warning(f"FRED fetch error for {series_id}: {e}")
        return None


def get_yield_spread() -> Optional[float]:
    """Get 10Y-2Y yield spread via FRED."""
    obs = get_fred_latest("T10Y2Y", limit=1)
    if not obs:
        return None
    return obs[0]["value"]


def get_buffett_indicator() -> Optional[float]:
    """Calculate Buffett Indicator (Market Cap / GDP)."""
    # Get last few quarters to ensure a common date
    num_obs = get_fred_latest("NCBEILQ027S", limit=6)
    gdp_obs = get_fred_latest("GDP", limit=6)

    if not num_obs or not gdp_obs:
        return None

    num_map = {o["date"]: o["value"] for o in num_obs}
    gdp_map = {o["date"]: o["value"] for o in gdp_obs}
    common_dates = sorted(set(num_map.keys()) & set(gdp_map.keys()), reverse=True)

    if not common_dates:
        logger.warning("No common quarter between NCBEILQ027S and GDP")
        return None

    d = common_dates[0]
    numerator_millions = num_map[d]
    denominator_billions = gdp_map[d]

    if denominator_billions == 0:
        return None

    numerator_billions = numerator_millions / 1000.0
    return (numerator_billions / denominator_billions) * 100.0


def get_fear_greed_index() -> Optional[float]:
    """Scrape Fear & Greed Index from CNN."""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {"User-Agent": UA}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()["fear_and_greed"]
        return float(data["score"])
    except Exception as e:
        logger.warning(f"Error fetching Fear & Greed Index: {e}")
        return None


def get_vix(session: Session) -> Optional[float]:
    """Get latest VIX close price from DB."""
    result = session.execute(
        select(PricesDaily.close_price).where(PricesDaily.symbol == VIX).order_by(PricesDaily.date.desc()).limit(1)
    )
    return result.scalar()


def get_market_breadth(session: Session) -> Optional[float]:
    """Calculate Market Breadth (Advance/Decline ratio)."""
    start_date = datetime.now(TIMEZONE).date() - timedelta(days=4)

    prices_result = session.execute(
        select(PricesDaily.symbol, PRICE_COLUMN)
        .where(PricesDaily.symbol.in_(SP500), PricesDaily.date >= start_date)
        .order_by(PricesDaily.date.desc())
    )

    prices = {}
    for row in prices_result:
        if row.symbol not in prices:
            prices[row.symbol] = []
        prices[row.symbol].append(row.price)

    advance = 0
    decline = 0

    for ticker in SP500:
        p_list = prices.get(ticker, [])
        if len(p_list) < 2:
            continue

        if p_list[0] > p_list[1]:
            advance += 1
        elif p_list[0] < p_list[1]:
            decline += 1

    if not SP500:
        return None

    return (advance - decline) / len(SP500)


def get_sp500_above_sma200(session: Session) -> Optional[float]:
    """Calculate % of S&P 500 stocks above SMA200."""
    start_date = datetime.now(TIMEZONE).date() - timedelta(days=400)

    result = session.execute(
        select(PricesDaily.symbol, PricesDaily.date, PRICE_COLUMN)
        .where(PricesDaily.symbol.in_(SP500), PricesDaily.date >= start_date)
        .order_by(PricesDaily.symbol, PricesDaily.date)
    )

    prices_by_symbol = {}
    for row in result:
        if row.symbol not in prices_by_symbol:
            prices_by_symbol[row.symbol] = []
        prices_by_symbol[row.symbol].append(row.price)

    above_count = 0
    total_count = 0

    for symbol in SP500:
        prices = prices_by_symbol.get(symbol, [])
        if len(prices) < 200:
            continue

        current_price = prices[-1]
        sma_200 = sum(prices[-200:]) / 200

        if current_price > sma_200:
            above_count += 1
        total_count += 1

    if total_count == 0:
        return 0.0

    return (above_count / total_count) * 100.0


def update_market_metrics():
    """Update all market metrics."""
    logger.info("Updating market metrics...")

    with SessionLocal() as session:
        today = datetime.now(TIMEZONE).date()

        # Calculate/Fetch metrics
        buffett = get_buffett_indicator()
        yield_spread = get_yield_spread()
        fear_greed = get_fear_greed_index()
        vix = get_vix(session)
        breadth = get_market_breadth(session)
        sma200_pct = get_sp500_above_sma200(session)

        logger.info(
            f"Metrics: Buffett={buffett}, Yield={yield_spread}, F&G={fear_greed}, VIX={vix}, Breadth={breadth}, SMA200%={sma200_pct}"
        )

        # Upsert into DB
        metric = session.query(MarketMetricsDaily).filter(MarketMetricsDaily.date == today).first()

        if not metric:
            metric = MarketMetricsDaily(date=today)
            session.add(metric)

        metric.buffett_indicator = buffett
        metric.yield_spread = yield_spread
        metric.fear_greed_index = fear_greed
        metric.vix = vix
        metric.market_breadth_indicator = breadth
        metric.sp500_above_sma200 = sma200_pct
        metric.updated_at = datetime.now(TIMEZONE)

        session.commit()
        logger.info("Market metrics updated successfully.")


if __name__ == "__main__":
    update_market_metrics()
