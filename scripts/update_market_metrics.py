"""
Market Metrics Update Script (Async)
===================================
Updates daily market metrics (Buffett Indicator, Yield Spread, Fear & Greed, etc.)
into the MarketMetricsDaily table using async logic and shared backend utilities.
"""

import asyncio
from datetime import datetime
from typing import Optional

import ssl

import aiohttp
import certifi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from utils.market_data import (
    gen_buffett_indicator,
    gen_fear_greed_index,
    gen_market_breadth_indicator,
    gen_sp500_above_sma200,
    get_consumer_sentiment,
    get_hy_oas,
    get_real_yield_10y,
    get_yield_spread,
)
from config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, TIMEZONE, VIX, logger
from models import MarketMetricsDaily, PricesDaily


async def get_vix(session: AsyncSession) -> Optional[float]:
    """Get latest VIX close price from DB (async)."""
    result = await session.execute(
        select(PricesDaily.close_price).where(PricesDaily.symbol == VIX).order_by(PricesDaily.date.desc()).limit(1)
    )
    return result.scalar()


async def update_market_metrics():
    """Update all market metrics asynchronously."""
    logger.info("Updating market metrics (Async)...")

    # Setup Async Database Connection
    DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_async_engine(DB_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    try:
        async with AsyncSessionLocal() as db_session:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            async with aiohttp.ClientSession(connector=connector) as http_session:
                today = datetime.now(TIMEZONE).date()

                # Run independent tasks concurrently
                # Note: breadth and sma200 need db_session, others need http_session
                # VIX needs db_session

                # 1. Fetch external API data concurrently
                logger.info("Fetching external data...")
                buffett, yield_spread, fear_greed_data, consumer_sentiment, real_yield_10y, hy_oas = await asyncio.gather(
                    gen_buffett_indicator(http_session),
                    get_yield_spread(http_session),
                    gen_fear_greed_index(http_session),
                    get_consumer_sentiment(http_session),
                    get_real_yield_10y(http_session),
                    get_hy_oas(http_session),
                )

                # 2. Fetch/Calculate DB-dependent metrics
                # We run these sequentially because they share the same db_session,
                # and SQLAlchemy AsyncSession is not concurrency-safe for parallel operations.
                logger.info("Calculating DB metrics...")
                vix = await get_vix(db_session)
                breadth = await gen_market_breadth_indicator(db_session)
                sma200_pct = await gen_sp500_above_sma200(db_session)

                # Extract value from Fear & Greed result
                fear_greed = fear_greed_data["value"] if fear_greed_data else None

                logger.info(
                    "Metrics: Buffett=%s, Yield=%s, F&G=%s, VIX=%s, Breadth=%s, SMA200%%=%s, CS=%s, Real10Y=%s, HY OAS=%s",
                    buffett,
                    yield_spread,
                    fear_greed,
                    vix,
                    breadth,
                    sma200_pct,
                    consumer_sentiment,
                    real_yield_10y,
                    hy_oas,
                )

                # Upsert into DB
                result = await db_session.execute(select(MarketMetricsDaily).filter(MarketMetricsDaily.date == today))
                metric = result.scalars().first()

                if not metric:
                    metric = MarketMetricsDaily(date=today)
                    db_session.add(metric)

                metric.buffett_indicator = buffett
                metric.yield_spread = yield_spread
                metric.fear_greed_index = fear_greed
                metric.vix = vix
                metric.market_breadth_indicator = breadth
                metric.sp500_above_sma200 = sma200_pct
                metric.consumer_sentiment = consumer_sentiment
                metric.real_yield_10y = real_yield_10y
                metric.hy_oas = hy_oas
                metric.updated_at = datetime.now(TIMEZONE).replace(tzinfo=None)

                await db_session.commit()
                logger.info("Market metrics updated successfully.")
    finally:
        await engine.dispose()


def main():
    """Entry point for the script."""
    asyncio.run(update_market_metrics())


if __name__ == "__main__":
    main()
