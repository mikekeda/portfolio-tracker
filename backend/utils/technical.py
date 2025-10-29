"""
Technical Analysis Utilities
===========================
Helper functions for technical analysis calculations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import PRICE_FIELD, SPY, TIMEZONE
from models import PricesDaily

PRICE_COLUMN = getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price")

logger = logging.getLogger(__name__)


def calculate_rsi(prices: list[float], period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index) for a series of prices."""
    if len(prices) <= period:
        return 50.0  # Return neutral if not enough data

    # Calculate price changes
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Separate gains and losses
    gains = [max(delta, 0) for delta in deltas]
    losses = [max(-delta, 0) for delta in deltas]

    # Calculate initial average gain and loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Calculate subsequent average gain and loss using Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate RS and RSI
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_sma(prices: list[float], period: int) -> Optional[float]:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def find_golden_cross_in_last_n_days(prices: list[float], n_days: int) -> Optional[int]:
    """
    Calculate days since a moving average cross (e.g., SMA50 vs SMA200) within the last n_days.
    This function detects both golden (50 > 200) and death (50 < 200) crosses.
    Returns the number of days ago the *most recent* cross occurred.
    """
    if len(prices) < 200:  # Need at least 200 days for SMA200
        return None

    # Calculate SMA50 and SMA200 efficiently
    sma_50_values = []
    sma_200_values = []

    # Use rolling window approach for simplicity and clarity
    for i in range(49, len(prices)):
        sma_50_values.append(sum(prices[i - 49 : i + 1]) / 50)

    for i in range(199, len(prices)):
        sma_200_values.append(sum(prices[i - 199 : i + 1]) / 200)

    # Align the shorter SMA50 list with the SMA200 list
    # SMA50 values are available from day 50 onwards
    # SMA200 values are available from day 200 onwards
    # We need to compare them starting from day 200
    offset = 199 - 49
    aligned_sma_50 = sma_50_values[offset:]

    # Ensure we have values to compare
    if not aligned_sma_50 or not sma_200_values:
        return None

    # Look for a cross in the last n_days
    # Start from the most recent day and work backwards
    lookback_period = min(n_days, len(aligned_sma_50), len(sma_200_values))

    for days_back in range(lookback_period):
        idx = len(aligned_sma_50) - 1 - days_back

        if idx > 0:
            current_sma50 = aligned_sma_50[idx]
            current_sma200 = sma_200_values[idx]

            prev_sma50 = aligned_sma_50[idx - 1]
            prev_sma200 = sma_200_values[idx - 1]

            # Check for a cross in either direction
            # Golden cross: (prev50 <= prev200) and (curr50 > curr200)
            # Death cross: (prev50 >= prev200) and (curr50 < curr200)
            if (prev_sma50 <= prev_sma200 and current_sma50 > current_sma200) or (
                prev_sma50 >= prev_sma200 and current_sma50 < current_sma200
            ):
                return days_back

    # No cross found in the last n_days
    return None


def calculate_gc_within_sma50(prices: list[float]) -> Optional[float]:
    """Calculate if current price is within SMA50 range."""
    if len(prices) < 50:
        return None

    current_price = prices[-1]
    sma_50 = calculate_sma(prices, 50)

    if sma_50 is None or sma_50 == 0:
        return None

    return (current_price - sma_50) / sma_50


def calculate_bb_width(prices: list[float], period: int) -> Optional[float]:
    """Calculate Bollinger Band width."""
    if len(prices) < period:
        return None

    sma = calculate_sma(prices, period)
    if sma is None:
        return None

    # Calculate standard deviation
    recent_prices = prices[-period:]
    variance = sum((price - sma) ** 2 for price in recent_prices) / period
    std_dev = variance**0.5

    # BB width = (Upper Band - Lower Band) / Middle Band = (4 * std_dev) / sma
    return (4 * std_dev) / sma if sma != 0 else None


def calculate_bb_width_percentile(
    prices: list[float], period: int, lookback: int, percentile: float
) -> Optional[float]:
    """Calculate BB width percentile over the last lookback days only."""
    if len(prices) < lookback + period:
        return None

    # Calculate BB width for each day in the last lookback days
    bb_widths = []
    for i in range(len(prices) - lookback, len(prices)):
        window = prices[i - period + 1 : i + 1]
        bb_width = calculate_bb_width(window, period)
        if bb_width is not None:
            bb_widths.append(bb_width)

    if not bb_widths:
        return None

    current_bb_width = calculate_bb_width(prices[-period:], period)
    if current_bb_width is None:
        return None

    # Find the rank of the current width among historical widths
    rank = sum(1 for width in bb_widths if width < current_bb_width)
    percentile_rank = rank / len(bb_widths)

    return percentile_rank


async def calculate_volume_ratio_from_db(symbol: str, session: AsyncSession) -> Optional[float]:
    """Calculate volume ratio (today / 20-day average) from database."""
    try:
        # Get recent volume data
        end_date = datetime.now(TIMEZONE).date()

        # Query volume data directly from database
        result = await session.execute(
            select(PricesDaily.volume)
            .filter(
                PricesDaily.symbol == symbol,
                PricesDaily.date <= end_date,
            )
            .order_by(PricesDaily.date.desc())
            .limit(21)
        )
        volumes = result.scalars().all()

        if len(volumes) < 21:  # Need at least 21 days (today + 20 days)
            return None

        today_volume = volumes[0]
        avg_20_volume = sum(volumes[1:21]) / 20

        if avg_20_volume == 0:
            return None

        return today_volume / avg_20_volume

    except Exception as e:
        logger.warning(f"Failed to calculate volume ratio for {symbol}: {e}")
        return None


async def calculate_volume_contraction_from_db(symbol: str, session: AsyncSession) -> Optional[bool]:
    """Calculate if 20-day volume is less than 60-day volume from database."""
    try:
        # Query volume data directly from database
        rows = await session.execute(
            select(PricesDaily.volume)
            .filter(
                PricesDaily.symbol == symbol,
            )
            .order_by(PricesDaily.date.desc())
            .limit(60)
        )
        volumes = rows.scalars().all()

        if len(volumes) < 60:
            return None

        vol_20 = sum(volumes[:20]) / 20  # Most recent 20 days
        vol_60 = sum(volumes) / 60  # All 60 days

        return vol_20 < vol_60

    except Exception as e:
        logger.warning(f"Failed to calculate volume contraction for {symbol}: {e}")
        return None


def calculate_relative_strength_vs_spy(symbol_prices: list[float], spy_prices: list[float]) -> Optional[float]:
    """Calculate 6-month relative strength vs SPY using growth factors."""
    if not spy_prices or len(symbol_prices) < 126 or len(spy_prices) < 126:
        return None
    try:
        # Ensure we are comparing the same time period
        symbol_period = symbol_prices[-126:]
        spy_period = spy_prices[-126:]

        if symbol_period[0] <= 0 or spy_period[0] <= 0:
            return None

        # Calculate 6-month growth factors (126 trading days)
        symbol_growth = symbol_period[-1] / symbol_period[0]
        spy_growth = spy_period[-1] / spy_period[0]

        if spy_growth == 0:
            return None

        # Relative strength = (Stock Growth / SPY Growth - 1) * 100
        relative_strength = (symbol_growth / spy_growth - 1) * 100

        return relative_strength
    except Exception as e:
        logger.warning(f"Failed to calculate relative strength: {e}")
        return None


async def calculate_technical_indicators_for_symbols(
    symbols: list[str], session: AsyncSession
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Calculate technical indicators for a list of symbols using available database data."""
    rsi_data: dict[str, float] = {}
    technical_data: dict[str, dict[str, Any]] = {}

    if not symbols:
        return rsi_data, technical_data

    try:
        # Get price history for all symbols
        price_result = await session.execute(
            select(
                PricesDaily.symbol,
                PRICE_COLUMN,
            )
            .filter(
                PricesDaily.symbol.in_(symbols), PricesDaily.date >= datetime.now(TIMEZONE).date() - timedelta(days=420)
            )
            .order_by(PricesDaily.date)
        )
        price_data = price_result.all()

        price_history: dict[str, list[float]] = {}
        for row in price_data:
            price_history.setdefault(row.symbol, []).append(row.price)

        # Get SPY data
        spy_result = await session.execute(
            select(PRICE_COLUMN)
            .filter(PricesDaily.symbol == SPY, PricesDaily.date >= datetime.now(TIMEZONE).date() - timedelta(days=420))
            .order_by(PricesDaily.date)
        )
        spy_prices = [row.price for row in spy_result.all()]

        # Calculate technical indicators
        for symbol, symbol_prices in price_history.items():
            rsi_data[symbol] = calculate_rsi(symbol_prices)

            # Calculate other indicators
            volume_ratio = await calculate_volume_ratio_from_db(symbol, session)
            vol20_lt_vol60 = await calculate_volume_contraction_from_db(symbol, session)
            rs_6m_vs_spy = calculate_relative_strength_vs_spy(symbol_prices, spy_prices)

            # Use the updated flexible golden cross function
            gc_days_since = find_golden_cross_in_last_n_days(symbol_prices, 60)

            technical_data[symbol] = {
                "sma_20": calculate_sma(symbol_prices, 20),
                "sma_50": calculate_sma(symbol_prices, 50),
                "sma_200": calculate_sma(symbol_prices, 200),
                "rs_6m_vs_spy": rs_6m_vs_spy,
                "gc_days_since": gc_days_since,
                "gc_within_sma50_frac": calculate_gc_within_sma50(symbol_prices),
                "bb_width_20": calculate_bb_width(symbol_prices, 20),
                "bb_width_20_p30_6m": calculate_bb_width_percentile(symbol_prices, 20, 126, 0.30),
                "vol20_lt_vol60": vol20_lt_vol60,
                "volume_ratio": volume_ratio,
            }
    except Exception as e:
        logger.error(f"Failed to calculate technical indicators: {e}", exc_info=True)

    return rsi_data, technical_data
