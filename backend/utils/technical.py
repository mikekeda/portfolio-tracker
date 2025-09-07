"""
Technical Analysis Utilities
===========================
Helper functions for technical analysis calculations.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from config import BENCH, PRICE_FIELD

logger = logging.getLogger(__name__)


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate RSI (Relative Strength Index) for a series of prices.

    Args:
        prices: List of closing prices
        period: RSI period (default 14)

    Returns:
        RSI value between 0 and 100, or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    # Calculate price changes
    deltas = []
    for i in range(1, len(prices)):
        deltas.append(prices[i] - prices[i-1])

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

    return round(rsi, 2)


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_golden_cross_days(prices: List[float]) -> Optional[int]:
    """Calculate days since golden cross (SMA50 > SMA200) using O(n) algorithm."""
    if len(prices) < 260:  # Need 200 + 60 trading days for proper detection
        return None

    # Calculate SMA50 and SMA200 using O(n) rolling mean with cumulative sums
    sma_50_values = []
    sma_200_values = []

    # Calculate cumulative sums for O(n) rolling mean
    cumsum: List[float] = [0] * (len(prices) + 1)
    for i in range(len(prices)):
        cumsum[i + 1] = cumsum[i] + prices[i]

    # Calculate SMAs starting from day 200
    for i in range(199, len(prices)):
        # Calculate SMA50 (O(1) using cumulative sum)
        if i >= 49:  # Need at least 50 prices for SMA50
            sma_50 = (cumsum[i + 1] - cumsum[i + 1 - 50]) / 50
            sma_50_values.append(sma_50)

        # Calculate SMA200 (O(1) using cumulative sum)
        if i >= 199:  # Need at least 200 prices for SMA200
            sma_200 = (cumsum[i + 1] - cumsum[i + 1 - 200]) / 200
            sma_200_values.append(sma_200)

    # Ensure we have enough SMA values
    if len(sma_50_values) < 60 or len(sma_200_values) < 60:
        return None

    # Look for golden cross in the last 60 days
    # Start from the most recent day and work backwards
    for days_back in range(min(60, len(sma_50_values), len(sma_200_values))):
        current_idx = len(sma_50_values) - 1 - days_back

        if current_idx > 0:  # Need previous day to detect cross
            # Current day values
            current_sma50 = sma_50_values[current_idx]
            current_sma200 = sma_200_values[current_idx]

            # Previous day values
            prev_sma50 = sma_50_values[current_idx - 1]
            prev_sma200 = sma_200_values[current_idx - 1]

            # Golden cross: SMA50 crosses above SMA200
            if (current_sma50 > current_sma200 and prev_sma50 <= prev_sma200):
                return days_back

    # No golden cross found in last 60 days
    return None


def calculate_gc_within_sma50(prices: List[float]) -> Optional[float]:
    """Calculate if current price is within SMA50 range."""
    if len(prices) < 50:
        return None

    current_price = prices[-1]
    sma_50 = calculate_sma(prices, 50)

    if sma_50 is None or sma_50 == 0:
        return None

    return (current_price - sma_50) / sma_50


def calculate_bb_width(prices: List[float], period: int) -> Optional[float]:
    """Calculate Bollinger Band width."""
    if len(prices) < period:
        return None

    sma = calculate_sma(prices, period)
    if sma is None:
        return None

    # Calculate standard deviation
    recent_prices = prices[-period:]
    variance = sum((price - sma) ** 2 for price in recent_prices) / period
    std_dev = variance ** 0.5

    # BB width = (2 * std_dev) / sma
    return (2 * std_dev) / sma if sma != 0 else None


def calculate_bb_width_percentile(prices: List[float], period: int, lookback: int, percentile: float) -> Optional[float]:
    """Calculate BB width percentile over the last lookback days only."""
    if len(prices) < max(period, lookback):
        return None

    # Calculate BB width for each day in the last lookback days only
    bb_widths = []

    # Start from the end and work backwards for the last lookback days
    start_idx = max(period - 1, len(prices) - lookback)

    for i in range(start_idx, len(prices)):
        bb_width = calculate_bb_width(prices[:i+1], period)
        if bb_width is not None:
            bb_widths.append(bb_width)

    if len(bb_widths) < 2:
        return None

    # Sort and find percentile using nearest-rank method
    bb_widths.sort()
    # Use nearest-rank: idx = int(p * (N-1)) for 0-based indexing
    percentile_index = int(percentile * (len(bb_widths) - 1))
    percentile_index = min(percentile_index, len(bb_widths) - 1)

    return bb_widths[percentile_index]


def calculate_volume_ratio_from_db(symbol: str, db_service) -> Optional[float]:
    """Calculate volume ratio (today / 20-day average) from database."""
    try:
        # Get recent volume data
        end_date = datetime.now()

        # Query volume data directly from database
        from models import DailyPrice
        with db_service.db_manager.SessionLocal() as session:
            volumes = session.query(DailyPrice.volume).filter(
                DailyPrice.symbol == symbol,
                DailyPrice.date <= end_date.date(),
                DailyPrice.volume.isnot(None),
                DailyPrice.volume > 0
            ).order_by(DailyPrice.date.desc()).limit(21).all()

        if len(volumes) < 21:  # Need at least 21 days (today + 20 days)
            return None

        volumes_list = [v[0] for v in volumes]
        today_volume = volumes_list[0]
        avg_20_volume = sum(volumes_list[1:21]) / 20

        if avg_20_volume == 0:
            return None

        return today_volume / avg_20_volume

    except Exception as e:
        logger.warning(f"Failed to calculate volume ratio for {symbol}: {e}")
        return None


def calculate_volume_contraction_from_db(symbol: str, db_service) -> Optional[bool]:
    """Calculate if 20-day volume is less than 60-day volume from database."""
    try:
        # Get recent volume data - need at least 60 days for proper calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)  # Get 120 days to ensure we have 60 trading days

        # Query volume data directly from database
        from models import DailyPrice
        session = db_service.db_manager.SessionLocal()

        volumes = session.query(DailyPrice.volume).filter(
            DailyPrice.symbol == symbol,
            DailyPrice.date >= start_date.date(),
            DailyPrice.date <= end_date.date(),
            DailyPrice.volume.isnot(None),
            DailyPrice.volume > 0
        ).order_by(DailyPrice.date.desc()).limit(60).all()

        session.close()

        # Need at least 60 days for proper 20d vs 60d comparison
        if len(volumes) < 60:
            logger.debug(f"Volume contraction for {symbol}: Only {len(volumes)} days of volume data (need 60)")
            return None

        volumes_list = [v[0] for v in volumes]

        # True 20d vs 60d comparison as per specification
        vol_20 = sum(volumes_list[:20]) / 20  # Most recent 20 days
        vol_60 = sum(volumes_list) / 60  # All 60 days

        result = vol_20 < vol_60
        logger.debug(f"Volume contraction for {symbol}: vol_20={vol_20:.0f}, vol_60={vol_60:.0f}, result={result}")
        return result

    except Exception as e:
        logger.warning(f"Failed to calculate volume contraction for {symbol}: {e}")
        return None


def calculate_relative_strength_vs_spy(symbol_prices: List[float], spy_prices: List[float]) -> Optional[float]:
    """Calculate 6-month relative strength vs SPY using growth factors."""
    if not spy_prices or len(symbol_prices) < 126 or len(spy_prices) < 126:
        return None
    try:
        # Sanity guards for corrupt data
        if symbol_prices[-126] <= 0 or spy_prices[-126] <= 0:
            return None

        # Calculate 6-month growth factors (126 trading days)
        symbol_growth = symbol_prices[-1] / symbol_prices[-126]
        spy_growth = spy_prices[-1] / spy_prices[-126]

        # Relative strength = (Stock Growth / SPY Growth - 1) * 100
        if spy_growth == 0:
            return None

        relative_strength = (symbol_growth / spy_growth - 1) * 100

        # Sanity check for reasonable values (avoid extreme outliers)
        if abs(relative_strength) > 1000:  # More than 1000% relative strength is likely an error
            return None

        return relative_strength
    except Exception as e:
        logger.warning(f"Failed to calculate relative strength: {e}")
        return None


def calculate_technical_indicators_for_symbols(symbols: List[str], db_service) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Calculate technical indicators for a list of symbols using available database data.

    Args:
        symbols: List of symbols to calculate indicators for
        db_service: Database service instance

    Returns:
        Tuple of (rsi_data, technical_data) dictionaries
    """
    rsi_data = {}
    technical_data = {}

    if not symbols:
        return rsi_data, technical_data

    try:
        # Get price history for all symbols
        price_history = db_service.get_price_history(
            tickers=symbols,
            start=datetime.now() - timedelta(days=420),  # Need 420 days for 260+ trading days
            end=datetime.now(),
            price_field=PRICE_FIELD
        )

        # Get SPY data for relative strength calculation
        spy_prices = None
        try:
            spy_history = db_service.get_price_history(
                tickers=[BENCH],  # SPY equivalent
                start=datetime.now() - timedelta(days=420),
                end=datetime.now(),
                price_field=PRICE_FIELD
            )
            if BENCH in spy_history.columns:
                spy_prices = spy_history[BENCH].dropna().tolist()
        except Exception as e:
            logger.warning(f"Failed to get SPY data: {e}")

        # Calculate technical indicators for each symbol
        for symbol in symbols:
            if symbol in price_history.columns:
                symbol_prices = price_history[symbol].dropna().tolist()

                # Calculate RSI
                if len(symbol_prices) >= 14:
                    rsi_value = calculate_rsi(symbol_prices)
                    rsi_data[symbol] = rsi_value

                # Calculate other technical indicators
                if len(symbol_prices) >= 20:  # Minimum for SMA20
                    # Calculate volume-based indicators
                    volume_ratio = calculate_volume_ratio_from_db(symbol, db_service)
                    vol20_lt_vol60 = calculate_volume_contraction_from_db(symbol, db_service)

                    # Calculate relative strength vs SPY
                    rs_6m_vs_spy = calculate_relative_strength_vs_spy(symbol_prices, spy_prices)

                    technical_data[symbol] = {
                        'sma_20': calculate_sma(symbol_prices, 20) if len(symbol_prices) >= 20 else None,
                        'sma_50': calculate_sma(symbol_prices, 50) if len(symbol_prices) >= 50 else None,
                        'sma_200': calculate_sma(symbol_prices, 200) if len(symbol_prices) >= 200 else None,
                        'rs_6m_vs_spy': rs_6m_vs_spy,
                        'gc_days_since': calculate_golden_cross_days(symbol_prices) if len(symbol_prices) >= 260 else None,
                        'gc_within_sma50_frac': calculate_gc_within_sma50(symbol_prices) if len(symbol_prices) >= 50 else None,
                        'bb_width_20': calculate_bb_width(symbol_prices, 20) if len(symbol_prices) >= 20 else None,
                        'bb_width_20_p30_6m': calculate_bb_width_percentile(symbol_prices, 20, 126, 0.30) if len(symbol_prices) >= 126 else None,
                        'vol20_lt_vol60': vol20_lt_vol60,
                        'volume_ratio': volume_ratio
                    }
    except Exception as e:
        logger.warning(f"Failed to calculate technical indicators: {e}")

    return rsi_data, technical_data
