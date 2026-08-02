"""
Technical Analysis Utilities
===========================
Helper functions for technical analysis calculations.
"""

import math
from bisect import bisect_right
from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import CURRENCIES, PRICE_FIELD, SPY, TIMEZONE, TRADING_DAYS_PER_YEAR, logger
from models import CurrencyRateDaily, PricesDaily
MIN_RETURNS_FOR_VOLATILITY = 60  # ~3 months — below this the estimate is too noisy

PRICE_COLUMN = getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price")


def calculate_rsi(prices: list[float], period: int = 14) -> Optional[float]:
    """Calculate RSI (Relative Strength Index) for a series of prices."""
    if len(prices) <= period:
        # None, not a fake-neutral 50: a fabricated 50 silently passes
        # "RSI <= 50/55" screener gates for young listings.
        return None

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


def calculate_annualized_volatility(
    prices: list[float],
    window_days: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """Annualised stdev of daily log returns, expressed as a decimal.

    A result of 0.25 means 25% annualised volatility. Returns None when there
    is not enough clean data to produce a meaningful estimate.
    """
    if not prices or len(prices) < MIN_RETURNS_FOR_VOLATILITY + 1:
        return None

    recent = prices[-(window_days + 1) :] if len(prices) > window_days + 1 else prices

    returns: list[float] = []
    for i in range(1, len(recent)):
        prev, curr = recent[i - 1], recent[i]
        if prev is None or curr is None or prev <= 0 or curr <= 0:
            continue
        returns.append(math.log(curr / prev))

    if len(returns) < MIN_RETURNS_FOR_VOLATILITY:
        return None

    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if var <= 0:
        return None
    return math.sqrt(var) * math.sqrt(TRADING_DAYS_PER_YEAR)


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
    """The `percentile`-th BB width over the last `lookback` days.

    Returned on the same scale as `calculate_bb_width`, so callers compare
    today's width against it directly.
    """
    if len(prices) < lookback + period:
        return None

    # Rolling sums rather than re-summing each of the `lookback` overlapping
    # windows — this was the hottest loop on the holdings path.
    start = len(prices) - lookback
    window = prices[start - period + 1 : start + 1]
    total = sum(window)
    total_sq = sum(price * price for price in window)

    bb_widths = []
    for i in range(start, len(prices)):
        if i > start:
            outgoing, incoming = prices[i - period], prices[i]
            total += incoming - outgoing
            total_sq += incoming * incoming - outgoing * outgoing

        mean = total / period
        if mean == 0:
            continue
        # Cancellation in E[x²]-E[x]² can push a flat window fractionally below zero
        variance = max(total_sq / period - mean * mean, 0.0)
        bb_widths.append((4 * variance**0.5) / mean)

    if not bb_widths:
        return None

    bb_widths.sort()
    return bb_widths[min(int(percentile * len(bb_widths)), len(bb_widths) - 1)]


def calculate_volume_ratio(volumes: list[int]) -> Optional[float]:
    """Latest volume over the 20-day average preceding it, oldest first."""
    if len(volumes) < 21:  # Need at least 21 days (today + 20 days)
        return None

    avg_20_volume = sum(volumes[-21:-1]) / 20
    if avg_20_volume == 0:
        return None

    return volumes[-1] / avg_20_volume


def calculate_volume_contraction(volumes: list[int]) -> Optional[bool]:
    """Whether 20-day average volume is below the 60-day average, oldest first."""
    if len(volumes) < 60:
        return None

    return sum(volumes[-20:]) / 20 < sum(volumes[-60:]) / 60


def calculate_relative_strength_vs_spy(
    symbol_prices: list[float], spy_prices: list[float], fx_factor: float = 1.0
) -> Optional[float]:
    """Calculate 6-month relative strength vs SPY using growth factors.

    ``fx_factor`` converts the stock's native-currency growth to the benchmark
    currency (GBP): rate_to_gbp(end) / rate_to_gbp(start) over the same window.
    """
    if not spy_prices or len(symbol_prices) < 126 or len(spy_prices) < 126:
        return None
    try:
        # Ensure we are comparing the same time period
        symbol_period = symbol_prices[-126:]
        spy_period = spy_prices[-126:]

        if symbol_period[0] <= 0 or spy_period[0] <= 0:
            return None

        # Calculate 6-month growth factors (126 trading days)
        symbol_growth = (symbol_period[-1] / symbol_period[0]) * fx_factor
        spy_growth = spy_period[-1] / spy_period[0]

        if spy_growth == 0:
            return None

        # Relative strength = (Stock Growth / SPY Growth - 1) * 100
        relative_strength = (symbol_growth / spy_growth - 1) * 100

        return relative_strength
    except Exception as e:
        logger.warning("Failed to calculate relative strength: %s", e)
        return None


async def _load_fx_series(
    session: AsyncSession, currencies: Optional[dict[str, str]]
) -> dict[str, tuple[list[date], list[float]]]:
    """Load to-GBP rate series (sorted dates + rates) per needed currency."""
    fx_series: dict[str, tuple[list[date], list[float]]] = {}
    needed = {c for c in (currencies or {}).values() if c in CURRENCIES}
    if not needed:
        return fx_series

    fx_result = await session.execute(
        select(CurrencyRateDaily.from_currency, CurrencyRateDaily.date, CurrencyRateDaily.rate)
        .filter(
            CurrencyRateDaily.from_currency.in_(needed),
            CurrencyRateDaily.to_currency == "GBP",
            CurrencyRateDaily.date >= datetime.now(TIMEZONE).date() - timedelta(days=420),
        )
        .order_by(CurrencyRateDaily.date)
    )
    for row in fx_result.all():
        fx_series.setdefault(row.from_currency, ([], []))
        fx_series[row.from_currency][0].append(row.date)
        fx_series[row.from_currency][1].append(row.rate)
    return fx_series


def _fx_growth_factor(series: Optional[tuple[list[date], list[float]]], symbol_dates: list[date]) -> float:
    """FX growth over the 6-month RS window: rate(end) / rate(start).

    1.0 when no conversion applies (GBP/GBX stocks — the pence factor cancels
    in the growth ratio — or missing rate data).
    """
    if not series or len(symbol_dates) < 126:
        return 1.0

    dates, rates = series

    def rate_on(target: date) -> Optional[float]:
        idx = bisect_right(dates, target) - 1  # nearest rate at or before target
        return rates[idx] if idx >= 0 else None

    start_rate = rate_on(symbol_dates[-126])
    end_rate = rate_on(symbol_dates[-1])
    if not start_rate or not end_rate:
        return 1.0
    return end_rate / start_rate


async def calculate_technical_indicators_for_symbols(
    symbols: list[str], session: AsyncSession, currencies: Optional[dict[str, str]] = None
) -> tuple[dict[str, Optional[float]], dict[str, dict[str, Any]]]:
    """Calculate technical indicators for a list of symbols using available database data.

    ``currencies`` maps yahoo_symbol -> instrument currency; when provided,
    rs_6m_vs_spy converts each stock's return to GBP so it is compared against
    the GBP benchmark (VUAG.L) on equal footing.
    """
    rsi_data: dict[str, Optional[float]] = {}
    technical_data: dict[str, dict[str, Any]] = {}

    if not symbols:
        return rsi_data, technical_data

    today = datetime.now(TIMEZONE).date()

    # Volume rides along with the price window rather than being queried per
    # symbol: two extra round trips each dominated everything else on this path.
    price_result = await session.execute(
        select(
            PricesDaily.symbol,
            PricesDaily.date,
            PricesDaily.volume,
            PRICE_COLUMN,
        )
        .filter(
            PricesDaily.symbol.in_(symbols),
            PricesDaily.date <= today,
            PricesDaily.date >= today - timedelta(days=420),
        )
        .order_by(PricesDaily.date)
    )

    price_history: dict[str, list[float]] = {}
    price_dates: dict[str, list[date]] = {}
    volume_history: dict[str, list[int]] = {}
    for row in price_result.all():
        price_history.setdefault(row.symbol, []).append(row.price)
        price_dates.setdefault(row.symbol, []).append(row.date)
        volume_history.setdefault(row.symbol, []).append(row.volume)

    spy_result = await session.execute(
        select(PRICE_COLUMN)
        .filter(PricesDaily.symbol == SPY, PricesDaily.date >= today - timedelta(days=420))
        .order_by(PricesDaily.date)
    )
    spy_prices = [row.price for row in spy_result.all()]

    fx_series = await _load_fx_series(session, currencies)

    for symbol, symbol_prices in price_history.items():
        # Per symbol, not per batch: one bad series must not blank the indicators
        # of every stock after it and then get persisted by update_features.
        try:
            rsi_data[symbol] = calculate_rsi(symbol_prices)

            symbol_volumes = volume_history[symbol]
            fx_factor = _fx_growth_factor(
                fx_series.get((currencies or {}).get(symbol, "")), price_dates[symbol]
            )
            rs_6m_vs_spy = calculate_relative_strength_vs_spy(symbol_prices, spy_prices, fx_factor)

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
                "vol20_lt_vol60": calculate_volume_contraction(symbol_volumes),
                "volume_ratio": calculate_volume_ratio(symbol_volumes),
            }
        except Exception as e:  # noqa: BLE001 — one bad symbol shouldn't stop the rest
            logger.error("Technical indicators failed for %s: %s", symbol, e, exc_info=True)

    return rsi_data, technical_data
