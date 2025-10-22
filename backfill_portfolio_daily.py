#!/usr/bin/env python3
"""
Backfill PortfolioDaily table from TransactionHistory data.

This script reconstructs daily portfolio values from transaction history,
correctly calculating each field according to Trading212 API definitions.

Key Definitions:
- value = cash + invested + unrealised_profit
- invested = cost basis of current holdings (what you paid for stocks you currently own)
- unrealised_profit = paper gain/loss on current holdings
- realised_profit = locked-in gains from completed sales
- cash = available cash balance

Usage:
    python backfill_portfolio_daily2.py
"""

from datetime import date, timedelta
from typing import Optional, Tuple

from sqlalchemy import select, func, and_

from update_data import get_session
from models import TransactionAction, TransactionHistory, Instrument, CurrencyRateDaily, PortfolioDaily, PricesDaily

# Constants
BACKFILL_END_DATE = date(2025, 8, 29)

# In-memory cache of last seen price per ISIN during this run.
# Updated as transactions are processed; read by get_price for daily calc.
PRICE_CACHE: dict[str, float] = {}


class Holding:
    """Represents a stock holding with cost basis tracking."""

    def __init__(self, isin: str, name: str):
        self.isin = isin
        self.name = name
        self.quantity = 0.0
        self.total_cost = 0.0  # Total money spent on this holding
        self.avg_buy_price = 0.0  # Weighted average buy price

    def add_buy(self, quantity: float, total: float):
        """Add a buy transaction to this holding."""
        self.quantity += quantity
        self.total_cost = round(self.total_cost + total, 9)
        self.avg_buy_price = self.total_cost / self.quantity

    def add_sell(self, quantity: float, total: float):
        """Add a sell transaction and return realised profit."""
        self.quantity = max(round(self.quantity - quantity, 9), 0)
        self.total_cost = round(self.total_cost - max(total, 0), 9)

    def __repr__(self) -> str:
        return f"<Holding(name='{self.name}' isin={self.isin}, quantity={self.quantity}, total_cost={self.total_cost}>"


def get_price(session, isin: str, target_date: date) -> Optional[float]:
    """
    Get market price for a stock on a specific date from PricesDaily table.

    Args:
        session: Database session
        isin: Stock ISIN
        target_date: Date to get price for

    Returns:
        Price per share in GBP, or None if not available
    """
    # Try market price from PricesDaily first
    instrument = session.execute(select(Instrument).where(Instrument.isin == isin)).scalar_one_or_none()

    if instrument and instrument.yahoo_symbol:
        price = session.execute(
            select(PricesDaily.close_price)
            .where(
                and_(
                    PricesDaily.symbol == instrument.yahoo_symbol,
                    PricesDaily.date <= target_date,
                )
            )
            .order_by(PricesDaily.date.desc())
            .limit(1)
        ).scalar_one_or_none()
        if price is not None:
            rate = get_currency_rate_on_date(session, instrument.currency, target_date)

            return price * rate

    # Fallback to last seen transaction price in cache
    return PRICE_CACHE.get(isin)


def get_instrument_by_isin(session, isin: str) -> Optional[Instrument]:
    """Get instrument by ISIN."""
    return session.execute(select(Instrument).where(Instrument.isin == isin)).scalar_one_or_none()


def get_currency_rate_on_date(session, from_currency: str, target_date: date) -> Optional[float]:
    """
    Get currency rate (to GBP) for a currency on a specific date.
    If not available (weekend/holiday), forward-fill from the last available rate.
    """
    # Handle GBP directly
    if from_currency == "GBP":
        return 1.0

    # Handle GBX (pence) - convert to GBP
    if from_currency == "GBX":
        return 0.01  # 1 GBX = 0.01 GBP

    # First try to get exact date
    rate = session.execute(
        select(CurrencyRateDaily.rate).where(
            and_(
                CurrencyRateDaily.from_currency == from_currency,
                CurrencyRateDaily.to_currency == "GBP",
                CurrencyRateDaily.date == target_date,
            )
        )
    ).scalar_one_or_none()

    if rate is not None:
        return float(rate)

    # TODO: Do we need this?
    # If not found, get the last available rate before the target date
    last_rate = session.execute(
        select(CurrencyRateDaily.rate)
        .where(
            and_(
                CurrencyRateDaily.from_currency == from_currency,
                CurrencyRateDaily.to_currency == "GBP",
                CurrencyRateDaily.date < target_date,
            )
        )
        .order_by(CurrencyRateDaily.date.desc())
        .limit(1)
    ).scalar_one_or_none()

    return float(last_rate) if last_rate is not None else None


def process_transaction_on_date(
    transaction: TransactionHistory,
    holdings: dict[str, Holding],
    cash_balance: float,
    total_realised_profit: float,
    session,
) -> Tuple[float, float]:
    """
    Process a single transaction and update holdings and cash balance.

    Returns: (updated_cash_balance, updated_total_realised_profit)
    """
    # Handle cash flows
    if transaction.action.is_cash_positive():
        cash_balance += transaction.total
    else:
        cash_balance -= transaction.total

    # Handle stock transactions
    if transaction.isin and transaction.quantity:
        isin = transaction.isin

        # Get or create holding
        if isin not in holdings:
            instrument = get_instrument_by_isin(session, isin)
            name = instrument.name if instrument else f"Unknown ({isin})"
            holdings[isin] = Holding(isin, name)

        holding = holdings[isin]

        # Calculate price in GBP from the actual transaction
        # transaction.total is already in GBP, so we can directly calculate the price
        # Update last seen price cache for this ISIN
        if transaction.action != TransactionAction.STOCK_SPLIT_OPEN:
            # we have STOCK_SPLIT_OPEN after CLOSE, so we will ignore open
            PRICE_CACHE[isin] = (transaction.total - transaction.total_fees) / transaction.quantity

        # Handle buy/sell transactions
        if transaction.action in {TransactionAction.MARKET_BUY, TransactionAction.LIMIT_BUY}:
            holding.add_buy(transaction.quantity, transaction.total - transaction.total_fees)
        elif transaction.action in {TransactionAction.MARKET_SELL, TransactionAction.LIMIT_SELL}:
            holding.add_sell(transaction.quantity, transaction.total - transaction.total_fees)
            total_realised_profit += transaction.result or 0
        elif transaction.action == TransactionAction.STOCK_SPLIT_CLOSE:
            holding.quantity = transaction.quantity

        if holding.quantity == 0:
            holdings.pop(holding.isin)
            del holding

    if cash_balance < -10:
        raise ValueError(
            f"Cash {cash_balance} can't be negative (processing {transaction.timestamp} {transaction.ticker} {transaction.action})"
        )

    return cash_balance, total_realised_profit


def backfill_portfolio_daily():
    """Main function to backfill PortfolioDaily table."""
    # Calculate total days

    with get_session() as session:
        # Get all transactions in date order for efficient processing
        all_transactions = (
            session.execute(
                select(TransactionHistory)
                .where(func.date(TransactionHistory.timestamp) <= BACKFILL_END_DATE)
                .order_by(TransactionHistory.timestamp, TransactionHistory.id)
            )
            .scalars()
            .all()
        )

        # Initialize state
        holdings: dict[str, Holding] = {}
        cash_balance = 0.0
        total_realised_profit = 0.0
        transaction_index = 0

        # Process each date
        processed = 0
        current_date = all_transactions[0].timestamp.date()
        backfill_start_date = current_date
        total_days = (BACKFILL_END_DATE - backfill_start_date).days + 1

        print(f"📊 Total days to backfill: {total_days}")
        print(f"📋 Loaded {len(all_transactions)} transactions")

        while current_date <= BACKFILL_END_DATE:
            # Process all transactions for this date
            while (
                transaction_index < len(all_transactions)
                and all_transactions[transaction_index].timestamp.date() == current_date
            ):
                transaction = all_transactions[transaction_index]
                cash_balance, total_realised_profit = process_transaction_on_date(
                    transaction, holdings, cash_balance, total_realised_profit, session
                )
                transaction_index += 1

            # Calculate portfolio metrics for this date
            invested = sum(holding.total_cost for holding in holdings.values())

            # Calculate unrealised profit using current market prices
            unrealised_profit = 0.0
            for isin, holding in holdings.items():
                if holding.quantity > 0:
                    current_price = get_price(session, isin, current_date)
                    if current_price is not None:
                        current_value = holding.quantity * current_price
                        unrealised_profit += current_value - holding.total_cost
                    else:
                        raise ValueError(f"No price available for {holding.name} on {current_date}")
                elif holding.quantity < 0:
                    raise ValueError("I should not have negative value")

            value = cash_balance + invested + unrealised_profit

            # Debug output for first few days
            print(
                f"📊 {current_date}: Cash={cash_balance:.2f}, Invested={invested:.2f}, Unrealised={unrealised_profit:.2f}, Realised={total_realised_profit:.2f}, Value={value:.2f}"
            )
            if current_date == BACKFILL_END_DATE:
                for isin, holding in sorted(list(holdings.items()), key=lambda x: x[1].name):
                    current_price = get_price(session, isin, current_date)
                    print(
                        f"   {holding.name}: {holding.quantity:.4f} @ £{holding.avg_buy_price:.2f} = £{holding.total_cost:.2f} (market: £{current_price:.2f})"
                    )

            # Check if record already exists
            existing = session.execute(
                select(PortfolioDaily).where(PortfolioDaily.date == current_date)
            ).scalar_one_or_none()

            if existing:
                # Update existing record
                existing.value = value
                existing.unrealised_profit = unrealised_profit
                existing.realised_profit = round(total_realised_profit, 9)
                existing.cash = round(cash_balance, 9)
                existing.invested = round(invested, 9)
            else:
                # Create new PortfolioDaily record
                portfolio_daily = PortfolioDaily(
                    date=current_date,
                    value=value,
                    unrealised_profit=unrealised_profit,
                    realised_profit=round(total_realised_profit, 9),
                    cash=round(cash_balance, 9),
                    invested=round(invested, 9),
                )
                session.add(portfolio_daily)

            processed += 1

            # Commit every 10 records to avoid long transactions
            if processed % 10 == 0:
                session.commit()
                print(f"📈 Processed {processed}/{total_days} days, current: {current_date}")

            current_date += timedelta(days=1)

        # Commit any remaining changes
        session.commit()

        print(f"\n✅ Backfill complete!")
        print(f"   📊 Processed: {processed} days")
        print(f"   📅 Date range: {backfill_start_date} to {BACKFILL_END_DATE}")
        print(f"\n📝 All calculations complete including unrealised_profit and value!")


if __name__ == "__main__":
    backfill_portfolio_daily()


