"""
SQLAlchemy models for Trading212 Portfolio Manager
==================================================
Defines the database schema using SQLAlchemy ORM.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, DateTime, Date,
    Index, UniqueConstraint, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.dialects.postgresql import JSONB


class Base(DeclarativeBase):
    """Base class for declarative models."""
    pass


class PricesDaily(Base):
    """Daily stock price data from Yahoo Finance."""
    __tablename__ = 'prices_daily'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)

    # Price data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    adj_close_price = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    # Metadata
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Constraints
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_symbol_date'),
        Index('idx_symbol_date', 'symbol', 'date'),
    )

    def __repr__(self):
        return f"<DailyPrice(symbol='{self.symbol}', date='{self.date}', close={self.close_price})>"


class Instrument(Base):
    """Trading212 instrument metadata."""
    __tablename__ = 'instruments'

    id = Column(Integer, primary_key=True)
    t212_code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    currency = Column(String(3), nullable=False)
    yahoo_symbol = Column(String(20), nullable=True, index=True)

    # Static metadata (doesn't change frequently)
    sector = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)

    # Yahoo Finance data cache
    yahoo_data = Column(JSONB, nullable=True)  # Stores Yahoo Finance profile data

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    holdings = relationship("HoldingDaily", back_populates="instrument")

    def __repr__(self):
        return f"<Instrument(t212_code='{self.t212_code}', name='{self.name}')>"


class HoldingDaily(Base):
    """Portfolio holdings from Trading212."""
    __tablename__ = 'holdings_daily'

    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id'), nullable=False)

    # Snapshot timestamp (when this holding was recorded)
    date = Column(Date, nullable=False, index=True)

    # Holding data
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    ppl = Column(Float, nullable=False)  # profit/loss
    fx_ppl = Column(Float, nullable=False, default=0.0)  # FX profit/loss

    # Market data (changes daily)
    market_cap = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    institutional = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)

    # Metadata
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    instrument = relationship("Instrument", back_populates="holdings")

    def __repr__(self):
        return f"<Holding(instrument='{self.instrument.t212_code}', quantity={self.quantity})>"


class CurrencyRateDaily(Base):
    """Currency exchange rates cache."""
    __tablename__ = 'currency_rates_daily'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    from_currency = Column(String(3), nullable=False)
    to_currency = Column(String(3), nullable=False)
    rate = Column(Float, nullable=False)

    # Metadata
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Constraints
    __table_args__ = (
        UniqueConstraint('from_currency', 'to_currency', 'date', name='uq_currency_rate_date'),
    )

    def __repr__(self):
        return f"<CurrencyRate({self.from_currency}->{self.to_currency}={self.rate})>"


class PortfolioDaily(Base):
    """Portfolio snapshots for historical tracking."""
    __tablename__ = 'portfolio_daily'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True, index=True)

    # Portfolio metrics
    total_value_gbp = Column(Float, nullable=False)
    total_profit_gbp = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=False)

    # Allocation data (stored as JSON for flexibility)
    country_allocation = Column(JSONB, nullable=True)
    sector_allocation = Column(JSONB, nullable=True)
    etf_equity_split = Column(JSONB, nullable=True)

    # Metadata
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<PortfolioSnapshot(date='{self.date}', value={self.total_value_gbp})>"
