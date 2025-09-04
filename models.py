"""
SQLAlchemy models for Trading212 Portfolio Manager
==================================================
Defines the database schema using SQLAlchemy ORM.
"""

from datetime import datetime, timezone
import os
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date,
    Index, UniqueConstraint, ForeignKey, create_engine, MetaData
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import JSONB

# Create base class for declarative models
Base = declarative_base()

# Metadata for Alembic migrations
metadata = MetaData()


class DailyPrice(Base):
    """Daily stock price data from Yahoo Finance."""
    __tablename__ = 'daily_prices'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    date = Column(Date, nullable=False)

    # Price data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    adj_close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)

    # Metadata
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

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
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    # Relationships
    holdings = relationship("Holding", back_populates="instrument")

    def __repr__(self):
        return f"<Instrument(t212_code='{self.t212_code}', name='{self.name}')>"


class Holding(Base):
    """Portfolio holdings from Trading212."""
    __tablename__ = 'holdings'

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
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    # Relationships
    instrument = relationship("Instrument", back_populates="holdings")

    def __repr__(self):
        return f"<Holding(instrument='{self.instrument.t212_code}', quantity={self.quantity})>"


class CurrencyRate(Base):
    """Currency exchange rates cache."""
    __tablename__ = 'currency_rates'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    from_currency = Column(String(3), nullable=False)
    to_currency = Column(String(3), nullable=False)
    rate = Column(Float, nullable=False)

    # Metadata
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    # Constraints
    __table_args__ = (
        UniqueConstraint('from_currency', 'to_currency', 'date', name='uq_currency_rate_date'),
    )

    def __repr__(self):
        return f"<CurrencyRate({self.from_currency}->{self.to_currency}={self.rate})>"


class PortfolioSnapshot(Base):
    """Portfolio snapshots for historical tracking."""
    __tablename__ = 'portfolio_snapshots'

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
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<PortfolioSnapshot(date='{self.date}', value={self.total_value_gbp})>"


# Database connection and session management
class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def close(self):
        """Close the database engine."""
        self.engine.dispose()


def get_db_url() -> str:
    """Construct database URL from environment variables."""
    db_name = os.getenv("DB_NAME", "trading212_portfolio")
    db_password = os.getenv("DB_PASSWORD")
    db_user = os.getenv("DB_USER", "postgres")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")

    if not db_password:
        raise ValueError("DB_PASSWORD environment variable is required")

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        db_url = get_db_url()
        _db_manager = DatabaseManager(db_url)
    return _db_manager


def init_database():
    """Initialize the database with tables."""
    db_manager = get_db_manager()
    db_manager.create_tables()
    return db_manager
