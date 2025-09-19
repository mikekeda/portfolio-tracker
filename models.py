"""
SQLAlchemy models for Trading212 Portfolio Manager
==================================================
Defines the database schema using SQLAlchemy ORM.
"""

from datetime import date, datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import BigInteger, Date, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import TIMEZONE


class Base(DeclarativeBase):
    """Base class for declarative models."""

    pass


class PricesDaily(Base):
    """Daily stock price data from Yahoo Finance."""

    __tablename__ = "prices_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)

    # Price data
    open_price: Mapped[float] = mapped_column(Float, nullable=False)
    high_price: Mapped[float] = mapped_column(Float, nullable=False)
    low_price: Mapped[float] = mapped_column(Float, nullable=False)
    close_price: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close_price: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_symbol_date"),
        Index("idx_symbol_date", "symbol", "date"),
    )

    def __repr__(self) -> str:
        return f"<DailyPrice(symbol='{self.symbol}', date='{self.date}', close={self.close_price})>"


class Instrument(Base):
    """Trading212 instrument metadata."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    t212_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    yahoo_symbol: Mapped[str] = mapped_column(String(20), nullable=True, index=True)

    # Static metadata (doesn't change frequently)
    sector: Mapped[str] = mapped_column(String(100), nullable=True)
    country: Mapped[str] = mapped_column(String(100), nullable=True)

    # TODO: Remove this later
    # Yahoo Finance data cache (legacy in-table storage; consider using InstrumentYahoo)
    yahoo_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)  # deprecated
    yahoo_cashflow: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)  # deprecated
    yahoo_earnings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)  # deprecated

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(TIMEZONE))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Relationships
    holdings: Mapped[List["HoldingDaily"]] = relationship(back_populates="instrument")
    # One-to-one detached Yahoo cache container
    yahoo: Mapped["InstrumentYahoo"] = relationship(back_populates="instrument", uselist=False)
    # Time-series of market metrics (market_cap, pe, etc.)
    metrics: Mapped[List["InstrumentMetricsDaily"]] = relationship(back_populates="instrument")

    def __repr__(self) -> str:
        return f"<Instrument(t212_code='{self.t212_code}', name='{self.name}')>"


class HoldingDaily(Base):
    """Portfolio holdings from Trading212."""

    __tablename__ = "holdings_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)

    # Snapshot timestamp (when this holding was recorded)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Holding data
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    avg_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    ppl: Mapped[float] = mapped_column(Float, nullable=False)  # profit/loss
    fx_ppl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # FX profit/loss

    # Market data (changes daily)
    market_cap: Mapped[float] = mapped_column(Float, nullable=True)
    pe_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    institutional: Mapped[float] = mapped_column(Float, nullable=True)
    beta: Mapped[float] = mapped_column(Float, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Relationships
    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="holdings")

    # Constraints
    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_holding_instrument_date"),
        Index("idx_holding_instrument_date", "instrument_id", "date"),
    )

    def __repr__(self) -> str:
        return f"<Holding(instrument='{self.instrument.t212_code}', quantity={self.quantity})>"


class InstrumentYahoo(Base):
    """One-to-one container for Yahoo Finance cached blobs per instrument"""

    __tablename__ = "instruments_yahoo"

    # Use instrument_id as the primary key to enforce one-to-one mapping
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), primary_key=True)

    # Cached JSONB payloads
    info: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    cashflow: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    earnings: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    recommendations: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    analyst_price_targets: Mapped[Dict[str, Any]] = mapped_column(JSONB)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship back to instrument
    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="yahoo")

    def __repr__(self) -> str:
        return f"<InstrumentYahoo(instrument_id={self.instrument_id})>"


class InstrumentMetricsDaily(Base):
    """Daily market metrics per instrument (market_cap, pe, beta, etc.)"""

    __tablename__ = "instruments_metrics_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    market_cap: Mapped[float] = mapped_column(Float, nullable=True)
    pe_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    institutional: Mapped[float] = mapped_column(Float, nullable=True)  # heldPercentInstitutions
    beta: Mapped[float] = mapped_column(Float, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_metrics_instrument_date"),
        Index("idx_metrics_instrument_date", "instrument_id", "date"),
    )

    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="metrics")

    def __repr__(self) -> str:
        return f"<InstrumentMetricsDaily(instrument_id={self.instrument_id}, date='{self.date}')>"


class CurrencyRateDaily(Base):
    """Currency exchange rates cache."""

    __tablename__ = "currency_rates_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    from_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    to_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    rate: Mapped[float] = mapped_column(Float, nullable=False)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Constraints
    __table_args__ = (UniqueConstraint("from_currency", "to_currency", "date", name="uq_currency_rate_date"),)

    def __repr__(self) -> str:
        return f"<CurrencyRate({self.from_currency}->{self.to_currency}={self.rate})>"


class PortfolioDaily(Base):
    """Portfolio snapshots for historical tracking."""

    __tablename__ = "portfolio_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)

    # Portfolio metrics
    total_value_gbp: Mapped[float] = mapped_column(Float, nullable=False)
    total_profit_gbp: Mapped[float] = mapped_column(Float, nullable=False)
    total_return_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # Allocation data (stored as JSON for flexibility)
    country_allocation: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)
    sector_allocation: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)
    currency_allocation: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)
    etf_equity_split: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    def __repr__(self) -> str:
        return f"<PortfolioSnapshot(date='{self.date}', value={self.total_value_gbp})>"
