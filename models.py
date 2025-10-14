"""
SQLAlchemy models for Trading212 Portfolio Manager
==================================================
Defines the database schema using SQLAlchemy ORM.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from dateutil.relativedelta import relativedelta
from sqlalchemy import BigInteger, Date, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import TIMEZONE


class TransactionAction(Enum):
    """Enumeration of all action types from Trading212 CSV exports."""

    # Buy orders
    MARKET_BUY = "Market buy"
    LIMIT_BUY = "Limit buy"

    # Sell orders
    MARKET_SELL = "Market sell"
    LIMIT_SELL = "Limit sell"

    # Dividends
    DIVIDEND = "Dividend (Dividend)"
    DIVIDEND_PROPERTY = "Dividend (Property income distribution)"
    DIVIDEND_TAX_EXEMPT = "Dividend (Tax exempted)"

    # Cash movements
    DEPOSIT = "Deposit"
    WITHDRAWAL = "Withdrawal"
    INTEREST = "Interest on cash"

    # Administrative
    STOCK_SPLIT_OPEN = "Stock split open"
    STOCK_SPLIT_CLOSE = "Stock split close"
    RESULT_ADJUSTMENT = "Result adjustment"


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
    isin: Mapped[str] = mapped_column(String(12), nullable=True, index=True, unique=True)

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
    # Transaction history from CSV exports
    transactions: Mapped[List["TransactionHistory"]] = relationship(
        back_populates="instrument",
        foreign_keys="[TransactionHistory.ticker]",
        primaryjoin="Instrument.t212_code == TransactionHistory.ticker",
    )

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
    splits: Mapped[Dict[str, Any]] = mapped_column(JSONB)
    pes: Mapped[Dict[str, Any]] = mapped_column(JSONB)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship back to instrument
    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="yahoo")

    def __repr__(self) -> str:
        return f"<InstrumentYahoo(instrument_id={self.instrument_id})>"

    @hybrid_property
    def avg_pe_5y(self) -> Optional[float]:
        """Return the average PE ratio over the last 5 years"""

        if not self.pes:
            return None

        start = datetime.now(TIMEZONE).date() + relativedelta(years=-5)

        values: List[float] = []
        for k, v in self.pes.items():
            d = datetime.strptime(k, "%Y-%m-%d").date()

            if d < start:
                continue

            pe = float(v["pe_ratio"])
            if pe > 0:
                values.append(pe)

        if not values:
            return None

        return float(sum(values) / len(values))


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
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unrealised_profit: Mapped[float] = mapped_column(Float, nullable=False)
    realised_profit: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    invested: Mapped[float] = mapped_column(Float, nullable=False)

    # Allocation data (stored as JSON for flexibility)
    country_allocation: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)
    sector_allocation: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)
    currency_allocation: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)
    etf_equity_split: Mapped[Dict[str, float]] = mapped_column(JSONB, nullable=True)

    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    sortino_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    beta: Mapped[float] = mapped_column(Float, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    @hybrid_property
    def return_pct(self) -> float:
        return (self.unrealised_profit + self.realised_profit) / self.invested * 100.0

    def __repr__(self) -> str:
        return f"<PortfolioSnapshot(date='{self.date}', value={self.value})>"


class Pie(Base):
    """Trading212 pie data for portfolio tracking."""

    __tablename__ = "pies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    progress: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    status: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, default=None)

    # Pie settings data (from second API call)
    name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    creation_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    dividend_cash_action: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    goal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Pie summary data (from first API call)
    dividend_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Raw settings data (keep for debugging/completeness)
    settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Relationships
    instruments: Mapped[List["PieInstrument"]] = relationship(
        "PieInstrument", back_populates="pie", cascade="all, delete-orphan"
    )

    # Table constraints
    __table_args__ = (
        Index("idx_pie_id", "id"),
        Index("idx_pie_name", "name"),
    )

    def __repr__(self) -> str:
        return f"<Pie(id={self.id}, name={self.name}, cash={self.cash})>"


class PieInstrument(Base):
    """Individual instruments within a Trading212 pie."""

    __tablename__ = "pie_instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pie_id: Mapped[int] = mapped_column(Integer, ForeignKey("pies.id"), nullable=False)
    t212_code: Mapped[str] = mapped_column(String(20), nullable=False)

    # Instrument allocation data
    expected_share: Mapped[float] = mapped_column(Float, nullable=False)
    current_share: Mapped[float] = mapped_column(Float, nullable=False)
    owned_quantity: Mapped[float] = mapped_column(Float, nullable=False)

    # Instrument result data
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Issues/notes
    issues: Mapped[Optional[List[str]]] = mapped_column(JSONB, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Relationships
    pie: Mapped["Pie"] = relationship("Pie", back_populates="instruments")
    instrument: Mapped[Optional["Instrument"]] = relationship(
        "Instrument", foreign_keys=[t212_code], primaryjoin="PieInstrument.t212_code == Instrument.t212_code"
    )

    # Table constraints
    __table_args__ = (
        Index("idx_pie_instrument_pie_id", "pie_id"),
        Index("idx_pie_instrument_t212_code", "t212_code"),
        UniqueConstraint("pie_id", "t212_code", name="uq_pie_t212_code"),
    )

    def __repr__(self) -> str:
        return f"<PieInstrument(pie_id={self.pie_id}, t212_code={self.t212_code}, current_share={self.current_share})>"


class TransactionHistory(Base):
    """Trading212 transaction history for tracking orders, dividends, interest, and deposits."""

    __tablename__ = "transaction_history"

    # Primary key (auto-generated, since CSV IDs are unreliable)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Core transaction details (CSV columns: "Time", "Ticker", "Action")
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    ticker: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, index=True
    )  # Nullable for deposits/interest
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # From TransactionAction enum

    # Original CSV ID (for reference, nullable since some transactions don't have IDs)
    csv_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    isin: Mapped[Optional[str]] = mapped_column(String(12), nullable=False, index=True)

    # Quantity (CSV "No. of shares")
    # Note: For orders - signed (positive for buys, negative for sells)
    #       For dividends - quantity of shares that earned the dividend
    #       For deposits/interest - 0.0
    quantity: Mapped[float] = mapped_column(Float, nullable=False)

    # Pricing (CSV columns: "Price / share", "Total", "Exchange rate", "Result")
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total: Mapped[float] = mapped_column(Float, nullable=False)  # Total value (GBP)
    exchange_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # GBP per original currency
    result: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Realized P&L for sells (GBP)

    # Notes field from CSV
    notes: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Fees (CSV fee columns: "Currency conversion fee", "Stamp duty reserve tax", etc.)
    # Structure: [{"name": "CURRENCY_CONVERSION_FEE", "quantity": -0.05, "timeCharged": "2024-04-18 18:03:20"}, ...]
    fees: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Relationships
    instrument: Mapped[Optional["Instrument"]] = relationship(
        "Instrument", foreign_keys=[ticker], primaryjoin="TransactionHistory.ticker == Instrument.t212_code"
    )

    # Table constraints
    __table_args__ = (
        # Compound index for stock transaction history queries
        Index("idx_transaction_ticker_time", "ticker", "timestamp"),
        # Unique constraint on CSV ID to prevent duplicates
        UniqueConstraint("csv_id", name="uq_transaction_csv_id"),
    )

    @hybrid_property
    def total_fees(self) -> float:
        """Calculate total fees from fees array."""
        if not self.fees:
            return 0.0
        return abs(sum(fee.get("quantity", 0) for fee in self.fees))

    @hybrid_property
    def net_cost(self) -> float:
        """Net cost including all fees (adjusted cost basis)."""
        fee_total = self.total_fees

        if self.action in {TransactionAction.MARKET_BUY, TransactionAction.LIMIT_BUY}:
            return self.total + fee_total
        elif self.action in {TransactionAction.MARKET_SELL, TransactionAction.LIMIT_SELL}:
            return -self.total - fee_total
        else:
            # Dividends, interest, deposits are all positive cash flows
            return self.total - fee_total

    def __repr__(self) -> str:
        if self.action in {
            TransactionAction.MARKET_BUY,
            TransactionAction.LIMIT_BUY,
            TransactionAction.MARKET_SELL,
            TransactionAction.LIMIT_SELL,
        }:
            qty = abs(self.quantity)
            price_str = f" @ £{self.price:.2f}" if self.price else ""
            return (
                f"<TransactionHistory({self.action} {qty:.4f} {self.ticker}{price_str}, net_cost=£{self.net_cost:.2f})>"
            )
        else:
            return f"<TransactionHistory({self.action} {self.ticker or 'N/A'} £{self.total:.2f}, net_cost=£{self.net_cost:.2f})>"
