"""
SQLAlchemy models for Trading212 Portfolio Manager
==================================================
Defines the database schema using SQLAlchemy ORM.
"""
# pylint: disable=unsubscriptable-object

import enum
from datetime import date, datetime
from typing import Any, Literal, Optional, TypedDict

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from backend.schemas.instrument_thesis import ThesisConviction
from backend.utils.pe_history import avg_pe
from config import TIMEZONE

InstrumentTag = Literal[
    "semiconductor",
    "ai",
    "software",
    "cloud",
    "defense",
    "space",
    "healthcare",
    "financial",
    "growth",
    "speculative",
    "EU",
    "etf",
    "commodity",
]


class ThesisRuleTypedDict(TypedDict, total=False):
    """Screener-style rule in thesis buy_rules / sell_rules.

    Entries may also be boolean groups: {"all": [rules…]} or {"any": [rules…]}
    with an optional description, nesting arbitrarily. Full validation lives in
    backend/schemas/instrument_thesis.py (ThesisRuleNode); this TypedDict stays
    loose because TypedDicts can't express the recursive union.
    """

    field: str
    operator: str
    op: str
    value: float | int | str | dict[str, str]
    description: str
    all: list[dict]
    any: list[dict]


class InstrumentThesisTypedDict(TypedDict, total=False):
    """User-authored thesis JSON stored on Instrument.thesis (JSONB)."""

    summary: str
    target_weight_min_pct: float
    target_weight_max_pct: float
    buy_rules: list[ThesisRuleTypedDict]
    sell_rules: list[ThesisRuleTypedDict]
    buy_triggers: list[str]
    sell_triggers: list[str]
    horizon_years: int
    conviction: ThesisConviction | None
    authored_on: str  # ISO date — when the thesis was written or last materially revised


class TransactionAction(enum.Enum):
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

    def is_cash_positive(self):
        return self not in {self.MARKET_BUY, self.LIMIT_BUY, self.WITHDRAWAL, self.STOCK_SPLIT_OPEN}


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

    # The unique constraint's index already serves every (symbol, date) lookup;
    # a second one on the same columns only doubles write cost on the hot path.
    __table_args__ = (UniqueConstraint("symbol", "date", name="uq_symbol_date"),)

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
    cik: Mapped[str] = mapped_column(String(10), nullable=True)
    thesis: Mapped[Optional[InstrumentThesisTypedDict]] = mapped_column(JSONB, nullable=True)
    tags: Mapped[Optional[list[InstrumentTag]]] = mapped_column(JSONB, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(TIMEZONE))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Relationships
    holdings: Mapped[list["HoldingDaily"]] = relationship(back_populates="instrument")
    # One-to-one detached Yahoo cache container
    yahoo: Mapped["InstrumentYahoo"] = relationship(back_populates="instrument", uselist=False)
    # Time-series of market metrics (market_cap, pe, etc.)
    metrics: Mapped[list["InstrumentMetricsDaily"]] = relationship(back_populates="instrument")
    # Transaction history from CSV exports
    transactions: Mapped[list["TransactionHistory"]] = relationship(
        back_populates="instrument",
        foreign_keys="[TransactionHistory.ticker]",
        primaryjoin="Instrument.t212_code == TransactionHistory.ticker",
    )
    # Earnings reports summaries
    earnings_reports: Mapped[list["EarningsReport"]] = relationship(back_populates="instrument")
    # 13F institutional holdings (when cusip matches)
    form13f_holdings: Mapped[list["Form13FHolding"]] = relationship(
        "Form13FHolding", back_populates="instrument"
    )
    # LLM position review snapshots
    position_reviews: Mapped[list["PositionReview"]] = relationship(back_populates="instrument")
    # Daily point-in-time feature snapshots (trade-suggestion agent / ML training)
    features: Mapped[list["FeaturesDaily"]] = relationship(back_populates="instrument")
    # Daily agent trade suggestions
    trade_suggestions: Mapped[list["TradeSuggestion"]] = relationship(back_populates="instrument")

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

    __table_args__ = (UniqueConstraint("instrument_id", "date", name="uq_holding_instrument_date"),)

    def __repr__(self) -> str:
        return f"<Holding(instrument='{self.instrument.t212_code}', quantity={self.quantity})>"


class InstrumentYahoo(Base):
    """One-to-one container for Yahoo Finance cached blobs per instrument"""

    __tablename__ = "instruments_yahoo"

    # Use instrument_id as the primary key to enforce one-to-one mapping
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), primary_key=True)

    # Cached JSONB payloads
    info: Mapped[dict[str, Any]] = mapped_column(JSONB)
    cashflow: Mapped[dict[str, Any]] = mapped_column(JSONB)
    earnings: Mapped[dict[str, Any]] = mapped_column(JSONB)
    recommendations: Mapped[dict[str, Any]] = mapped_column(JSONB)
    analyst_price_targets: Mapped[dict[str, Any]] = mapped_column(JSONB)
    splits: Mapped[dict[str, Any]] = mapped_column(JSONB)
    news: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    pes: Mapped[dict[str, Any]] = mapped_column(JSONB)
    balance_sheet: Mapped[dict[str, Any]] = mapped_column(JSONB)
    income_stmt: Mapped[dict[str, Any]] = mapped_column(JSONB)

    # Quarterly statements, fetched once per quarter (see the mostRecentQuarter
    # gate in update_data). Annual payloads above stay — Piotroski needs them.
    quarterly_cashflow: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    quarterly_balance_sheet: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    quarterly_income_stmt: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    # Analyst estimate revisions and forward estimates.
    estimates: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    # Bumped by ANY writer (onupdate) — including the nightly PE scrapers that
    # only touch `pes`. Never use it to decide profile freshness.
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    # When update_data last fetched the Yahoo profile (info & friends).
    # Drives the staleness queue; only the fetch path may set it.
    profile_fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Estimates move weekly at most, so they have their own slower gate.
    estimates_fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationship back to instrument
    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="yahoo")

    def __repr__(self) -> str:
        return f"<InstrumentYahoo(instrument_id={self.instrument_id})>"

    @hybrid_property
    def avg_pe_5y(self) -> Optional[float]:
        """Return the representative PE ratio over the last 5 years"""

        return avg_pe(self.pes, datetime.now(TIMEZONE).date())


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

    # Insider trades aggregated over trailing 90 days (yfinance, US-only coverage in practice)
    insider_buy_count_90d: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    insider_sell_count_90d: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    insider_net_value_90d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("instrument_id", "date", name="uq_metrics_instrument_date"),)

    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="metrics")

    def __repr__(self) -> str:
        return f"<InstrumentMetricsDaily(instrument_id={self.instrument_id}, date='{self.date}')>"


class FeaturesDaily(Base):
    """Daily point-in-time feature snapshot per instrument (scripts/update_features.py).

    Persists the derived, model-ready features that otherwise exist only in the
    overwritten InstrumentYahoo cache or are computed at request time, so the
    trade-suggestion agent can be backtested and an ML ranker trained on them.
    """

    __tablename__ = "features_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Quality / fundamentals (percent units where the source is a ratio)
    roic: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gross_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revenue_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fcf_yield: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    short_percent_float: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Analyst sentiment
    analyst_rec_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    analyst_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    analyst_target_upside: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % to median target

    # Valuation
    forward_pe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    peg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ps_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pe_5y_avg_vs_current_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dcf_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dcf_diff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dcf_implied_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Composites
    rule_of_40: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    screener_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Score reachable in this instrument's sector — the denominator that makes
    # screener_score comparable across sectors with different exclusions.
    screener_score_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Trends from quarterly statements — the only multi-period fundamentals here.
    # roic_ttm is trailing-twelve-month, directly comparable with `roic` above.
    roic_ttm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Change in TTM ROIC across however many rolling windows Yahoo's quarterly
    # statements support — usually 1-2, so not a fixed 4-quarter span.
    roic_ttm_trend: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gross_margin_trend_4q: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_margin_trend_4q: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revenue_growth_4q_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eps_revision_ratio_30d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eps_next_q_growth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    passed_screeners: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)
    thesis_rule_eval: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Spillover for new features without a migration
    extras: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("instrument_id", "date", name="uq_features_instrument_date"),)

    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="features")

    def __repr__(self) -> str:
        return f"<FeaturesDaily(instrument_id={self.instrument_id}, date='{self.date}')>"


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
    country_allocation: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=True)
    sector_allocation: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=True)
    currency_allocation: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=True)
    etf_equity_split: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=True)

    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    beta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mwrr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Money-Weighted Return (annualized)
    twrr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Time-Weighted Return (annualized)
    jensens_alpha: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    positions_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    positions_winning: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

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
    dividend_details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Raw settings data (keep for debugging/completeness)
    settings: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE)
    )

    # Relationships
    instruments: Mapped[list["PieInstrument"]] = relationship(
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
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Issues/notes
    issues: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)

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
    action: Mapped[TransactionAction] = mapped_column(Enum(TransactionAction), nullable=False, index=True)

    # Original CSV ID (for reference, nullable since some transactions don't have IDs)
    # No index=True: uq_transaction_csv_id below already indexes this column.
    csv_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    isin: Mapped[Optional[str]] = mapped_column(String(12), nullable=True, index=True)

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

    # Hand-written record of *why* this transaction was made — the reasoning behind a
    # deposit, buy or sell ("looked oversold, deposited £300 from salary"). Distinct
    # from `notes`, which is broker-supplied: update_history_from_csv.py fills that
    # from the CSV export and sync_transactions.py sets it to None. Importers must
    # never write this column, or a re-import would erase the annotations.
    decision_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Fees (CSV fee columns: "Currency conversion fee", "Stamp duty reserve tax", etc.)
    # Structure: [{"name": "CURRENCY_CONVERSION_FEE", "quantity": -0.05, "timeCharged": "2024-04-18 18:03:20"}, ...]
    fees: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONB, nullable=True)

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
            return f"<TransactionHistory({self.timestamp.date().isoformat()} {self.action} {qty:.4f} {self.ticker}{price_str}, net_cost=£{self.net_cost:.2f})>"
        else:
            return f"<TransactionHistory({self.timestamp.date().isoformat()} {self.action} {self.ticker or 'N/A'} £{self.total:.2f}, net_cost=£{self.net_cost:.2f})>"


class MarketMetricsDaily(Base):
    """Daily market metrics (buffett_indicator, yield_spread, fear_greed_index, etc.)"""

    __tablename__ = "market_metrics_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)

    buffett_indicator: Mapped[float] = mapped_column(Float, nullable=True)
    yield_spread: Mapped[float] = mapped_column(Float, nullable=True)
    fear_greed_index: Mapped[float] = mapped_column(Float, nullable=True)
    vix: Mapped[float] = mapped_column(Float, nullable=True)
    market_breadth_indicator: Mapped[float] = mapped_column(Float, nullable=True)
    sp500_above_sma200: Mapped[float] = mapped_column(Float, nullable=True)
    consumer_sentiment: Mapped[float] = mapped_column(Float, nullable=True)
    real_yield_10y: Mapped[float] = mapped_column(Float, nullable=True)
    hy_oas: Mapped[float] = mapped_column(Float, nullable=True)

    # Metadata
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class EarningsReport(Base):
    """Earnings Report"""

    __tablename__ = "earnings_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # No index=True: uq_earnings_reports_instrument_date leads with this column.
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(TIMEZONE))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="earnings_reports")

    __table_args__ = (UniqueConstraint("instrument_id", "date", name="uq_earnings_reports_instrument_date"),)

    def __repr__(self) -> str:
        return f"<EarningsReport(instrument_id={self.instrument_id}, date='{self.date}')>"


class PositionReview(Base):
    """LLM position-review snapshot for a held instrument (scripts/run_position_review.py)."""

    __tablename__ = "position_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # No index=True: idx_position_reviews_instrument_created leads with this column.
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)
    model: Mapped[str] = mapped_column(String(50), nullable=False)
    # sha256 of the rounded context JSON — identical inputs skip regeneration
    inputs_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Full validated PositionAssessmentSchema output
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="position_reviews")

    __table_args__ = (Index("idx_position_reviews_instrument_created", "instrument_id", "created_at"),)

    def __repr__(self) -> str:
        return f"<PositionReview(instrument_id={self.instrument_id}, created_at='{self.created_at}')>"


class TradeSuggestion(Base):
    """Daily agent trade suggestion (scripts/run_trade_agent.py) — suggest-only.

    value_gbp == 0 rows are vetoed intents kept for transparency; the
    constraint_adjustments field explains what blocked or clipped the trade.
    `status` records the user's decision and doubles as a future ML label.
    """

    __tablename__ = "trade_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(30), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)  # buy | add | trim | exit
    quantity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    value_gbp: Mapped[float] = mapped_column(Float, nullable=False)
    weight_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fee_gbp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rationale: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    constraint_adjustments: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="proposed")  # proposed | accepted | dismissed

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    instrument: Mapped["Instrument"] = relationship("Instrument", back_populates="trade_suggestions")

    __table_args__ = (UniqueConstraint("date", "instrument_id", "strategy", name="uq_suggestion_date_instrument_strategy"),)

    def __repr__(self) -> str:
        return f"<TradeSuggestion(date='{self.date}', instrument_id={self.instrument_id}, action='{self.action}')>"


# ─── Form 13F (Institutional Holdings) ──────────────────────────────────────


class Form13FManager(Base):
    """SEC 13F institutional investment manager (e.g. Berkshire, Baupost)."""

    __tablename__ = "form13f_managers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    cik: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(TIMEZONE))

    # Relationships
    filings: Mapped[list["Form13FFiling"]] = relationship(
        "Form13FFiling", back_populates="manager", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Form13FManager(name='{self.name}', cik='{self.cik}')>"


class Form13FFiling(Base):
    """A single 13F-HR or 13F-HR/A filing (one report period per manager)."""

    __tablename__ = "form13f_filings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    manager_id: Mapped[int] = mapped_column(Integer, ForeignKey("form13f_managers.id"), nullable=False, index=True)
    report_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    form: Mapped[str] = mapped_column(String(20), nullable=False)  # 13F-HR, 13F-HR/A
    accession_number: Mapped[str] = mapped_column(String(30), nullable=False)
    total_value: Mapped[int] = mapped_column(BigInteger, nullable=False)  # USD, nearest dollar

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(TIMEZONE))

    # Relationships
    manager: Mapped["Form13FManager"] = relationship("Form13FManager", back_populates="filings")
    holdings: Mapped[list["Form13FHolding"]] = relationship(
        "Form13FHolding", back_populates="filing", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("manager_id", "report_date", name="uq_form13f_manager_report_date"),
    )

    def __repr__(self) -> str:
        return f"<Form13FFiling(manager_id={self.manager_id}, report_date='{self.report_date}')>"


class Form13FHolding(Base):
    """Individual holding from a 13F filing (issuer, CUSIP, value, shares)."""

    __tablename__ = "form13f_holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filing_id: Mapped[int] = mapped_column(Integer, ForeignKey("form13f_filings.id"), nullable=False, index=True)
    instrument_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("instruments.id"), nullable=True, index=True
    )
    issuer: Mapped[str] = mapped_column(String(200), nullable=False)
    cusip: Mapped[str] = mapped_column(String(9), nullable=False)
    title_of_class: Mapped[str] = mapped_column(String(50), nullable=False, default="COM")
    value: Mapped[int] = mapped_column(BigInteger, nullable=False)  # USD, nearest dollar
    shares: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Relationships
    filing: Mapped["Form13FFiling"] = relationship("Form13FFiling", back_populates="holdings")
    instrument: Mapped[Optional["Instrument"]] = relationship(
        "Instrument", back_populates="form13f_holdings"
    )

    __table_args__ = (Index("idx_form13f_holding_cusip", "cusip"),)

    def __repr__(self) -> str:
        return f"<Form13FHolding(issuer='{self.issuer}', cusip='{self.cusip}', value={self.value})>"
