"""
Data Update Script for Trading212 Portfolio Manager
==================================================
Updates all database tables with fresh data from Trading212 API and Yahoo Finance.
"""

import logging
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Generator, Literal, Optional, TypeAlias, TypedDict, Union, cast

import numpy as np
import pandas as pd
import requests
import yfinance as yf  # type: ignore[import-untyped]
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert  # Added for bulk upsert
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.sql import func

from config import (
    BATCH_SIZE_YF,
    CURRENCIES,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    HISTORY_YEARS,
    PATTERN_MULTI,
    REQUEST_RETRY,
    SPY,
    TIMEZONE,
    TRADING212_API_KEY,
)
from data import ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION, STOCKS_ALIASES, STOCKS_DELISTED, STOCKS_SUFFIX
from models import (
    CurrencyRateDaily,
    HoldingDaily,
    Instrument,
    InstrumentMetricsDaily,
    InstrumentYahoo,
    PortfolioDaily,
    PricesDaily,
)


# GET /api/v0/equity/metadata/instruments
class T212Instrument(TypedDict):
    id: int
    addedOn: str  # ISO8601
    currencyCode: str  # e.g. "USD"
    isin: str
    maxOpenQuantity: int
    name: str
    shortName: str
    ticker: str  # e.g. "AAPL_US_EQ"
    type: Literal[
        "STOCK", "ETF", "FUND", "ETC", "ETN", "ADR", "REIT", "ETF_LEVERAGED", "ETF_INVERSE", "ETF_COMPLEX"
    ]  # expand if needed
    workingScheduleId: int


# GET /api/v0/equity/portfolio
class T212Position(TypedDict):
    id: int
    averagePrice: float
    currentPrice: float
    frontend: Literal["API", "WEB", "MOBILE"]  # docs example shows "API"
    fxPpl: float
    initialFillDate: str  # ISO8601
    maxBuy: float
    maxSell: float
    pieQuantity: float
    ppl: float
    quantity: float
    ticker: str


# GET /api/v0/equity/account/cash
class T212Portfolio(TypedDict):
    id: int
    blocked: float
    free: float
    invested: float
    pieCash: float
    ppl: float
    result: float
    total: float


class YahooData(TypedDict):
    info: dict[str, Any]
    cashflow: dict[str, Any]
    earnings: dict[str, Any]
    recommendations: dict[str, Any]
    analyst_price_targets: dict[str, Any]
    splits: dict[str, Any]
    news: dict[str, Any]


TRADING212_API_RESPONSE: TypeAlias = list[Union[T212Instrument, T212Position]]

YAHOO_UPDATE_LIMIT = 100
YAHOO_UPDATE_INTERVAL_DAYS = 1


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker:
    """Create and cache the database engine and session factory."""
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url, echo=False, pool_pre_ping=True)

    # from models import Pie, PieInstrument, TransactionHistory
    # TransactionHistory.__table__.create(bind=engine, checkfirst=True)

    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    SessionLocal = _get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def convert_ticker(t212: str) -> str:
    """Convert a Trading 212 code to a Yahoo Finance symbol (example: 'BARCl_EQ' -> 'BARC.L')."""
    if t212 in STOCKS_DELISTED:
        raise ValueError(f"{t212} is delisted")

    if not t212.endswith("_EQ"):
        raise ValueError(f"Unknown format: {t212}")

    core = t212[:-3]  # strip _EQ

    m = PATTERN_MULTI.match(core)
    if m:
        sym, tag = m.group("sym"), m.group("tag")
    elif core[-1].islower() or core[-1].isnumeric():  # single‑letter tag
        sym, tag = core[:-1], core[-1]
    elif t212 == "IITU_EQ":
        return core
    else:
        raise ValueError(f"Cannot parse: {t212}")

    sym = sym.rstrip("_")
    sym = STOCKS_ALIASES.get(sym, sym)
    suffix = STOCKS_SUFFIX.get(tag)
    if suffix is None:
        raise ValueError(f"No Yahoo suffix for tag {tag} in {t212}")

    return sym + suffix


def _fetch_rates_batch(currencies: tuple[str, ...]) -> dict[str, float]:
    """Fetch currency rates from a reliable API in batch."""
    rates = {}

    # Use exchangerate-api.com (free tier available)
    url = "https://api.exchangerate-api.com/v4/latest/GBP"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Convert from GBP to other currencies (invert the rates)
    for currency in currencies:
        if currency in data["rates"]:
            # Invert the rate since we want TO GBP, not FROM GBP
            rates[currency] = 1.0 / data["rates"][currency]
        else:
            raise KeyError(f"Currency {currency} not found in fallback API response")

    return rates


def update_currency_rates(currencies: tuple[str, ...]) -> dict[str, float]:
    """Get currency exchange rates to GBP with database caching."""
    today = datetime.now(TIMEZONE).date()

    rates = _fetch_rates_batch(currencies)

    with get_session() as session:
        for currency in currencies:
            existing = (
                session.query(CurrencyRateDaily)
                .filter(
                    CurrencyRateDaily.from_currency == currency,
                    CurrencyRateDaily.to_currency == "GBP",
                    CurrencyRateDaily.date == today,
                )
                .first()
            )

            if existing:
                logging.info(f"Updated rate {currency}: {rates[currency]}")
                existing.rate = rates[currency]
                existing.updated_at = datetime.now(TIMEZONE)
            else:
                currency_rate = CurrencyRateDaily(
                    from_currency=currency, to_currency="GBP", rate=rates[currency], date=today
                )
                session.add(currency_rate)

    rates.update({"GBX": 0.01, "GBP": 1.0})

    return rates


# Holdings and Instruments


def request_json(url: str, headers: dict[str, str], retries: int = REQUEST_RETRY) -> TRADING212_API_RESPONSE:
    """Make HTTP request with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            logging.warning(f"HTTP {r.status_code} for {url}: {r.text}")
        except requests.RequestException as exc:
            logging.warning(f"Request error for {url}: {exc}")
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


@lru_cache()
def fetch_holdings() -> dict[str, T212Position]:
    """Fetch portfolio holdings from Trading212 API."""
    logging.info("Fetching portfolio from Trading212 API")
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    raw = cast(
        list[T212Position],
        request_json(url, {"Authorization": TRADING212_API_KEY}),
    )
    holdings = {h["ticker"]: h for h in raw if h["ticker"] not in STOCKS_DELISTED}

    return holdings


def update_holdings() -> list[HoldingDaily]:
    """Update holdings in the database."""
    created = 0
    updated = 0
    result = []
    current_date = datetime.now(TIMEZONE).date()

    holdings = fetch_holdings()

    with get_session() as session:
        instruments = session.query(Instrument).order_by(Instrument.updated_at).all()

        instruments_dict = {i.t212_code: i for i in instruments}  # convert to dict t212_code: instrument
        # Update this data only once a day
        yahoo_datas = get_yahoo_ticker_data(
            [
                i.yahoo_symbol
                for i in instruments
                if i.updated_at > datetime.now() - timedelta(days=YAHOO_UPDATE_INTERVAL_DAYS)
            ][:YAHOO_UPDATE_LIMIT]
        )

        # Delete sold holdings
        deleted = (
            session.query(HoldingDaily)
            .filter(
                HoldingDaily.instrument_id.notin_({i.id for i in instruments if i.t212_code in holdings}),
                HoldingDaily.date == current_date,
            )
            .delete(synchronize_session=False)
        )
        if deleted:
            logging.warning("Deleted %s HoldingDaily", deleted)

        # Update instruments
        for instrument in instruments:
            yahoo_symbol = instrument.yahoo_symbol
            if yahoo_symbol not in yahoo_datas["info"]:
                # This stock was recently updated or wasn't selected for update because of the limit
                continue

            yahoo_data = yahoo_datas["info"][yahoo_symbol]

            # Upsert InstrumentYahoo (detached Yahoo blobs)
            yahoo_row = session.get(InstrumentYahoo, instrument.id)
            if yahoo_row:
                yahoo_row.info = yahoo_data
                yahoo_row.cashflow = yahoo_datas["cashflow"][yahoo_symbol]
                yahoo_row.earnings = yahoo_datas["earnings"][yahoo_symbol]
                # TODO: Keep only 12 - 24 recommendations
                yahoo_row.recommendations = {
                    **yahoo_row.recommendations,
                    **yahoo_datas["recommendations"][yahoo_symbol],
                }
                yahoo_row.analyst_price_targets = yahoo_datas["analyst_price_targets"][yahoo_symbol]
                yahoo_row.splits = yahoo_datas["splits"][yahoo_symbol]
                yahoo_row.news = yahoo_datas["news"][yahoo_symbol]  # TODO: Keep old news?
                yahoo_row.updated_at = datetime.now(TIMEZONE)
            else:
                session.add(
                    InstrumentYahoo(
                        instrument_id=instrument.id,
                        info=yahoo_data,
                        cashflow=yahoo_datas["cashflow"][yahoo_symbol],
                        earnings=yahoo_datas["earnings"][yahoo_symbol],
                        recommendations=yahoo_datas["recommendations"][yahoo_symbol],
                        analyst_price_targets=yahoo_datas["analyst_price_targets"][yahoo_symbol],
                        splits=yahoo_datas["splits"][yahoo_symbol],
                        news=yahoo_datas["news"][yahoo_symbol],
                    )
                )

            # Upsert InstrumentMetricsDaily for this date (market facts)
            metrics_row = (
                session.query(InstrumentMetricsDaily)
                .filter(
                    InstrumentMetricsDaily.instrument_id == instrument.id,
                    InstrumentMetricsDaily.date == current_date,
                )
                .first()
            )
            if metrics_row:
                metrics_row.market_cap = yahoo_data.get("marketCap")
                metrics_row.pe_ratio = yahoo_data.get("trailingPE")
                metrics_row.institutional = yahoo_data.get("heldPercentInstitutions")
                metrics_row.beta = yahoo_data.get("beta")
                metrics_row.updated_at = datetime.now(TIMEZONE)
            else:
                session.add(
                    InstrumentMetricsDaily(
                        instrument_id=instrument.id,
                        date=current_date,
                        market_cap=yahoo_data.get("marketCap"),
                        pe_ratio=yahoo_data.get("trailingPE"),
                        institutional=yahoo_data.get("heldPercentInstitutions"),
                        beta=yahoo_data.get("beta"),
                    )
                )

        # Update holdings
        for t212_code, holding in holdings.items():
            instrument = instruments_dict[t212_code]

            if holding:
                existing_holding = (
                    session.query(HoldingDaily)
                    .filter(HoldingDaily.instrument_id == instrument.id, HoldingDaily.date == current_date)
                    .first()
                )

                if existing_holding:
                    # Update existing holding
                    existing_holding.quantity = holding["quantity"]
                    existing_holding.avg_price = holding["averagePrice"]
                    existing_holding.current_price = holding["currentPrice"]
                    existing_holding.ppl = holding["ppl"]
                    existing_holding.fx_ppl = holding["fxPpl"] or 0
                    existing_holding.updated_at = datetime.now(TIMEZONE)
                    result.append(existing_holding)
                    updated += 1
                else:
                    # Create new holding record
                    new_holding = HoldingDaily(
                        instrument_id=instrument.id,
                        quantity=holding["quantity"],
                        avg_price=holding["averagePrice"],
                        current_price=holding["currentPrice"],
                        ppl=holding["ppl"],
                        fx_ppl=holding["fxPpl"] or 0,
                        date=current_date,
                    )
                    session.add(new_holding)
                    result.append(new_holding)
                    created += 1

    logging.info(f"Created {created} holdings, updated {updated} holdings")

    return result


def update_instruments(isins: set[tuple[str, str]]) -> list[Instrument]:
    """Update instruments in the database from Trading212 API."""
    logging.info("Fetching instruments from Trading212 API")
    instruments = []
    url = "https://live.trading212.com/api/v0/equity/metadata/instruments"
    instruments_from_api = cast(
        list[T212Instrument],
        request_json(url, {"Authorization": TRADING212_API_KEY}),
    )
    holdings_from_api = fetch_holdings()
    tickers = set(holdings_from_api.keys())

    with get_session() as session:
        created = 0
        updated = 0
        for instrument in instruments_from_api:
            if instrument["ticker"] in tickers or (instrument["isin"], instrument["currencyCode"]) in isins:
                existing = session.query(Instrument).filter(Instrument.t212_code == instrument["ticker"]).first()

                try:
                    if existing:
                        existing.name = instrument["name"]
                        existing.currency = instrument["currencyCode"]
                        existing.yahoo_symbol = convert_ticker(instrument["ticker"])
                        existing.isin = instrument["isin"]
                        existing.updated_at = datetime.now(TIMEZONE)
                        instruments.append(existing)
                        updated += 1
                    else:
                        # Create new instrument only
                        new_instrument = Instrument(
                            t212_code=instrument["ticker"],
                            name=instrument["name"],
                            currency=instrument["currencyCode"],
                            yahoo_symbol=convert_ticker(instrument["ticker"]),
                            isin=instrument["isin"],
                        )
                        session.add(new_instrument)
                        instruments.append(new_instrument)
                        created += 1
                except Exception as e:
                    print(instrument["isin"], instrument["ticker"])
                    raise e

    logging.info(f"Created {created} instruments, updated {updated} instruments")

    return instruments


def _update_prices(session: Session, tickers: list[str], start: date) -> None:
    """Update price data from Yahoo Finance."""
    now = datetime.now(TIMEZONE)

    for i in range(0, len(tickers), BATCH_SIZE_YF):
        sub = tickers[i : i + BATCH_SIZE_YF]
        logging.info("Downloading prices (start: %s) for %s", start.strftime("%Y-%m-%d"), sub)

        df = yf.download(
            tickers=sub,
            start=start.strftime("%Y-%m-%d"),
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
        ).rename(columns={"Adj Close": "Adj_Close"})

        # Bulk upsert
        all_price_data = []
        for ticker in df.columns.get_level_values(0).unique():
            tdf = df[ticker].dropna(how="all")  # Open/High/Low/Close/Adj Close/Volume
            for dt, row in tdf.iterrows():
                all_price_data.append(
                    {
                        "symbol": ticker,
                        "date": dt.date(),
                        "open_price": float(row["Open"]),
                        "high_price": float(row["High"]),
                        "low_price": float(row["Low"]),
                        "close_price": float(row["Close"]),
                        "adj_close_price": float(row["Adj_Close"]),
                        "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                        "updated_at": now,
                    }
                )

        if all_price_data:
            insert_stmt = pg_insert(PricesDaily).values(all_price_data)

            # Define what to do ON CONFLICT (i.e., when symbol/date key exists)
            # We update the existing row with the new values
            on_conflict_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["symbol", "date"],  # This is the UniqueConstraint
                set_={
                    "open_price": insert_stmt.excluded.open_price,
                    "high_price": insert_stmt.excluded.high_price,
                    "low_price": insert_stmt.excluded.low_price,
                    "close_price": insert_stmt.excluded.close_price,
                    "adj_close_price": insert_stmt.excluded.adj_close_price,
                    "volume": insert_stmt.excluded.volume,
                    "updated_at": insert_stmt.excluded.updated_at,
                },
            )

            # Execute the single bulk statement
            try:
                session.execute(on_conflict_stmt)
            except Exception as e:
                logging.error(f"Failed to bulk upsert prices for {sub}: {e}")
                session.rollback()  # Rollback this batch


def update_prices(tickers_to_add: set[str]) -> None:
    """Get and update price data for all tickers."""
    today = datetime.now(TIMEZONE).date()

    with get_session() as session:
        tickers = set(session.scalars(select(Instrument.yahoo_symbol)).all())

        existing_prices = (
            session.query(
                PricesDaily.symbol,
                func.max(PricesDaily.date).label("max_date"),
            )
            .where(PricesDaily.symbol.notin_(STOCKS_DELISTED))
            .group_by(PricesDaily.symbol)
            .all()
        )

        existing_tickers = set(row.symbol for row in existing_prices)
        if existing_tickers:
            existing_tickers -= STOCKS_DELISTED
            latest_date = min([row.max_date for row in existing_prices])
            start = latest_date + timedelta(days=1)

            # Get prices for existing tickers
            if start <= (today - timedelta(days=[3, 1, 1, 1, 1, 1, 2][today.weekday()])):
                _update_prices(session, list(existing_tickers), start)

        # Get prices for new tickers
        new_tickers = list(tickers_to_add | tickers - existing_tickers - STOCKS_DELISTED)
        if new_tickers:
            start = today - timedelta(days=HISTORY_YEARS * 366)  # 10 years of data
            _update_prices(session, new_tickers, start)


def scrub_for_json(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    if isinstance(obj, dict):
        return {k.date().isoformat() if isinstance(k, datetime) else str(k): scrub_for_json(v) for k, v in obj.items()}

    return obj


def get_yahoo_ticker_data(symbols: list[str]) -> YahooData:
    """Update Yahoo Finance data for all holdings."""
    logging.info(f"Fetching {len(symbols)} Yahoo Finance profiles")
    yahoo_data: YahooData = {
        "info": {},
        "cashflow": defaultdict(dict),
        "earnings": defaultdict(dict),
        "recommendations": defaultdict(dict),
        "analyst_price_targets": defaultdict(dict),
        "splits": defaultdict(dict),
        "news": defaultdict(dict),
    }

    # Fetch in batches
    for i in range(0, len(symbols), BATCH_SIZE_YF):
        batch = symbols[i : i + BATCH_SIZE_YF]

        # Use yfinance's batch download capability
        tickers = yf.Tickers(" ".join(batch))
        for symbol in batch:
            try:
                yahoo_data["info"][symbol] = tickers.tickers[symbol].info

                if yahoo_data["info"][symbol].get("quoteType") == "ETF":
                    # ETF don't have cashflow, earnings, recommendations, analyst_price_targets, splits
                    continue

                data = tickers.tickers[symbol].cashflow.to_dict()
                yahoo_data["cashflow"][symbol] = scrub_for_json(data)

                try:
                    data = tickers.tickers[symbol].get_earnings_dates(limit=40)
                except KeyError as e:
                    logging.warning("Failed to get earnings for %s: %s", symbol, e)
                    continue

                if data is None:
                    # ETFs don't have earnings
                    yahoo_data["earnings"][symbol] = {}
                else:
                    yahoo_data["earnings"][symbol] = scrub_for_json(data.to_dict(orient="index"))

                data = tickers.tickers[symbol].recommendations.to_dict(orient="index")
                data = {
                    datetime.now(TIMEZONE).date().replace(day=1)
                    + relativedelta(months=int(v.pop("period").rstrip("m"))): v
                    for k, v in data.items()
                }
                yahoo_data["recommendations"][symbol] = scrub_for_json(data)

                yahoo_data["analyst_price_targets"][symbol] = tickers.tickers[symbol].analyst_price_targets

                # Fetch and store splits
                splits_data = tickers.tickers[symbol].splits.to_dict()
                yahoo_data["splits"][symbol] = scrub_for_json(splits_data)

                # Fetch and store news
                yahoo_data["news"][symbol] = tickers.tickers[symbol].get_news()
            except ValueError as e:
                logging.warning(f"Problem with parsing DataFrame {symbol}: {e}")

    return yahoo_data


def get_rates(session: Session) -> dict[str, float]:
    """Get current currency exchange rates to GBP."""
    table = {"GBX": 0.01, "GBP": 1.0, "GBp": 0.01}

    result = session.execute(
        select(CurrencyRateDaily.from_currency, CurrencyRateDaily.rate).filter(
            CurrencyRateDaily.from_currency.in_(CURRENCIES),
            CurrencyRateDaily.to_currency == "GBP",
            CurrencyRateDaily.date == datetime.now(TIMEZONE).date(),
        )
    )
    rates = result.all()
    for currency, rate in rates:
        table[currency] = rate

    return table


def get_portfolio_allocation(
    session: Session, portfolio_value: float, snapshot_date: date
) -> dict[str, dict[str, float]]:
    """Get portfolio allocation per country, sector, currency and etf/equity split"""
    allocations: dict[str, dict[str, float]] = {
        "country": defaultdict(float),
        "sector": defaultdict(float),
        "currency": defaultdict(float),
        "etf_equity": defaultdict(float),
    }

    rates = get_rates(session)

    holdings_result = session.execute(
        select(HoldingDaily)
        .join(Instrument)
        .filter(HoldingDaily.date == snapshot_date)
        .order_by(Instrument.name)
        .options(selectinload(HoldingDaily.instrument).selectinload(Instrument.yahoo))
    )
    holdings = holdings_result.scalars().all()

    for h in holdings:
        instrument = h.instrument
        yahoo_symbol = instrument.yahoo_symbol
        info = instrument.yahoo.info

        gbp_value = h.quantity * h.current_price * rates[instrument.currency]
        allocations["etf_equity"][info["quoteType"]] += gbp_value
        allocations["currency"][instrument.currency] += gbp_value

        if info["quoteType"] == "EQUITY":
            allocations["country"][info.get("country") or "Other"] += gbp_value
            allocations["sector"][info.get("sector") or "Other"] += gbp_value
        elif info["quoteType"] == "ETF":
            for country, percent in ETF_COUNTRY_ALLOCATION[yahoo_symbol].items():
                allocations["country"][country] += gbp_value * percent / 100
            for sector, percent in ETF_SECTOR_ALLOCATION[yahoo_symbol].items():
                allocations["sector"][sector] += gbp_value * percent / 100
        else:
            raise ValueError(f"Unknown quoteType '{info['quoteType']}' for {yahoo_symbol}")

    allocations["currency"]["GBP"] += allocations["currency"].pop("GBX", 0.0)
    for country in allocations["country"]:
        allocations["country"][country] = round(100 * allocations["country"][country] / portfolio_value, 2)
    for sector in allocations["sector"]:
        allocations["sector"][sector] = round(100 * allocations["sector"][sector] / portfolio_value, 2)
    for currency in allocations["currency"]:
        allocations["currency"][currency] = round(100 * allocations["currency"][currency] / portfolio_value, 2)
    for quote_type in allocations["etf_equity"]:
        allocations["etf_equity"][quote_type] = round(100 * allocations["etf_equity"][quote_type] / portfolio_value, 2)

    return allocations


def update_portfolio():
    """Update portfolio."""
    logging.info("Fetching portfolio from Trading212 API")
    snapshot_date = datetime.now(TIMEZONE).date()
    url = "https://live.trading212.com/api/v0/equity/account/cash"
    portfolio_from_api = cast(
        T212Portfolio,
        request_json(url, {"Authorization": TRADING212_API_KEY}),
    )
    cash = sum(
        [
            portfolio_from_api["free"] or 0,
            portfolio_from_api["pieCash"] or 0,
            portfolio_from_api["blocked"] or 0,
        ]
    )

    with get_session() as session:
        # Check if snapshot already exists for this date
        existing_snapshot = session.query(PortfolioDaily).filter(PortfolioDaily.date == snapshot_date).first()
        allocations = get_portfolio_allocation(session, portfolio_from_api["total"] - cash, snapshot_date)
        risk_metrics = calculate_portfolio_risk_metrics(session, snapshot_date)

        if existing_snapshot:
            # Update existing snapshot
            existing_snapshot.value = portfolio_from_api["total"]
            existing_snapshot.unrealised_profit = portfolio_from_api["ppl"]
            existing_snapshot.realised_profit = portfolio_from_api["result"]
            existing_snapshot.cash = cash
            existing_snapshot.invested = portfolio_from_api["invested"]
            existing_snapshot.country_allocation = allocations["country"]
            existing_snapshot.sector_allocation = allocations["sector"]
            existing_snapshot.currency_allocation = allocations["currency"]
            existing_snapshot.etf_equity_split = allocations["etf_equity"]
            existing_snapshot.sharpe_ratio = risk_metrics["sharpe"]
            existing_snapshot.sortino_ratio = risk_metrics["sortino"]
            existing_snapshot.beta = risk_metrics["beta"]
            existing_snapshot.updated_at = datetime.now(TIMEZONE)
        else:
            # Create new snapshot
            snapshot = PortfolioDaily(
                date=snapshot_date,
                value=portfolio_from_api["total"],
                unrealised_profit=portfolio_from_api["ppl"],
                realised_profit=portfolio_from_api["result"],
                cash=cash,
                invested=portfolio_from_api["invested"],
                country_allocation=allocations["country"],
                sector_allocation=allocations["sector"],
                currency_allocation=allocations["currency"],
                etf_equity_split=allocations["etf_equity"],
                sharpe_ratio=risk_metrics["sharpe"],
                sortino_ratio=risk_metrics["sortino"],
                beta=risk_metrics["beta"],
            )
            session.add(snapshot)


def calculate_portfolio_risk_metrics(session: Session, snapshot_date: date) -> dict[str, Optional[float]]:
    """Calculates Sharpe, Sortino, and Beta for the portfolio."""

    # Look back up to 1 year for calculations
    start_date = snapshot_date - timedelta(days=365)

    # 1. Get historical portfolio values
    portfolio_history = (
        session.query(PortfolioDaily.date, PortfolioDaily.value)
        .filter(PortfolioDaily.date >= start_date, PortfolioDaily.date <= snapshot_date)
        .order_by(PortfolioDaily.date)
        .all()
    )

    if len(portfolio_history) < 2:
        return {"sharpe": None, "sortino": None, "beta": None}

    # 2. Get historical benchmark prices (e.g., S&P 500)
    benchmark_symbol = SPY
    benchmark_prices = (
        session.query(PricesDaily.date, PricesDaily.close_price)
        .filter(
            PricesDaily.symbol == benchmark_symbol, PricesDaily.date >= start_date, PricesDaily.date <= snapshot_date
        )
        .order_by(PricesDaily.date)
        .all()
    )

    if len(benchmark_prices) < 2:
        return {"sharpe": None, "sortino": None, "beta": None}

    benchmark_df = pd.DataFrame(benchmark_prices, columns=["date", "price"]).set_index("date")
    portfolio_df = pd.DataFrame(portfolio_history, columns=["date", "value"]).set_index("date")

    # Re-index the daily portfolio data to match the benchmark's trading-day index.
    # 'ffill' ensures that if a portfolio snapshot was missed, we use the last known value.
    portfolio_df = portfolio_df.reindex(benchmark_df.index, method="ffill")

    # Now, calculate returns on the *aligned* price/value data.
    # Both series now represent returns from one trading day to the next.
    aligned_portfolio = portfolio_df["value"].pct_change().dropna()
    aligned_benchmark = benchmark_df["price"].pct_change().dropna()

    if aligned_portfolio.empty or len(aligned_portfolio) < 2:
        return {"sharpe": None, "sortino": None, "beta": None}

    # --- DYNAMIC ANNUALIZATION FACTOR ---
    trading_days_per_year = 252
    num_days = len(aligned_portfolio)  # This is now the number of *trading days*
    annualization_factor = np.sqrt(trading_days_per_year) if num_days > 60 else 1.0

    # Assume a risk-free rate (e.g., 4% annually for UK)
    risk_free_rate_annual = 0.04
    risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1 / trading_days_per_year) - 1

    excess_returns = aligned_portfolio - risk_free_rate_daily

    # --- CALCULATIONS ---

    # Sharpe Ratio
    sharpe_ratio = (
        float((excess_returns.mean() / excess_returns.std()) * annualization_factor)
        if excess_returns.std() != 0
        else None
    )

    # Sortino Ratio
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (
        float((excess_returns.mean() / downside_std) * annualization_factor)
        if downside_std != 0 and downside_std is not np.nan
        else None
    )

    # Beta
    covariance = aligned_portfolio.cov(aligned_benchmark)
    variance = cast(float, aligned_benchmark.var())
    beta = float(covariance / variance) if variance != 0 else None

    return {"sharpe": sharpe_ratio, "sortino": sortino_ratio, "beta": beta}


def update_data():
    """Update all data in the database."""
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting data update process")

    # 1. Update rates
    update_currency_rates(CURRENCIES)

    # 2. Update instruments
    _isins_to_add: set[tuple[str, str]] = set()
    update_instruments(_isins_to_add)

    # 3. Update holdings
    update_holdings()

    # 4. Update prices
    _tickers_to_add: set[str] = set()  # "^VIX"
    update_prices(_tickers_to_add)

    # 5. Update portfolio
    update_portfolio()


if __name__ == "__main__":
    update_data()
