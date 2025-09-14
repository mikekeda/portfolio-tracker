"""
Data Update Script for Trading212 Portfolio Manager
==================================================
Updates all database tables with fresh data from Trading212 API and Yahoo Finance.
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, Generator, List, Literal, Set, Tuple, TypeAlias, TypedDict, Union, cast

import pandas as pd
import requests
import yfinance as yf  # type: ignore[import-untyped]
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import func

from config import BATCH_SIZE_YF, CURRENCIES, HISTORY_YEARS, PATTERN_MULTI, REQUEST_RETRY, TRADING212_API_KEY, TIMEZONE
from data import ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION, STOCKS_ALIASES, STOCKS_DELISTED, STOCKS_SUFFIX
from models import CurrencyRateDaily, HoldingDaily, Instrument, PortfolioDaily, PricesDaily


# GET /api/v0/equity/metadata/instruments
class T212Instrument(TypedDict):
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


TRADING212_API_RESPONSE: TypeAlias = List[Union[T212Instrument, T212Position]]


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker:
    """Create and cache the database engine and session factory."""
    db_url = "postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}".format(
        db_name=os.getenv("DB_NAME", "trading212_portfolio"),
        db_password=os.getenv("DB_PASSWORD"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
    )
    engine = create_engine(db_url, echo=False, pool_pre_ping=True)

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
    elif core[-1].islower():  # single‑letter tag
        sym, tag = core[:-1], core[-1]
    else:
        raise ValueError(f"Cannot parse: {t212}")

    sym = sym.rstrip("_")
    sym = STOCKS_ALIASES.get(sym, sym)
    suffix = STOCKS_SUFFIX.get(tag)
    if suffix is None:
        raise ValueError(f"No Yahoo suffix for tag {tag} in {t212}")

    return sym + suffix


def _fetch_rates_batch(currencies: Tuple[str, ...]) -> Dict[str, float]:
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


def get_currency_table(currencies: Tuple[str, ...]) -> Dict[str, float]:
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


def request_json(url: str, headers: Dict[str, str], retries: int = REQUEST_RETRY) -> TRADING212_API_RESPONSE:
    """Make HTTP request with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            logging.warning(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as exc:
            logging.warning(f"Request error for {url}: {exc}")
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


def fetch_holdings() -> List[T212Position]:
    """Fetch portfolio holdings from Trading212 API."""
    logging.info("Fetching portfolio from Trading212 API")
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    raw = cast(
        List[T212Position],
        request_json(url, {"Authorization": TRADING212_API_KEY}),
    )
    holdings = [h for h in raw if h["ticker"] not in STOCKS_DELISTED]

    return holdings


def update_holdings(
    holdings: List[T212Position], instruments: List[Instrument], yahoo_datas: Dict[str, Any]
) -> List[HoldingDaily]:
    """Update holdings in the database."""
    created = 0
    updated = 0
    result = []
    current_date = datetime.now(TIMEZONE).date()
    instruments_dict = {i.t212_code: i for i in instruments}  # convert to dict t212_code: instrument

    with get_session() as session:
        for holding in holdings:
            yahoo_symbol = instruments_dict[holding["ticker"]].yahoo_symbol
            yahoo_data = yahoo_datas[yahoo_symbol]

            stmt = (
                update(Instrument)
                .where(Instrument.t212_code == holding["ticker"])
                .values(
                    sector=yahoo_data.get("sector"),
                    country=yahoo_data.get("country"),
                    yahoo_data=yahoo_data,
                    updated_at=datetime.now(TIMEZONE),
                )
                .returning(Instrument.id)
            )
            instrument_id = session.execute(stmt).scalar_one()

            existing_holding = (
                session.query(HoldingDaily)
                .filter(HoldingDaily.instrument_id == instrument_id, HoldingDaily.date == current_date)
                .first()
            )

            if existing_holding:
                # Update existing holding
                existing_holding.quantity = holding["quantity"]
                existing_holding.avg_price = holding["averagePrice"]
                existing_holding.current_price = holding["currentPrice"]
                existing_holding.ppl = holding["ppl"]
                existing_holding.fx_ppl = holding["fxPpl"] or 0
                existing_holding.market_cap = yahoo_data.get("marketCap")
                existing_holding.pe_ratio = yahoo_data.get("trailingPE")
                existing_holding.beta = yahoo_data.get("beta")
                existing_holding.institutional = yahoo_data.get("heldPercentInstitutions")
                existing_holding.updated_at = datetime.now(TIMEZONE)
                result.append(existing_holding)
                updated += 1
            else:
                # Create new holding record
                new_holding = HoldingDaily(
                    instrument_id=instrument_id,
                    quantity=holding["quantity"],
                    avg_price=holding["averagePrice"],
                    current_price=holding["currentPrice"],
                    ppl=holding["ppl"],
                    fx_ppl=holding["fxPpl"] or 0,
                    market_cap=yahoo_data.get("marketCap"),
                    pe_ratio=yahoo_data.get("trailingPE"),
                    institutional=yahoo_data.get("heldPercentInstitutions"),
                    beta=yahoo_data.get("beta"),
                    date=current_date,
                )
                session.add(new_holding)
                result.append(new_holding)
                created += 1

    logging.info(f"Created {created} holdings, updated {updated} holdings")

    return result


def update_instruments(tickers: Set[str]) -> List[Instrument]:
    """Update instruments in the database from Trading212 API."""
    logging.info("Fetching instruments from Trading212 API")
    instruments = []
    url = "https://live.trading212.com/api/v0/equity/metadata/instruments"
    instruments_from_api = cast(
        List[T212Instrument],
        request_json(url, {"Authorization": TRADING212_API_KEY}),
    )

    with get_session() as session:
        created = 0
        updated = 0
        for instrument in instruments_from_api:
            if instrument["ticker"] in tickers:
                existing = session.query(Instrument).filter(Instrument.t212_code == instrument["ticker"]).first()

                if existing:
                    existing.name = instrument["name"]
                    existing.currency = instrument["currencyCode"]
                    existing.yahoo_symbol = convert_ticker(instrument["ticker"])
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
                    )
                    session.add(new_instrument)
                    instruments.append(new_instrument)
                    created += 1

    logging.info(f"Created {created} instruments, updated {updated} instruments")

    return instruments


def update_prices(session: Session, tickers: List[str], start: date) -> None:
    """Update price data from Yahoo Finance."""
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
        for ticker in df.columns.get_level_values(0).unique():
            tdf = df[ticker].dropna(how="all")  # Open/High/Low/Close/Adj Close/Volume
            for dt, row in tdf.iterrows():
                existing = (
                    session.query(PricesDaily)
                    .filter(PricesDaily.symbol == ticker, PricesDaily.date == dt.date())
                    .first()
                )

                if existing:
                    # Update existing record
                    existing.open_price = float(row["Open"])
                    existing.high_price = float(row["High"])
                    existing.low_price = float(row["Low"])
                    existing.close_price = float(row["Close"])
                    existing.adj_close_price = float(row["Adj_Close"])
                    existing.volume = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
                    existing.updated_at = datetime.now(TIMEZONE)
                else:
                    # Insert new record
                    prices = PricesDaily(
                        symbol=ticker,
                        date=dt.date(),
                        open_price=float(row["Open"]),
                        high_price=float(row["High"]),
                        low_price=float(row["Low"]),
                        close_price=float(row["Close"]),
                        adj_close_price=float(row["Adj_Close"]),
                        volume=int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    )
                    session.add(prices)


def get_and_update_prices(tickers: Set[str]) -> None:
    """Get and update price data for all tickers."""
    today = datetime.now(TIMEZONE).date()

    with get_session() as session:
        existing_prices = (
            session.query(
                PricesDaily.symbol,
                func.max(PricesDaily.date).label("max_date"),
            )
            .group_by(PricesDaily.symbol)
            .all()
        )

        existing_tickers = set(row.symbol for row in existing_prices)
        if existing_tickers:
            latest_date = min([row.max_date for row in existing_prices])
            start = latest_date + timedelta(days=1)

            # Get prices for existing tickers
            if start <= (today - timedelta(days=[3, 1, 1, 1, 1, 1, 2][today.weekday()])):
                update_prices(session, list(existing_tickers), start)

        # Get prices for new tickers
        new_tickers = list(tickers - existing_tickers)
        if new_tickers:
            start = today - timedelta(days=HISTORY_YEARS * 366)  # 10 years of data
            update_prices(session, new_tickers, start)


def get_yahoo_ticker_data(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Update Yahoo Finance data for all holdings."""
    logging.info(f"Fetching {len(symbols)} Yahoo Finance profiles")
    yahoo_data = {}

    # Fetch in batches
    for i in range(0, len(symbols), BATCH_SIZE_YF):
        batch = symbols[i : i + BATCH_SIZE_YF]

        # Use yfinance's batch download capability
        tickers = yf.Tickers(" ".join(batch))
        for symbol in batch:
            info = tickers.tickers[symbol].info
            yahoo_data[symbol] = info

    return yahoo_data


def save_portfolio_snapshots(
    holdings: List[HoldingDaily], instruments: List[Instrument], rates: Dict[str, float], yahoo_data: Dict[str, Any]
) -> None:
    """Save portfolio snapshot to database."""
    snapshot_date = datetime.now(TIMEZONE).date()
    total_value = 0.0
    total_profit = 0.0
    country_allocation: Dict[str, float] = defaultdict(float)
    sector_allocation: Dict[str, float] = defaultdict(float)
    currency_allocation: Dict[str, float] = defaultdict(float)
    etf_equity_allocation: Dict[str, float] = defaultdict(float)

    instruments_dict = {i.id: i for i in instruments}

    for h in holdings:
        instrument = instruments_dict[h.instrument_id]
        yahoo_symbol = instrument.yahoo_symbol
        info = yahoo_data[yahoo_symbol]

        gbp_value = h.quantity * h.current_price * rates[instrument.currency]
        total_value += gbp_value
        total_profit += h.ppl
        etf_equity_allocation[info["quoteType"]] += gbp_value
        currency_allocation[info.get("currency") or "Other"] += gbp_value

        if info["quoteType"] == "EQUITY":
            country_allocation[info.get("country") or "Other"] += gbp_value
            sector_allocation[info.get("sector") or "Other"] += gbp_value
        elif info["quoteType"] == "ETF":
            for country, percent in ETF_COUNTRY_ALLOCATION[yahoo_symbol].items():
                country_allocation[country] += gbp_value * percent / 100
            for sector, percent in ETF_SECTOR_ALLOCATION[yahoo_symbol].items():
                sector_allocation[sector] += gbp_value * percent / 100
        else:
            raise ValueError(f"Unknown quoteType '{info['quoteType']}' for {yahoo_symbol}")

    return_pct = total_profit / (total_value - total_profit) * 100.0 if total_value != total_profit else 0.0
    for country in country_allocation:
        country_allocation[country] = round(100 * country_allocation[country] / total_value, 2)
    for sector in sector_allocation:
        sector_allocation[sector] = round(100 * sector_allocation[sector] / total_value, 2)
    for currency in currency_allocation:
        currency_allocation[currency] = round(100 * currency_allocation[currency] / total_value, 2)
    for quote_type in etf_equity_allocation:
        etf_equity_allocation[quote_type] = round(100 * etf_equity_allocation[quote_type] / total_value, 2)

    currency_allocation["GBP"] += currency_allocation.pop("GBp", 0.0)

    with get_session() as session:
        # Check if snapshot already exists for this date
        existing_snapshot = session.query(PortfolioDaily).filter(PortfolioDaily.date == snapshot_date).first()

        if existing_snapshot:
            # Update existing snapshot
            existing_snapshot.total_value_gbp = total_value
            existing_snapshot.total_profit_gbp = total_profit
            existing_snapshot.total_return_pct = return_pct
            existing_snapshot.country_allocation = country_allocation
            existing_snapshot.sector_allocation = sector_allocation
            existing_snapshot.currency_allocation = currency_allocation
            existing_snapshot.etf_equity_split = etf_equity_allocation
            existing_snapshot.updated_at = datetime.now(TIMEZONE)
        else:
            # Create new snapshot
            snapshot = PortfolioDaily(
                date=snapshot_date,
                total_value_gbp=total_value,
                total_profit_gbp=total_profit,
                total_return_pct=return_pct,
                country_allocation=country_allocation,
                sector_allocation=sector_allocation,
                currency_allocation=currency_allocation,
                etf_equity_split=etf_equity_allocation,
            )
            session.add(snapshot)


def update_holdings_and_instruments(rates: Dict[str, float]) -> None:
    """Update holdings and instruments in the database."""
    holdings_from_api = fetch_holdings()
    t212_codes = set(h["ticker"] for h in holdings_from_api)

    # Update instruments
    instruments = update_instruments(t212_codes)
    yahoo_symbols = set(i.yahoo_symbol for i in instruments)

    # Update prices
    get_and_update_prices(yahoo_symbols)

    yahoo_data = get_yahoo_ticker_data(list(yahoo_symbols))
    holdings = update_holdings(holdings_from_api, instruments, yahoo_data)

    save_portfolio_snapshots(holdings, instruments, rates, yahoo_data)


if __name__ == "__main__":
    """update all data in the database."""
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting data update process")

    # 1. Update rates
    _rates = get_currency_table(CURRENCIES)

    # 2. Update holdings and instruments
    update_holdings_and_instruments(_rates)
