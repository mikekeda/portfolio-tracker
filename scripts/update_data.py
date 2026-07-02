"""
Data Update Script for Trading212 Portfolio Manager
==================================================
Updates all database tables with fresh data from Trading212 API and Yahoo Finance.
"""

import concurrent.futures
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
from sqlalchemy import case, create_engine, select
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
    DAYS_PER_YEAR,
    RISK_FREE_RATE,
    TRADING212_API_KEY,
    logger,
)
from utils.insider import compute_insider_signal
from data import ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION, STOCKS_ALIASES, STOCKS_DELISTED, STOCKS_SUFFIX
from models import (
    CurrencyRateDaily,
    HoldingDaily,
    Instrument,
    InstrumentMetricsDaily,
    InstrumentYahoo,
    PortfolioDaily,
    PricesDaily,
    TransactionAction,
    TransactionHistory,
)

# yfinance can emit very noisy per-symbol error logs on known delisted/legacy tickers.
# We keep our own aggregated error reporting and suppress third-party spam.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("yfinance.shared").setLevel(logging.CRITICAL)
logging.getLogger("yfinance.multi").setLevel(logging.CRITICAL)


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
    cashflow: dict[str, dict[str, Optional[int]]]
    earnings: dict[str, dict[str, Optional[float]]]
    recommendations: dict[str, dict[str, int]]
    analyst_price_targets: dict[str, float]
    splits: dict[str, float]
    news: list[dict[str, Any]]
    balance_sheet: dict[str, dict[str, Optional[int]]]
    income_stmt: dict[str, dict[str, Optional[int]]]
    insider_transactions: list[dict[str, Any]]


TRADING212_API_RESPONSE: TypeAlias = list[Union[T212Instrument, T212Position]]

# Sized so the weekday schedule covers the whole universe daily:
# 9 runs/day x 150 = 1350 refreshes vs ~1035 tracked instruments. Each profile
# is ~8 Yahoo endpoints, so bigger batches raise the 429/rate-limit risk —
# watch for "Empty Yahoo info" warnings before raising further.
YAHOO_UPDATE_LIMIT = 150
YAHOO_UPDATE_INTERVAL_DAYS = 1

_BAD_NUMERIC_STRINGS = frozenset({"infinity", "-infinity", "inf", "-inf", "nan"})


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
                logger.info("Updated rate %s: %s", currency, rates[currency])
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
            logger.warning("HTTP %s for %s: %s", r.status_code, url, r.text)
        except requests.RequestException as exc:
            logger.warning("Request error for %s: %s", url, exc)
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


@lru_cache()
def fetch_holdings() -> dict[str, T212Position]:
    """Fetch portfolio holdings from Trading212 API."""
    logger.info("Fetching portfolio from Trading212 API")
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

        # Prioritize instruments that have NO InstrumentYahoo (backfill) - these cause 500s on portfolio page.
        # Then include instruments whose InstrumentYahoo row is older than the refresh cutoff.
        # Cap at YAHOO_UPDATE_LIMIT.
        instruments_without_yahoo_ids: set[int] = set(
            session.execute(
                select(Instrument.id)
                .outerjoin(InstrumentYahoo, Instrument.id == InstrumentYahoo.instrument_id)
                .where(InstrumentYahoo.instrument_id.is_(None))
                .where(Instrument.yahoo_symbol.isnot(None))
                .where(Instrument.yahoo_symbol.notin_(STOCKS_DELISTED))
            )
            .scalars()
            .all()
        )
        cutoff = datetime.now(TIMEZONE) - timedelta(days=YAHOO_UPDATE_INTERVAL_DAYS)
        stale_yahoo_ids: list[int] = list(
            session.execute(
                select(InstrumentYahoo.instrument_id)
                .join(Instrument, Instrument.id == InstrumentYahoo.instrument_id)
                .where(InstrumentYahoo.updated_at < cutoff)
                .where(Instrument.yahoo_symbol.notin_(STOCKS_DELISTED))
                .order_by(InstrumentYahoo.updated_at)
            )
            .scalars()
            .all()
        )
        # ~20x more instruments are tracked than held, so with a budget of
        # YAHOO_UPDATE_LIMIT per run current holdings go first (stable sort
        # keeps oldest-first order within each group).
        held_instrument_ids = {i.id for i in instruments if i.t212_code in holdings}
        stale_yahoo_ids.sort(key=lambda inst_id: inst_id not in held_instrument_ids)

        id_to_yahoo_symbol: dict[int, str] = {i.id: i.yahoo_symbol for i in instruments if i.yahoo_symbol}
        symbols_to_fetch: list[str] = []
        symbols_seen: set[str] = set()
        for inst_id in list(instruments_without_yahoo_ids) + stale_yahoo_ids:
            if len(symbols_to_fetch) >= YAHOO_UPDATE_LIMIT:
                break
            sym = id_to_yahoo_symbol.get(inst_id)
            if sym and sym not in symbols_seen:
                symbols_to_fetch.append(sym)
                symbols_seen.add(sym)

        if instruments_without_yahoo_ids:
            logger.info(
                "Backfilling InstrumentYahoo for %d instruments missing Yahoo data",
                len(instruments_without_yahoo_ids),
            )
        yahoo_datas = get_yahoo_ticker_data(symbols_to_fetch)

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
            logger.warning("Deleted %s HoldingDaily", deleted)

        # Update instruments
        for instrument in instruments:
            yahoo_symbol = str(instrument.yahoo_symbol)
            if yahoo_symbol not in yahoo_datas:
                # This stock was recently updated or wasn't selected for update because of the limit
                continue

            payload = yahoo_datas[yahoo_symbol]
            yahoo_data = payload["info"]

            # Upsert InstrumentYahoo (detached Yahoo blobs)
            yahoo_row = session.get(InstrumentYahoo, instrument.id)
            if not yahoo_data:
                # A transient yfinance failure returns empty payloads — keep the
                # cached blobs (and don't create a row with an empty info, which
                # would break quoteType lookups downstream).
                logger.warning("Empty Yahoo info for %s — keeping cached data", yahoo_symbol)
                if yahoo_row:
                    yahoo_row.updated_at = datetime.now(TIMEZONE)
                continue
            if yahoo_row:
                yahoo_row.info = yahoo_data
                # Fetch failures past the info stage (e.g. the flaky
                # get_earnings_dates) return empty dicts — don't wipe cached data.
                if payload["cashflow"]:
                    yahoo_row.cashflow = payload["cashflow"]
                if payload["earnings"]:
                    yahoo_row.earnings = payload["earnings"]
                # TODO: Keep only 12 - 24 recommendations
                yahoo_row.recommendations = {
                    **yahoo_row.recommendations,
                    **payload["recommendations"],
                }
                if payload["analyst_price_targets"]:
                    yahoo_row.analyst_price_targets = payload["analyst_price_targets"]
                if payload["splits"]:
                    yahoo_row.splits = payload["splits"]
                if payload["news"]:
                    yahoo_row.news = payload["news"]  # TODO: Keep old news?
                if payload["balance_sheet"]:
                    yahoo_row.balance_sheet = payload["balance_sheet"]
                if payload["income_stmt"]:
                    yahoo_row.income_stmt = payload["income_stmt"]
                yahoo_row.updated_at = datetime.now(TIMEZONE)
            else:
                session.add(
                    InstrumentYahoo(
                        instrument_id=instrument.id,
                        info=yahoo_data,
                        cashflow=payload["cashflow"],
                        earnings=payload["earnings"],
                        recommendations=payload["recommendations"],
                        analyst_price_targets=payload["analyst_price_targets"],
                        splits=payload["splits"],
                        news=payload["news"],
                        pes={},  # Populated by scrape_wisesheets_pe / scrape_macrotrends_pe
                        balance_sheet=payload["balance_sheet"],
                        income_stmt=payload["income_stmt"],
                    )
                )

            # Upsert InstrumentMetricsDaily for this date (market facts)
            insider_signal = compute_insider_signal(
                payload.get("insider_transactions"),
                current_date,
            )
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
                metrics_row.insider_buy_count_90d = insider_signal["buy_count"]
                metrics_row.insider_sell_count_90d = insider_signal["sell_count"]
                metrics_row.insider_net_value_90d = insider_signal["net_value"]
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
                        insider_buy_count_90d=insider_signal["buy_count"],
                        insider_sell_count_90d=insider_signal["sell_count"],
                        insider_net_value_90d=insider_signal["net_value"],
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

    logger.info("Created %s holdings, updated %s holdings", created, updated)

    return result


def update_instruments(isins: set[tuple[str, str]]) -> list[Instrument]:
    """Update instruments in the database from Trading212 API."""
    logger.info("Fetching instruments from Trading212 API")
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
                except Exception:
                    logger.exception(
                        "Failed to upsert instrument (isin=%s, ticker=%s)",
                        instrument.get("isin"),
                        instrument.get("ticker"),
                    )
                    raise

    logger.info("Created %s instruments, updated %s instruments", created, updated)

    return instruments


def _update_prices(session: Session, tickers: list[str], start: date) -> None:
    """Update price data from Yahoo Finance."""
    if not tickers:
        return None

    now = datetime.now(TIMEZONE)

    preview = tickers[:10]
    logger.info(
        "Downloading prices (start: %s) for %d tickers (first 10: %s)",
        start.strftime("%Y-%m-%d"),
        len(tickers),
        preview,
    )

    df = yf.download(
        tickers=tickers,
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
            logger.error("Failed to bulk upsert prices for %s: %s", tickers, e)
            session.rollback()  # Rollback this batch


def update_prices(tickers_to_add: set[str]) -> None:
    """Get and update price data for all tickers."""

    today = datetime.now(TIMEZONE).date()

    with get_session() as session:
        tickers = {
            t for t in session.scalars(select(Instrument.yahoo_symbol)).all()
            if t and t not in STOCKS_DELISTED
        }

        existing_prices = (
            session.query(
                PricesDaily.symbol,
                func.max(PricesDaily.date).label("max_date"),
            )
            .where(PricesDaily.symbol.notin_(STOCKS_DELISTED))
            .group_by(PricesDaily.symbol)
            .order_by(func.max(PricesDaily.date))
            .all()
        )

        # Get prices for existing tickers. Commit per batch so a later batch
        # failing (and triggering the rollback in _update_prices) doesn't
        # discard prices already upserted in earlier batches.
        for i in range(0, len(existing_prices), BATCH_SIZE_YF):
            sub = existing_prices[i : i + BATCH_SIZE_YF]
            start = min([row.max_date for row in sub]) + timedelta(days=1)
            if start > (today - timedelta(days=[3, 1, 1, 1, 1, 1, 2][today.weekday()])):
                break
            existing_tickers: list[str] = [row.symbol for row in sub]
            _update_prices(session, existing_tickers, start)
            session.commit()

        # Get prices for new tickers
        new_tickers = list(tickers_to_add | (tickers - {row.symbol for row in existing_prices}))
        start = today - timedelta(days=HISTORY_YEARS * 366)  # 10 years of data
        _update_prices(session, new_tickers, start)  # all new tickers at once
        session.commit()


def scrub_for_json(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # Yahoo sometimes returns "Infinity"/"NaN" as strings for ratios with near-zero denominators.
    if isinstance(obj, str) and obj.strip().lower() in _BAD_NUMERIC_STRINGS:
        return None

    if isinstance(obj, dict):
        return {k.date().isoformat() if isinstance(k, datetime) else str(k): scrub_for_json(v) for k, v in obj.items()}

    return obj


def fetch_profile_for_ticker(ticker: yf.Ticker) -> tuple[str, YahooData]:
    yahoo_data: YahooData = {
        "info": {},
        "cashflow": {},
        "earnings": {},
        "recommendations": {},
        "analyst_price_targets": {},
        "splits": {},
        "news": [],
        "balance_sheet": {},
        "income_stmt": {},
        "insider_transactions": [],
    }

    try:
        yahoo_data["info"] = scrub_for_json(ticker.info)

        if yahoo_data["info"].get("quoteType") == "ETF":
            # ETF don't have cashflow, earnings, recommendations, analyst_price_targets, splits
            return ticker.ticker, yahoo_data

        data = ticker.cashflow.to_dict()
        yahoo_data["cashflow"] = scrub_for_json(data)

        # Annual balance sheet + income statement power the Piotroski F-Score (needs YoY deltas).
        yahoo_data["balance_sheet"] = scrub_for_json(ticker.balance_sheet.to_dict())
        yahoo_data["income_stmt"] = scrub_for_json(ticker.income_stmt.to_dict())

        # Yahoo's holders modules are by far the flakiest part of yfinance
        # (404s, KeyErrors, empty frames — see ranaroussi/yfinance#1904). Wrap
        # in its own try so a single bad response doesn't taint the rest of
        # the profile fetch. UK/EU tickers will simply return [].
        try:
            insider_df = ticker.insider_transactions
            if insider_df is not None and not insider_df.empty:
                yahoo_data["insider_transactions"] = scrub_for_json(insider_df.to_dict(orient="records"))
        except Exception as e:  # noqa: BLE001 — yfinance throws every exception type imaginable here
            logger.debug("No insider_transactions for %s: %s", ticker.ticker, e)

        try:
            data = ticker.get_earnings_dates(limit=40)
        except KeyError as e:
            logger.warning("Failed to get earnings for %s: %s", ticker.ticker, e)
            return ticker.ticker, yahoo_data

        if data is None:
            # ETFs don't have earnings
            yahoo_data["earnings"] = {}
        else:
            yahoo_data["earnings"] = scrub_for_json(data.to_dict(orient="index"))

        data = ticker.recommendations.to_dict(orient="index")
        data = {
            datetime.now(TIMEZONE).date().replace(day=1) + relativedelta(months=int(v.pop("period").rstrip("m"))): v
            for k, v in data.items()
        }
        yahoo_data["recommendations"] = scrub_for_json(data)

        yahoo_data["analyst_price_targets"] = ticker.analyst_price_targets

        # Fetch and store splits
        splits_data = ticker.splits.to_dict()
        yahoo_data["splits"] = scrub_for_json(splits_data)

        # Fetch and store news
        yahoo_data["news"] = ticker.get_news()
    except (ValueError, AttributeError) as e:
        logger.warning("Problem with parsing DataFrame %s: %s", ticker.ticker, e)

    return ticker.ticker, yahoo_data


def get_yahoo_ticker_data(symbols: list[str]) -> dict[str, YahooData]:
    """Update Yahoo Finance data for all holdings."""
    logger.info("Fetching %s Yahoo Finance profiles (first 10: %s)", len(symbols), symbols[:10])
    yahoo_data: dict[str, YahooData] = {}

    # Fetch in batches
    for i in range(0, len(symbols), BATCH_SIZE_YF):
        batch = symbols[i : i + BATCH_SIZE_YF]

        # Use yfinance's batch download capability
        tickers = yf.Tickers(" ".join(batch))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(fetch_profile_for_ticker, tickers.tickers.values())

            for ticker, data in results:
                yahoo_data[ticker] = data

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
    logger.info("Fetching portfolio from Trading212 API")
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

        # Calculate positions stats
        holdings = session.query(HoldingDaily).filter(HoldingDaily.date == snapshot_date).all()
        positions_total = len(holdings)
        positions_winning = sum(1 for h in holdings if h.ppl > 0)

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
            existing_snapshot.positions_total = positions_total
            existing_snapshot.positions_winning = positions_winning
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
                positions_total=positions_total,
                positions_winning=positions_winning,
            )
            session.add(snapshot)


def calculate_portfolio_risk_metrics(session: Session, snapshot_date: date) -> dict[str, Optional[float]]:
    """
    Sharpe, Sortino and Beta on a cash-flow-neutral daily return series.

    Raw ``PortfolioDaily.value.pct_change()`` treats a deposit as a daily
    return (a £10k top-up on a £70k book shows as +14%). Those spikes inflate
    ``excess_returns.mean()`` but never enter downside deviation (they're
    positive), which makes Sharpe and Sortino meaningless.

    Instead, daily factors are computed the same way TWRR does in
    ``scripts/update_returns.py``::

        factor_t = value_t / (value_{t-1} + net_cash_flow_t)

    so deposits/withdrawals pass through cleanly. The resulting series is the
    same wealth-index concept used for Max DD and the Underwater chart.
    """
    start_date = snapshot_date - timedelta(days=365)

    # 1. Portfolio total value, calendar-daily (PortfolioDaily has a row per day).
    portfolio_history = (
        session.query(PortfolioDaily.date, PortfolioDaily.value)
        .filter(PortfolioDaily.date >= start_date, PortfolioDaily.date <= snapshot_date)
        .order_by(PortfolioDaily.date)
        .all()
    )
    if len(portfolio_history) < 2:
        return {"sharpe": None, "sortino": None, "beta": None}

    # 2. Net daily cash flows (deposit +, withdrawal -) over the same window.
    cash_flow_rows = (
        session.query(
            func.date(TransactionHistory.timestamp).label("day"),
            func.sum(
                case(
                    (TransactionHistory.action == TransactionAction.DEPOSIT, TransactionHistory.total),
                    else_=-TransactionHistory.total,
                )
            ).label("net_flow"),
        )
        .filter(
            TransactionHistory.action.in_([TransactionAction.DEPOSIT, TransactionAction.WITHDRAWAL]),
            func.date(TransactionHistory.timestamp) >= start_date,
            func.date(TransactionHistory.timestamp) <= snapshot_date,
        )
        .group_by(func.date(TransactionHistory.timestamp))
        .all()
    )
    cash_flows: dict[date, float] = {r.day: float(r.net_flow) for r in cash_flow_rows}

    # 3. Cash-flow-neutral daily returns. Convention matches update_returns.py:
    #    the flow is applied at the start of the day, so it enters the denominator.
    portfolio_returns: list[float] = []
    portfolio_dates: list[date] = []
    for (_, prev_v), (curr_d, curr_v) in zip(portfolio_history, portfolio_history[1:]):
        denominator = prev_v + cash_flows.get(curr_d, 0.0)
        if denominator > 0 and curr_v > 0:
            portfolio_returns.append(curr_v / denominator - 1.0)
            portfolio_dates.append(curr_d)

    if len(portfolio_returns) < 2:
        return {"sharpe": None, "sortino": None, "beta": None}

    returns_s = pd.Series(portfolio_returns, index=pd.Index(portfolio_dates))

    # 4. Sharpe / Sortino. Annualise over calendar days (matches TWRR's DAYS_PER_YEAR
    #    convention) since the series covers every day, not just trading days.
    annualization_factor = np.sqrt(DAYS_PER_YEAR) if len(returns_s) > 60 else 1.0

    risk_free_rate_daily = (1 + RISK_FREE_RATE) ** (1 / DAYS_PER_YEAR) - 1
    excess_returns = returns_s - risk_free_rate_daily

    sharpe_ratio: Optional[float] = (
        float((excess_returns.mean() / excess_returns.std()) * annualization_factor)
        if excess_returns.std() != 0
        else None
    )

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio: Optional[float] = (
        float((excess_returns.mean() / downside_std) * annualization_factor)
        if downside_std != 0 and not np.isnan(downside_std)
        else None
    )

    # 5. Beta: benchmark (SPY) daily returns aligned to the portfolio dates.
    #    Benchmark trades on weekdays only, so ffill from benchmark onto the
    #    portfolio's calendar-day index and inner-join on shared dates.
    benchmark_prices = (
        session.query(PricesDaily.date, PricesDaily.adj_close_price)
        .filter(PricesDaily.symbol == SPY, PricesDaily.date >= start_date, PricesDaily.date <= snapshot_date)
        .order_by(PricesDaily.date)
        .all()
    )
    beta: Optional[float] = None
    if len(benchmark_prices) >= 2:
        benchmark_df = pd.DataFrame(benchmark_prices, columns=["date", "price"]).set_index("date")
        bench_aligned = benchmark_df["price"].reindex(pd.Index(portfolio_dates), method="ffill")
        bench_returns = bench_aligned.pct_change().dropna()
        joined = pd.concat([returns_s.rename("p"), bench_returns.rename("b")], axis=1).dropna()
        if len(joined) >= 20:
            covariance = joined["p"].cov(joined["b"])
            variance = cast(float, joined["b"].var())
            if variance != 0:
                beta = float(covariance / variance)

    return {"sharpe": sharpe_ratio, "sortino": sortino_ratio, "beta": beta}


def update_data():
    """Update all data in the database."""
    logger.info("Starting data update process")

    # 1. Update rates
    update_currency_rates(CURRENCIES)

    # 2. Update instruments
    # _isins_to_add: (isin, currencyCode) must match Trading 212 equity metadata exactly
    # (see update_instruments: same ISIN can list twice with different currencies).
    # LSE stocks quoted in pence (UI shows "p…") use currencyCode "GBX", not "GBP".
    _isins_to_add: set[tuple[str, str]] = set()
    update_instruments(_isins_to_add)

    # 3. Update holdings
    update_holdings()

    # 4. Update prices
    _tickers_to_add: set[str] = set()  # "^VIX"
    update_prices(_tickers_to_add)

    # 5. Update portfolio
    update_portfolio()

    # 6. Clear caches
    fetch_holdings.cache_clear()


if __name__ == "__main__":
    update_data()
