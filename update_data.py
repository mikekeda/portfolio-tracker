"""Trading212 Data Update Script
================================
A streamlined script that updates all data in the database without displaying it.
This script is designed to be run as a scheduled job (cron, GitHub Actions, etc.)

Key features:
- Updates portfolio holdings from Trading212 API
- Fetches and caches Yahoo Finance data
- Updates portfolio snapshots
- Minimal logging and no terminal output

You **must** set environment variables:
export TRADING212_API_KEY="live‑api‑key‑goes‑here"
export DB_NAME="your_database_name"
export DB_PASSWORD="your_database_password"
"""

import logging
from datetime import datetime
from typing import List
from collections import defaultdict

import pandas as pd
import requests
import yfinance as yf

from data import STOCKS_DELISTED, ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION
from database import get_db_service
from currency import get_currency_service
from config import TRADING212_API_KEY, REQUEST_RETRY


# Initialize services
db_service = get_db_service()
currency_service = get_currency_service()


def request_json(url: str, headers: dict, retries: int = REQUEST_RETRY):
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
            import time
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


def convert_ticker(t212: str) -> str:
    """Convert Trading212 code to Yahoo symbol."""
    from data import STOCKS_ALIASES, STOCKS_SUFFIX
    import re

    if t212 in STOCKS_DELISTED:
        raise ValueError(f"{t212} is delisted")

    if not t212.endswith("_EQ"):
        raise ValueError(f"Unknown format: {t212}")

    core = t212[:-3]  # strip _EQ
    PATTERN_MULTI = re.compile(r"^(?P<sym>.+?)_(?P<tag>[A-Z]{2,3})$")

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


def fetch_instruments(tickers: set) -> dict:
    """Fetch instruments from database first, then API if needed."""
    # Check what we have in database
    existing_instruments = db_service.get_instruments_by_codes(tickers)
    missing_tickers = tickers - set(existing_instruments.keys())

    instruments_map = {}

    # Use existing instruments from database
    for t212_code, instrument_data in existing_instruments.items():
        instruments_map[t212_code] = {
            't212_code': instrument_data['t212_code'],
            'name': instrument_data['name'],
            'currency': instrument_data['currency'],
            'sector': instrument_data.get('sector'),
            'country': instrument_data.get('country'),
            'yahoo_symbol': instrument_data.get('yahoo_symbol'),
            'market_cap': instrument_data["yahoo_data"].get("marketCap"),
            'pe_ratio': instrument_data["yahoo_data"].get("trailingPE"),
            'institutional': instrument_data["yahoo_data"].get("heldPercentInstitutions"),
            'beta': instrument_data["yahoo_data"].get("beta"),
        }

    # Fetch missing instruments from API
    if missing_tickers:
        logging.info(f"Fetching {len(missing_tickers)} missing instruments from API")
        url = "https://live.trading212.com/api/v0/equity/metadata/instruments"
        raw = request_json(url, {"Authorization": TRADING212_API_KEY})

        # Store new instruments in database
        instruments_data = []
        for item in raw:
            if item["ticker"] in missing_tickers:
                instruments_data.append({
                    't212_code': item["ticker"],
                    'name': item["name"],
                    'currency': item["currencyCode"],
                    'yahoo_symbol': convert_ticker(item["ticker"])
                })

        # Save to database
        if instruments_data:
            db_service.save_instruments(instruments_data)
            logging.info(f"Saved {len(instruments_data)} new instruments to database")

        # Add new instruments to map
        for item in raw:
            if item["ticker"] in missing_tickers:
                instruments_map[item["ticker"]] = {
                    't212_code': item["ticker"],
                    'name': item["name"],
                    'currency': item["currencyCode"],
                    'yahoo_symbol': convert_ticker(item["ticker"]),
                    # TODO: Set the rest fields
                }
    else:
        logging.info("All instruments found in database, no API call needed")

    return instruments_map


def fetch_portfolio() -> List[dict]:
    """Fetch portfolio holdings from Trading212 API."""
    logging.info("Fetching portfolio from Trading212 API")
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    raw = request_json(url, {"Authorization": TRADING212_API_KEY})
    raw = [h for h in raw if h["ticker"] not in STOCKS_DELISTED]

    instr_map = fetch_instruments({h["ticker"] for h in raw})

    holdings = []
    for h in raw:
        instrument = instr_map.get(h["ticker"], {})
        holdings.append({
            't212_code': h["ticker"],
            'name': instrument.get('name', ''),
            'currency': instrument.get('currency', ''),
            'yahoo_symbol': instrument.get('yahoo_symbol'),
            'quantity': h["quantity"],
            'avg_price': h["averagePrice"],
            'current_price': h["currentPrice"],
            'ppl': h["ppl"],
            'fx_ppl': h["fxPpl"] or 0.0,
            'country': instrument["country"],
            'sector': instrument["sector"],
            'market_cap': instrument["market_cap"],
            'pe_ratio': instrument["pe_ratio"],
            'institutional': instrument["institutional"],
            'beta': instrument["beta"],
        })

    return holdings


def update_yahoo_data(holdings: List[dict]) -> None:
    """Update Yahoo Finance data for all holdings."""
    logging.info("Updating Yahoo Finance data")

    # Get symbols that need Yahoo data
    symbols = [h['yahoo_symbol'] for h in holdings if h.get('yahoo_symbol')]
    if not symbols:
        logging.info("No Yahoo symbols to update")
        return

    # Get instruments that have fresh Yahoo data (less than 24 hours old)
    t212_codes = set(db_service.get_instruments_by_yahoo_symbols(symbols))
    fresh_instruments = db_service.get_instruments_with_fresh_yahoo_data(t212_codes, max_age_days=1)

    # Find symbols that need to be fetched
    symbols_to_fetch = []
    for symbol in symbols:
        t212_code = db_service.get_instruments_by_yahoo_symbols([symbol])[0] if symbol else None
        if t212_code and t212_code not in fresh_instruments:
            symbols_to_fetch.append(symbol)

    if not symbols_to_fetch:
        logging.info("All Yahoo data is fresh, no updates needed")
        return

    logging.info(f"Fetching {len(symbols_to_fetch)} Yahoo Finance profiles")

    # Fetch in batches
    batch_size = 10
    for i in range(0, len(symbols_to_fetch), batch_size):
        batch = symbols_to_fetch[i:i + batch_size]
        try:
            # Use yfinance's batch download capability
            tickers = yf.Tickers(' '.join(batch))
            for symbol in batch:
                try:
                    info = tickers.tickers[symbol].info or {}

                    # Cache the result in database
                    t212_code = db_service.get_instruments_by_yahoo_symbols([symbol])[0]
                    db_service.update_instrument_yahoo_data(t212_code, info)

                except Exception as e:
                    logging.warning(f"Failed to fetch {symbol}: {e}")

        except Exception as e:
            logging.warning(f"Failed to fetch batch {batch}: {e}")
            # Fallback to individual requests
            for symbol in batch:
                try:
                    info = yf.Ticker(symbol).info or {}
                    t212_code = db_service.get_instruments_by_yahoo_symbols([symbol])[0]
                    db_service.update_instrument_yahoo_data(t212_code, info)
                except Exception as e2:
                    logging.warning(f"Failed to fetch {symbol}: {e2}")


def _weighted_add(target: dict[str, float], weights: dict[str, float], value: float):
    """Add *value* into target buckets using percentage weights."""
    for bucket, pct in weights.items():
        target[bucket] += value * pct / 100.0


def country_allocation(df: pd.DataFrame) -> pd.Series:
    """% NAV by country, treating ETFs via ETF_COUNTRY_ALLOCATION map."""
    buckets: defaultdict[str, float] = defaultdict(float)
    for sym, row in df.iterrows():
        val = row["value_gbp"]
        if sym in ETF_COUNTRY_ALLOCATION:
            _weighted_add(buckets, ETF_COUNTRY_ALLOCATION[sym], val)
        else:
            buckets[row["country"]] += val
    total = sum(buckets.values())
    return pd.Series({k: round(v / total * 100, 2) for k, v in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)})


def sector_allocation(df: pd.DataFrame) -> pd.Series:
    """% NAV by sector, treating ETFs via ETF_SECTOR_ALLOCATION map."""
    buckets: defaultdict[str, float] = defaultdict(float)
    for sym, row in df.iterrows():
        val = row["value_gbp"]
        if sym in ETF_SECTOR_ALLOCATION:
            _weighted_add(buckets, ETF_SECTOR_ALLOCATION[sym], val)
        else:
            buckets[row["sector"]] += val
    total = sum(buckets.values())
    return pd.Series({k: round(v / total * 100, 2) for k, v in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)})


def etf_equity_allocation(df: pd.DataFrame) -> pd.Series:
    """Return % of portfolio in ETFs vs direct equities."""
    kind = df["quoteType"].fillna("EQUITY").where(df["quoteType"] == "ETF", "EQUITY")
    value_by_kind = df.groupby(kind)["value_gbp"].sum()
    pct = (value_by_kind / value_by_kind.sum() * 100).round(2)
    pct = pct.rename_axis(None)
    return pct


def build_snapshot(holdings: List[dict]) -> pd.DataFrame:
    """Build portfolio snapshot from holdings data."""
    # Get currency rates once for all conversions
    rates = currency_service.get_currency_table()

    # Get all Yahoo symbols for batch processing
    yahoo_symbols = [h['yahoo_symbol'] for h in holdings if h.get('yahoo_symbol')]

    # Fetch all Yahoo profiles in batch
    yahoo_profiles = {}
    if yahoo_symbols:
        # Get instruments that have fresh Yahoo data
        t212_codes = set(db_service.get_instruments_by_yahoo_symbols(yahoo_symbols))
        fresh_instruments = db_service.get_instruments_with_fresh_yahoo_data(t212_codes, max_age_days=1)

        # Map Yahoo symbols to their data
        for t212_code, instrument_data in fresh_instruments.items():
            if instrument_data.get('yahoo_symbol') in yahoo_symbols:
                yahoo_profiles[instrument_data['yahoo_symbol']] = instrument_data.get('yahoo_data', {})

    records: list[dict] = []
    for h in holdings:
        currency = h.get('currency', 'GBP')
        rate = rates.get(currency, 1.0)
        gbp_value = h['quantity'] * h['current_price'] * rate
        info = yahoo_profiles.get(h.get('yahoo_symbol'), {}) if h.get('yahoo_symbol') else {}

        # Calculate return percentage safely
        cost_basis = gbp_value - h['ppl']
        return_pct = round(h['ppl'] / cost_basis * 100.0, 2) if cost_basis > 0 else 0.0

        records.append({
            "ticker": h.get('yahoo_symbol'),
            "name": h['name'],
            "%": 0.0,
            "value_gbp": gbp_value,
            "profit": h['ppl'],
            "return": return_pct,
            "prediction": round((info["targetMedianPrice"] / h['current_price'] - 1) * 100.0) if info.get("targetMedianPrice") else "",
            "instit": round(info["heldPercentInstitutions"] * 100.0) if info.get("heldPercentInstitutions") else "",
            "marketCap": round(info["marketCap"] / 1_000_000_000.0) if info.get("marketCap") else "",
            "peg": round(info["trailingPegRatio"], 2) if info.get("trailingPegRatio") else "",
            "PE": round(info["trailingPE"]) if info.get("trailingPE") else "",
            "beta": round(info["beta"], 2) if info.get("beta") else "",
            "margins": round(info["profitMargins"] * 100.0) if info.get("profitMargins") else "",
            "grow": round(info["revenueGrowth"] * 100.0) if info.get("revenueGrowth") else "",
            "roa": round(info["returnOnAssets"] * 100.0) if info.get("returnOnAssets") else "",
            "fcf_yld": round(info["freeCashflow"] / info["enterpriseValue"] * 100, 2) if (info.get("freeCashflow") and info.get("enterpriseValue")) else "",
            "recommendation": round(info["recommendationMean"], 2) if info.get("recommendationMean") else "",
            "52WeekHighChange": round(info["fiftyTwoWeekHighChangePercent"] * 100) if info.get("fiftyTwoWeekHighChangePercent") else "",
            "short": round(info["shortPercentOfFloat"] * 100) if info.get("shortPercentOfFloat") else "",
            "RSI": "",
            "country": info.get("country", "Other"),
            "sector": info.get("sector", "Other"),
            "quoteType": info.get("quoteType", "Unknown"),
            "currency": currency,
            "current_price": h['current_price'],
        })

    df = (
        pd.DataFrame(records)
        .set_index("ticker")
        .sort_values("value_gbp", ascending=False)
    )
    df["%"] = (df["value_gbp"] / df["value_gbp"].sum() * 100).round(2)

    return df


def update_portfolio_snapshot(holdings: List[dict]) -> None:
    """Update portfolio snapshot in database."""
    logging.info("Updating portfolio snapshot")

    try:
        # Build snapshot
        snapshot = build_snapshot(holdings)

        # Calculate portfolio metrics
        total_value = float(snapshot["value_gbp"].sum())
        total_profit = float(snapshot["profit"].sum())
        total_return_pct = (total_profit / (total_value - total_profit) * 100) if (total_value - total_profit) > 0 else 0

        # Calculate allocations
        country_alloc = country_allocation(snapshot)
        sector_alloc = sector_allocation(snapshot)
        etf_equity = etf_equity_allocation(snapshot)

        # Create snapshot data
        snapshot_data = {
            'total_value': total_value,
            'total_profit': total_profit,
            'total_return_pct': total_return_pct,
            'snapshot_date': datetime.now().date(),
            'country_allocation': country_alloc.to_dict(),
            'sector_allocation': sector_alloc.to_dict(),
            'etf_equity_split': etf_equity.to_dict(),
        }

        db_service.save_portfolio_snapshot(snapshot_data)
        logging.info("Saved portfolio snapshot to database")

    except Exception as e:
        logging.error(f"Failed to update portfolio snapshot: {e}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting data update process")

    # Step 1: Fetch portfolio holdings
    holdings = fetch_portfolio()
    db_service.save_holdings(holdings)
    logging.info(f"Fetched {len(holdings)} holdings")

    # Step 2: Update Yahoo Finance data
    update_yahoo_data(holdings)

    # Step 3: Update portfolio snapshot
    update_portfolio_snapshot(holdings)

    logging.info("Data update process completed successfully")
