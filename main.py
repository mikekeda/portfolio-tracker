"""Trading212 → Yahoo Finance integration
================================================
A small, self‑contained script that
1. maps Trading212 instrument codes to Yahoo symbols,
2. pulls your live portfolio & instrument metadata,
3. converts everything to GBP, and
4. prints a quick P&L + daily‑change summary.

Designed for UK ISAs (no tax logic) but easy to extend.

* Run ad‑hoc: ``python main.py``
* Schedule via cron / GitHub Actions for daily Slack posts

You **must** set environment variables:
export TRADING212_API_KEY="live‑api‑key‑goes‑here"
export DB_NAME="your_database_name"
export DB_PASSWORD="your_database_password"
"""
from __future__ import annotations

import logging
import re
import sys

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep
from typing import Iterable, Mapping, Set, List, Dict, Optional

import pandas as pd
import requests
import yfinance as yf
import numpy as np

from data import STOCKS_SUFFIX, STOCKS_ALIASES, STOCKS_DELISTED, ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION
from models import init_database
from database import get_db_service
from currency import get_currency_service
from config import (
    TRADING212_API_KEY, PRICE_FIELD, REQUEST_RETRY, MAX_COUNTRIES_DISPLAY,
    MAX_SECTORS_DISPLAY, MAX_PERFORMERS_DISPLAY, RED, GREEN, RESET, validate_config
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  ───────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

# Initialize services
db_service = get_db_service()
currency_service = get_currency_service()


# ──────────────────────────────────────────────────────────────────────────────
# Helper dataclasses  ─────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------
@dataclass(slots=True)
class Instrument:
    t212_code: str
    name: str
    currency: str
    sector: Optional[str] = None
    country: Optional[str] = None

    @property
    def yahoo(self) -> str | None:
        """Return Yahoo‑Finance symbol or *None* if delisted/unknown."""
        try:
            return convert_ticker(self.t212_code)
        except ValueError:
            return None

@dataclass(slots=True)
class Holding:
    instr: Instrument
    quantity: float
    avg_price: float
    current_price: float
    ppl: float
    fx_ppl: float
    prices: pd.DataFrame = None

    @property
    def market_value_native(self) -> float:
        return self.quantity * self.current_price

# ──────────────────────────────────────────────────────────────────────────────
# Ticker conversion  ──────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------
PATTERN_MULTI = re.compile(r"^(?P<sym>.+?)_(?P<tag>[A-Z]{2,3})$")


def convert_ticker(t212: str) -> str:
    """Convert a Trading 212 code like 'BARCl_EQ' to a Yahoo symbol 'BARC.L'."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Network helpers  ────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

def request_json(url: str, headers: Mapping[str, str], retries: int = REQUEST_RETRY):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            logging.warning("%s → HTTP %s, %s", url, r.status_code, r.text)
        except requests.RequestException as exc:
            logging.warning("Request error %s → %s", url, exc)
        sleep(2 ** attempt)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


# ──────────────────────────────────────────────────────────────────────────────
# Data pulls  ─────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

def fetch_instruments(tickers: Set[str]) -> dict[str, Instrument]:
    """Fetch instruments from database first, then API if needed."""
    # Check what we have in database
    existing_instruments = db_service.get_instruments_by_codes(tickers)
    missing_tickers = tickers - set(existing_instruments.keys())

    instruments_map = {}

    # Use existing instruments from database
    for t212_code, instrument_data in existing_instruments.items():
        instruments_map[t212_code] = Instrument(
            t212_code=instrument_data['t212_code'],
            name=instrument_data['name'],
            currency=instrument_data['currency'],
            sector=instrument_data['sector'],
            country=instrument_data['country'],
        )

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
                    'yahoo_symbol': convert_ticker(item["ticker"]) if item["ticker"] not in STOCKS_DELISTED else None
                })

        # Save to database
        db_service.save_instruments(instruments_data)
        logging.debug(f"Saved {len(instruments_data)} new instruments to database")

        # Add new instruments to map
        for item in raw:
            if item["ticker"] in missing_tickers:
                instruments_map[item["ticker"]] = Instrument(
                    t212_code=item["ticker"],
                    name=item["name"],
                    currency=item["currencyCode"]
                )
    else:
        logging.info("All instruments found in database, no API call needed")

    return instruments_map


def fetch_portfolio() -> Iterable[Holding]:
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    raw = request_json(url, {"Authorization": TRADING212_API_KEY})
    raw = [h for h in raw if h["ticker"] not in STOCKS_DELISTED]

    instr_map = fetch_instruments({h["ticker"] for h in raw})

    tickers = [instr_map[h["ticker"]].yahoo for h in raw if instr_map[h["ticker"]].yahoo]
    hist = db_service.get_price_history(tickers, datetime.now() - timedelta(days=8), datetime.now() - timedelta(days=1), PRICE_FIELD)

    holdings = [
        Holding(
            instr=instr_map[h["ticker"]],
            quantity=h["quantity"],
            avg_price=h["averagePrice"],
            current_price=h["currentPrice"],
            ppl=h["ppl"],
            fx_ppl=h["fxPpl"] or 0.0,
            prices=hist[instr_map[h["ticker"]].yahoo] if instr_map[h["ticker"]].yahoo in hist.columns else None
        )
        for h in raw
    ]

    # Store holdings in database
    holdings_data = []
    for holding in holdings:
        # Get Yahoo Finance data for additional metadata
        yahoo_info = yahoo_profile(holding.instr.yahoo) if holding.instr.yahoo else {}

        holdings_data.append({
            't212_code': holding.instr.t212_code,
            'name': holding.instr.name,
            'currency': holding.instr.currency,
            'yahoo_symbol': holding.instr.yahoo,
            'quantity': holding.quantity,
            'avg_price': holding.avg_price,
            'current_price': holding.current_price,
            'ppl': holding.ppl,
            'fx_ppl': holding.fx_ppl,
            'country': yahoo_info.get("country", "Other"),
            'sector': yahoo_info.get("sector", "Other"),
            'market_cap': yahoo_info.get("marketCap"),
            'pe_ratio': yahoo_info.get("trailingPE"),
            'institutional': yahoo_info.get("heldPercentInstitutions"),
            'beta': yahoo_info.get("beta"),
        })

    # Save to database
    db_service.save_holdings(holdings_data)
    logging.info(f"Saved {len(holdings_data)} holdings to database")

    return holdings

# ──────────────────────────────────────────────────────────────────────────────
# Finance helpers  ────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------


def yahoo_profile(symbol: str) -> dict:  # cached 24h
    """Get Yahoo Finance profile for a single symbol using database cache."""
    # Check if we have fresh data in database
    cached_data = db_service.get_yahoo_profile_from_cache(symbol)
    if cached_data:
        return cached_data

    # Fetch fresh data from Yahoo Finance
    info = yf.Ticker(symbol).info or {}

    # Cache the result in database
    t212_code = db_service.get_instruments_by_yahoo_symbols([symbol])[0]
    db_service.update_instrument_yahoo_data(t212_code, info)

    return info


def yahoo_profiles_batch(symbols: List[str]) -> Dict[str, dict]:
    """Get Yahoo Finance profiles for multiple symbols efficiently using database cache."""
    profiles = {}

    # Get instruments that have fresh Yahoo data (less than 24 hours old)
    # Get T212 codes for these Yahoo symbols
    t212_codes = set(db_service.get_instruments_by_yahoo_symbols(symbols))

    # Get fresh data from database
    fresh_instruments = db_service.get_instruments_with_fresh_yahoo_data(t212_codes, max_age_days=1)

    # Map Yahoo symbols to their data
    for t212_code, instrument_data in fresh_instruments.items():
        if instrument_data.get('yahoo_symbol') in symbols:
            profiles[instrument_data['yahoo_symbol']] = instrument_data.get('yahoo_data', {})

    # Find symbols that need to be fetched
    symbols_to_fetch = [symbol for symbol in symbols if symbol not in profiles]

    # Fetch missing symbols in batches
    if symbols_to_fetch:
        logging.info(f"Fetching {len(symbols_to_fetch)} Yahoo Finance profiles")
        batch_size = 10  # Yahoo Finance can handle small batches

        for i in range(0, len(symbols_to_fetch), batch_size):
            batch = symbols_to_fetch[i:i + batch_size]
            try:
                # Use yfinance's batch download capability
                tickers = yf.Tickers(' '.join(batch))
                for symbol in batch:
                    info = tickers.tickers[symbol].info or {}
                    profiles[symbol] = info

                    # Cache the result in database
                    t212_code = db_service.get_instruments_by_yahoo_symbols([symbol])[0]
                    db_service.update_instrument_yahoo_data(t212_code, info)

            except Exception as e:
                logging.warning(f"Failed to fetch batch {batch}: {e}")
                # Fallback to individual requests
                for symbol in batch:
                    try:
                        info = yf.Ticker(symbol).info or {}
                        profiles[symbol] = info

                        # Cache the result in database
                        t212_code = db_service.get_instruments_by_yahoo_symbols([symbol])[0]
                        db_service.update_instrument_yahoo_data(t212_code, info)
                    except Exception as e2:
                        logging.warning(f"Failed to fetch {symbol}: {e2}")
                        profiles[symbol] = {}

    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# Main business logic  ────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

def build_snapshot(holdings) -> pd.DataFrame:
    # Get currency rates once for all conversions
    rates = currency_service.get_currency_table()

    # Get all Yahoo symbols for batch processing
    yahoo_symbols = [h.instr.yahoo for h in holdings if h.instr.yahoo]

    # Fetch all Yahoo profiles in batch
    yahoo_profiles = yahoo_profiles_batch(yahoo_symbols) if yahoo_symbols else {}

    records: list[dict] = []
    for h in holdings:
        gbp_value = h.market_value_native * rates[h.instr.currency]
        info = yahoo_profiles.get(h.instr.yahoo, {}) if h.instr.yahoo else {}

        # Calculate return percentage safely
        cost_basis = gbp_value - h.ppl
        return_pct = round(h.ppl / cost_basis * 100.0, 2) if cost_basis > 0 else 0.0

        records.append(
            {
                "ticker": h.instr.yahoo,
                "name": h.instr.name,
                "%": 0.0,
                "value_gbp": gbp_value,
                "profit": h.ppl,
                "return": return_pct,
                "prediction": round((info["targetMedianPrice"] / h.current_price - 1) * 100.0) if info.get("targetMedianPrice") else "",
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
                "currency": h.instr.currency,
                "current_price": h.current_price,
            }
        )

    df = (
        pd.DataFrame(records)
        .set_index("ticker")
        .sort_values("value_gbp", ascending=False)
    )
    df["%"] = (df["value_gbp"] / df["value_gbp"].sum() * 100).round(2)

    return df


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
    """
    Return % of portfolio in ETFs vs direct equities.
    """
    # 1. make a clean "kind" Series
    kind = df["quoteType"].fillna("EQUITY").where(df["quoteType"] == "ETF", "EQUITY")

    # 2. aggregate and convert to %
    value_by_kind = df.groupby(kind)["value_gbp"].sum()
    pct = (value_by_kind / value_by_kind.sum() * 100).round(2)

    # 3. remove the index name so it doesn't print
    pct = pct.rename_axis(None)

    return pct

def daily_changes(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    syms = df.index.dropna().tolist()
    hist = db_service.get_price_history(syms, datetime.now() - timedelta(days=lookback), datetime.now() - timedelta(days=1), PRICE_FIELD)
    # use first valid price in window per symbol
    first_px = hist.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    out = (df["current_price"].astype(float).reindex(first_px.index) / first_px - 1).round(4)
    return out.dropna()


def daily_changes_from_holdings(holdings: Iterable["Holding"]) -> pd.Series:
    """
    Compute % change from the first available price in each holding.prices
    to the current price, using the history already fetched in fetch_portfolio().
    """
    out = {}
    for h in holdings:
        sym = h.instr.yahoo
        if not sym or h.prices is None or h.prices.empty:
            continue
        # first valid price in the stored window
        first = float(h.prices.dropna().iloc[0])
        if first > 0:
            out[sym] = round(h.current_price / float(first) - 1, 4)
    return pd.Series(out)


def colour_money(x: float, /, *, pct: bool=False,
                 min_: float = 0, max_: float = 0,
                 reverse: bool=False, sign: bool = True) -> str:
    if not isinstance(x, (int, float)):
        return str(x)

    val = f"{x:+}" if sign else str(x)
    val += "%" if pct else ""

    is_positive = x > max_
    is_negative = x < min_

    if is_positive or is_negative:
        # XOR decides whether the "alert" colour should be red
        want_red = is_negative ^ reverse
        colour   = RED if want_red else GREEN
        return f"{colour}{val}{RESET}"

    # falls through when x is inside [min_, max_]
    return val


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry‑point  ────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s ▸ %(message)s")

    # Validate configuration
    try:
        validate_config()
        logging.debug("Configuration validated successfully")
    except Exception as e:
        logging.error("Configuration error: %s", e)
        sys.exit(1)

    # Initialize database
    try:
        init_database()
        logging.debug("Database initialized successfully")
    except Exception as e:
        logging.error("Failed to initialize database: %s", e)
        sys.exit(1)

    number_of_holdings = 100
    holdings = fetch_portfolio()
    snapshot = build_snapshot(holdings)

    colored_snapshot = snapshot.copy()
    print(f"Portfolio ({len(colored_snapshot)}):")
    print(colored_snapshot.index.tolist())
    print([t for t in colored_snapshot.index.tolist() if not "." in t])
    print(f"\nCurrent holdings (top‑{number_of_holdings}):")
    colored_snapshot["profit"] = colored_snapshot["profit"].apply(colour_money)
    colored_snapshot["return"] = colored_snapshot["return"].apply(
        lambda x: colour_money(x, pct=True)
    )
    colored_snapshot["prediction"] = colored_snapshot["prediction"].apply(
        lambda x: colour_money(x, pct=True, min_=0, max_=20)
    )
    colored_snapshot["instit"] = colored_snapshot["instit"].apply(
        lambda x: colour_money(x, pct=True, min_=40, max_=80, sign=False)
    )
    colored_snapshot["peg"] = colored_snapshot["peg"].apply(
        lambda x: colour_money(x, min_=1.5, max_=3.0, reverse=True, sign=False)
    )
    colored_snapshot["PE"] = colored_snapshot["PE"].apply(
        lambda x: colour_money(x, min_=30, max_=100, reverse=True, sign=False)
    )
    colored_snapshot["beta"] = colored_snapshot["beta"].apply(
        lambda x: colour_money(x, min_=1, max_=2, reverse=True, sign=False)
    )
    colored_snapshot["margins"] = colored_snapshot["margins"].apply(
        lambda x: colour_money(x, min_=10, max_=30, pct=True)
    )
    colored_snapshot["grow"] = colored_snapshot["grow"].apply(
        lambda x: colour_money(x, min_=15, max_=40, pct=True)
    )
    colored_snapshot["roa"] = colored_snapshot["roa"].apply(
        lambda x: colour_money(x, min_=2, max_=10, pct=True)
    )
    # Free cash flow yield is a financial ratio that indicates how much free cash flow a company generates relative to its market capitalization.
    # FCF-yield = freeCashflow / EV (%): 🟢 > 6 cheap, ⚪ 2–6 normal, 🔴 < 2 or neg cash-burn.  Verify any >15% for data quirks.
    colored_snapshot["fcf_yld"] = colored_snapshot["fcf_yld"].apply(
        lambda x: colour_money(x, min_=2, max_=6, pct=True)
    )
    colored_snapshot["recommendation"] = colored_snapshot["recommendation"].apply(
        lambda x: colour_money(x, min_=1.5, max_=2.5, reverse=True, sign=False)
    )
    colored_snapshot["52WeekHighChange"] = colored_snapshot["52WeekHighChange"].apply(
        lambda x: colour_money(x, min_=-20, max_=0, pct=True, reverse=True)
    )
    colored_snapshot["short"] = colored_snapshot["short"].apply(
        lambda x: colour_money(x, min_=0, max_=20, pct=True, reverse=True, sign=False)
    )
    print(colored_snapshot.head(number_of_holdings).to_markdown(floatfmt=".2f"))
    print()

    # Calculate summary statistics
    total_value = float(snapshot["value_gbp"].sum())
    total_profit = float(snapshot["profit"].sum())
    total_return_pct = (total_profit / (total_value - total_profit) * 100) if (total_value - total_profit) > 0 else 0

    # Portfolio summary
    print("=" * 80)
    print("📊 PORTFOLIO SUMMARY")
    print("=" * 80)
    print(f"💰 Total Value: £{total_value:,.2f}")
    print(f"📈 Total Profit: £{total_profit:,.2f} ({total_return_pct:.1f}%)")
    print(f"📦 Number of Holdings: {len(snapshot)}")
    print(f"🏢 Top 5 Holdings: {', '.join(snapshot.head(5).index.tolist())}")

    # Risk metrics
    profitable_holdings = snapshot[snapshot["profit"] > 0]
    losing_holdings = snapshot[snapshot["profit"] < 0]

    print(f"✅ Profitable Positions: {len(profitable_holdings)} ({len(profitable_holdings)/len(snapshot)*100:.1f}%)")
    print(f"❌ Losing Positions: {len(losing_holdings)} ({len(losing_holdings)/len(snapshot)*100:.1f}%)")

    if len(losing_holdings) > 0:
        worst_loss = losing_holdings["profit"].min()
        worst_ticker = losing_holdings.loc[losing_holdings["profit"].idxmin()].name
        print(f"📉 Biggest Loss: {worst_ticker} (£{worst_loss:,.2f})")

    if len(profitable_holdings) > 0:
        best_gain = profitable_holdings["profit"].max()
        best_ticker = profitable_holdings.loc[profitable_holdings["profit"].idxmax()].name
        print(f"📈 Biggest Gain: {best_ticker} (£{best_gain:,.2f})")

    # Allocations
    print()
    print("🌍 ALLOCATIONS")
    print("-" * 80)

    print("ETF / Equity split:")
    etf_equity = etf_equity_allocation(snapshot)
    for asset_type, pct in etf_equity.items():
        print(f"  {asset_type}: {pct:.1f}%")

    print(f"\nTop {MAX_COUNTRIES_DISPLAY} Countries:")
    country_alloc = country_allocation(snapshot)
    for country, pct in country_alloc.head(MAX_COUNTRIES_DISPLAY).items():
        print(f"  {country}: {pct:.1f}%")

    print(f"\nTop {MAX_SECTORS_DISPLAY} Sectors:")
    sector_alloc = sector_allocation(snapshot)
    for sector, pct in sector_alloc.head(MAX_SECTORS_DISPLAY).items():
        print(f"  {sector}: {pct:.1f}%")

    print()

    # Weekly performance
    print("📅 WEEKLY PERFORMANCE")
    print("-" * 80)

    pct = daily_changes_from_holdings(holdings)

    print("🔥 Best Performers:")
    best = pct.nlargest(MAX_PERFORMERS_DISPLAY)
    for ticker, change in best.items():
        print(f"  {ticker:<8} {colour_money(round(change * 100, 2), pct=True)}")

    print("\n📉 Worst Performers:")
    worst = pct.nsmallest(MAX_PERFORMERS_DISPLAY)
    for ticker, change in worst.items():
        print(f"  {ticker:<8} {colour_money(round(change * 100, 2), pct=True)}")

    # Portfolio performance
    portfolio_change = (pct * snapshot.loc[pct.index, "value_gbp"]).sum() / total_value
    print(f"\n📊 Portfolio Weekly Change: {colour_money(round(portfolio_change * 100, 2), pct=True)}")

    print()
    print("=" * 80)

    # Store portfolio snapshot in database
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
