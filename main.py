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

You **must** set an environment variable:
export TRADING212_API_KEY="live‑api‑key‑goes‑here"
"""
from __future__ import annotations

import logging
import json
import os
import re
import sys

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Iterable, Mapping

import pandas as pd
import requests
import yfinance as yf
import numpy as np

from data import STOCKS_SUFFIX, STOCKS_ALIASES, STOCKS_DELISTED, ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION
from db import currency_table, DailyPriceManager

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  ───────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------
API_KEY: str = os.environ["TRADING212_API_KEY"]
CACHE_DIR = Path("~/.cache/t212").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FIELD = "Adj Close"            # or "Close" if you prefer raw closes
BATCH_SIZE_YF = 25                    # tickers per yahoo request
REQUEST_RETRY = 5
REQUEST_SLEEP = 1.0                   # polite pause between Yahoo calls
HISTORY_YEARS = 10

BENCH = "SPY"         # or VUAG.L to keep GBP base
RISK_FREE = 0.045     # annual risk-free guess (4.5 %); replace with 3-month T-bill

RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

price_manager = DailyPriceManager()


# ──────────────────────────────────────────────────────────────────────────────
# Helper dataclasses  ─────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------
@dataclass(slots=True)
class Instrument:
    t212_code: str
    name: str
    currency: str

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

def fetch_instruments() -> dict[str, Instrument]:
    url = "https://live.trading212.com/api/v0/equity/metadata/instruments"
    raw = request_json(url, {"Authorization": API_KEY})
    return {i["ticker"]: Instrument(i["ticker"], i["name"], i["currencyCode"]) for i in raw}


def fetch_portfolio() -> Iterable[Holding]:
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    raw = request_json(url, {"Authorization": API_KEY})
    raw = [h for h in raw if h["ticker"] not in STOCKS_DELISTED]

    instr_map = fetch_instruments()

    tickers = [instr_map[h["ticker"]].yahoo for h in raw]
    hist = price_manager.history(tickers, datetime.now() - timedelta(days=7), datetime.now(), PRICE_FIELD)

    return [
        Holding(
            instr=instr_map[h["ticker"]],
            quantity=h["quantity"],
            avg_price=h["averagePrice"],
            current_price=h["currentPrice"],
            ppl=h["ppl"],
            fx_ppl=h["fxPpl"] or 0.0,
            prices=hist[instr_map[h["ticker"]].yahoo]
        )
        for h in raw
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Finance helpers  ────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------


def yahoo_profile(symbol: str) -> dict:  # cached 24h
    cache_file = CACHE_DIR / f"profile_{symbol}.json"
    if cache_file.exists() and cache_file.stat().st_mtime > (
        datetime.now().timestamp() - 86_400
    ):
        return json.loads(cache_file.read_text())
    info = yf.Ticker(symbol).info or {}
    cache_file.write_text(json.dumps(info))

    return info


# ──────────────────────────────────────────────────────────────────────────────
# Main business logic  ────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------

def build_snapshot(holdings) -> pd.DataFrame:
    rates = currency_table()

    records: list[dict] = []
    for h in holdings:
        gbp_value = h.market_value_native * rates[h.instr.currency]
        info = yahoo_profile(h.instr.yahoo)

        records.append(
            {
                "ticker": h.instr.yahoo,
                "name": h.instr.name,
                "%": 0.0,
                "value_gbp": gbp_value,
                "profit": h.ppl,
                "return": round(h.ppl / (gbp_value - h.ppl) * 100.0, 2),
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
    # 1. make a clean “kind” Series
    kind = df["quoteType"].fillna("EQUITY").where(df["quoteType"] == "ETF", "EQUITY")

    # 2. aggregate and convert to %
    value_by_kind = df.groupby(kind)["value_gbp"].sum()
    pct = (value_by_kind / value_by_kind.sum() * 100).round(2)

    # 3. remove the index name so it doesn’t print
    pct = pct.rename_axis(None)

    return pct

def daily_changes(df: pd.DataFrame, lookback: int = 7) -> pd.Series:
    syms = df.index.dropna().tolist()
    hist = price_manager.history(syms, datetime.now() - timedelta(days=lookback), datetime.now(), PRICE_FIELD)
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
        first = h.prices.dropna().iloc[0]
        if first and first > 0:
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
        # XOR decides whether the “alert” colour should be red
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
    print("\nTotal ISA value £{:.2f}".format(snapshot["value_gbp"].sum()))
    print("Total ISA profit £{:.2f} ({:.0f}%)".format(snapshot["profit"].sum(), snapshot["profit"].sum() / (snapshot["value_gbp"].sum() - snapshot["profit"].sum()) * 100.0))

    print(f"\nETF / Equity split:")
    print(etf_equity_allocation(snapshot).to_string(float_format=lambda x: f"{x:5.2f}%"))

    print("\nCountry allocation:")
    print(country_allocation(snapshot).to_string(float_format=lambda x: f"{x:5.2f}%"))

    print("\nSector allocation:")
    print(sector_allocation(snapshot).to_string(float_format=lambda x: f"{x:5.2f}%"))

    # Daily change
    pct = daily_changes_from_holdings(holdings)
    worst = pct.nsmallest(10)
    print("\nWorst 1‑week moves:")
    for i, (k, s) in enumerate(worst.items()):
        print(f"{k:<8} {colour_money(round(s * 100, 2), pct=True)}")

    best = pct.nlargest(10)
    print("\nBest 1‑week moves:")
    for i, (k, s) in enumerate(best.items()):
        print(f"{k:<8} {colour_money(round(s * 100, 2), pct=True)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
