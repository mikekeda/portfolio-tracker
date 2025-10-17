#!/usr/bin/env python3
"""
Scrape Macrotrends PE ratio historical data table for a given stock.

Usage examples:
  - By full URL (recommended):
      python tools/scrape_macrotrends_pe.py --url https://macrotrends.net/stocks/charts/NVDA/nvidia/pe-ratio

  - By ticker and slug (slug is the company name part in the URL, lowercased):
      python tools/scrape_macrotrends_pe.py --ticker NVDA --slug nvidia

Outputs JSON to stdout by default. Use --out csv to output CSV.

Notes:
  - Macrotrends pages may change; this scraper targets the table titled
    "<TICKER> PE Ratio Historical Data" (columns: Date, Stock Price, TTM Net EPS, PE Ratio).
  - Requires: requests, pandas, beautifulsoup4
"""

from __future__ import annotations

import re
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup  # type: ignore[import-untyped]
from sqlalchemy.orm import selectinload
from time import sleep

from config import TIMEZONE
from models import InstrumentYahoo
from update_data import get_session


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def build_url(ticker: str, slug: str) -> str:
    ticker = ticker.strip().upper()
    slug = slug.strip().lower()
    return f"https://macrotrends.net/stocks/charts/{ticker}/{slug}/pe-ratio"


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def _clean_number(value: str) -> Optional[float]:
    s = value.strip()
    if not s or s in {"-", "N/A", "NaN"}:
        return None
    s = s.replace(",", "").replace("$", "")
    # Handle negative shown with parentheses
    if s.startswith("(") and s.endswith(")"):
        s = f"-{s[1:-1]}"
    try:
        return float(s)
    except ValueError:
        return None


def _parse_table_by_pandas(html: str) -> Optional[pd.DataFrame]:
    # Pandas can read tables robustly; pick the one with expected columns
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None
    expected = {"Date", "Stock Price", "TTM Net EPS", "PE Ratio"}
    for df in tables:
        cols = {str(c).strip() for c in df.columns}
        if expected.issubset(cols):
            return df
    return None


def _parse_table_by_bs4(html: str) -> Optional[pd.DataFrame]:
    # Fallback: directly find the historical data table by its header text
    soup = BeautifulSoup(StringIO(html), "html.parser")
    header_re = re.compile(r"PE Ratio Historical Data", re.I)
    header = soup.find("th", string=header_re).parent.parent
    if not header:
        return None
    tbody = header.find_next("tbody")
    if not tbody:
        return None
    # Build rows
    rows: list[list[str]] = []
    headers: list[str] = []
    thead = header.next_sibling.next
    if thead and thead.find_all("th"):
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    if not headers and rows:
        headers = rows.pop(0)
    if not headers or not rows:
        return None
    df = pd.DataFrame(rows, columns=headers)
    return df


def parse_pe_table(html: str) -> dict[str, dict[str, Optional[float]]]:
    df = _parse_table_by_pandas(html)
    if df is None:
        df = _parse_table_by_bs4(html)
    if df is None:
        raise RuntimeError("Failed to locate the PE Ratio Historical Data table on the page")

    # Normalize expected columns
    rename_map = {
        "Date": "date",
        "Stock Price": "stock_price",
        "TTM Net EPS": "ttm_eps",
        "PE Ratio": "pe_ratio",
    }
    # Some pages may include currency signs in column names—strip whitespace
    cols = {c: str(c).strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    for src, dst in rename_map.items():
        if src not in df.columns:
            raise RuntimeError(f"Expected column '{src}' not found in table columns: {list(df.columns)}")
    df = df[["Date", "Stock Price", "TTM Net EPS", "PE Ratio"]].copy()

    # Clean data
    records: dict[str, dict[str, Optional[float]]] = {}
    for _, row in df.iterrows():
        date_raw = str(row["Date"]).strip()
        # Try parse known formats
        dt: Optional[str] = None
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%b %d, %Y", "%Y-%m", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(date_raw, fmt).date().isoformat()
                break
            except ValueError:
                continue
        if dt is None:
            # Fallback: keep as-is
            dt = date_raw

        price = _clean_number(str(row["Stock Price"]))
        eps = _clean_number(str(row["TTM Net EPS"]))
        pe = _clean_number(str(row["PE Ratio"]))

        if dt:
            records[dt] = {
                "stock_price": price,
                "ttm_eps": eps,
                "pe_ratio": pe,
            }

    return records


if __name__ == "__main__":
    with get_session() as session:
        rows = session.query(InstrumentYahoo).options(selectinload(InstrumentYahoo.instrument)).all()

        for row in rows:
            ticker = row.instrument.yahoo_symbol
            slug = row.instrument.name.lower().replace(" ", "-")
            url = f"https://macrotrends.net/stocks/charts/{ticker}/{slug}/pe-ratio"
            print(url, row.instrument_id)
            pes = parse_pe_table(fetch_html(url))
            if pes:
                row.pes = pes
                row.updated_at = datetime.now(TIMEZONE)
                session.commit()
                # import sys, json
                # json.dump(row.pes, sys.stdout, indent=2)

            sleep(15)
