"""
Scrape Wisesheets PE ratio historical data for a given stock.
"""

import sys
from collections import defaultdict
from datetime import datetime
from time import sleep
from typing import Optional

# import pandas as pd  # Not needed for this scraper
from bs4 import BeautifulSoup  # type: ignore[import-untyped]
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import func, select
from sqlalchemy import text as sql_text
from sqlalchemy.orm import selectinload

from config import TIMEZONE, logger
from models import InstrumentYahoo
from scripts.update_data import get_session

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def build_url(ticker: str) -> str:
    ticker = ticker.strip().upper()
    return f"https://www.wisesheets.io/pe-ratio/{ticker}"


def fetch_html(url: str) -> str:
    """Fetch HTML from Wisesheets, handling the Quarterly/Annual toggle."""
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(f"--user-agent={HEADERS['User-Agent']}")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        # Wait for the page to load and look for the Quarterly (TTM) toggle
        wait = WebDriverWait(driver, 10)

        # Try to find and click the "Quarterly (TTM)" button
        try:
            # Try different selectors for the quarterly button
            selectors = [
                "//button[contains(text(), 'Quarterly (TTM)')]",
                "//button[contains(text(), 'Quarterly')]",
                "//button[contains(text(), 'TTM')]",
                "//button[contains(@class, 'quarterly')]",
                "//button[contains(@class, 'ttm')]",
                "//*[contains(text(), 'Quarterly (TTM)')]",
                "//*[contains(text(), 'Quarterly')]",
            ]

            quarterly_button = None
            for selector in selectors:
                try:
                    quarterly_button = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                    logger.debug(f"Found quarterly button with selector: {selector}", file=sys.stderr)
                    break
                except Exception:
                    continue

            if quarterly_button:
                quarterly_button.click()
                # Wait a moment for the data to update
                sleep(3)
                logger.debug("Successfully clicked quarterly button", file=sys.stderr)
            else:
                logger.warning("Could not find quarterly button with any selector", file=sys.stderr)

        except Exception as e:
            logger.warning(f"Could not click Quarterly (TTM) button: {e}", file=sys.stderr)
            # Continue anyway, we might still get some data

        # Get the updated HTML
        html = driver.page_source
        return html

    finally:
        if driver:
            driver.quit()


def _clean_number(value: str) -> Optional[float]:
    s = value.strip()
    if not s or s in {"-", "N/A", "NaN", ""}:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "")
    # Handle negative shown with parentheses
    if s.startswith("(") and s.endswith(")"):
        s = f"-{s[1:-1]}"
    try:
        return float(s)
    except ValueError:
        return None


def parse_pe_data(html: str) -> dict[str, Optional[float]]:
    """Parse PE ratio data from Wisesheets HTML."""
    soup = BeautifulSoup(html, "html.parser")
    records: dict[str, Optional[float]] = {}

    # Look for tables with Year and PE Ratio columns
    tables = soup.find_all("table")

    for table in tables:
        headers = table.find_all("th")
        header_texts = [h.get_text(strip=True) for h in headers]

        # Look for the historical PE table (Year, PE Ratio, Change)
        if "Year" in header_texts and "PE Ratio" in header_texts:
            rows = table.find_all("tr")[1:]  # Skip header row

            # Group rows by year to handle quarterly data
            year_data: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    year_text = cells[0].get_text(strip=True)
                    pe_text = cells[1].get_text(strip=True)

                    try:
                        year = int(year_text)
                        pe_value = _clean_number(pe_text)
                        if pe_value is not None:
                            year_data[year].append(pe_value)
                    except ValueError:
                        continue

            # Convert grouped data to quarterly dates
            for year, pe_values in year_data.items():
                if len(pe_values) == 1:
                    # Single value for the year - use end of year
                    records[f"{year}-12-31"] = pe_values[0]
                elif len(pe_values) == 4:
                    # Four values - quarterly data
                    quarters = [
                        f"{year}-03-31",  # Q1
                        f"{year}-06-30",  # Q2
                        f"{year}-09-30",  # Q3
                        f"{year}-12-31",  # Q4
                    ]
                    for i, pe_value in enumerate(pe_values):
                        records[quarters[i]] = pe_value
                else:
                    # Other number of values - distribute evenly
                    # For 2 values: Q2 and Q4
                    # For 3 values: Q2, Q3, Q4
                    if len(pe_values) == 2:
                        quarters = [f"{year}-06-30", f"{year}-12-31"]
                    elif len(pe_values) == 3:
                        quarters = [f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
                    else:
                        # More than 4 values - use monthly distribution
                        months_per_value = 12 // len(pe_values)
                        quarters = []
                        for i in range(len(pe_values)):
                            month = (i + 1) * months_per_value
                            if month == 12:
                                quarters.append(f"{year}-12-31")
                            elif month in [3, 6, 9]:
                                quarters.append(f"{year}-{month:02d}-30")
                            else:
                                quarters.append(f"{year}-{month:02d}-28")

                    for i, pe_value in enumerate(pe_values):
                        if i < len(quarters):
                            records[quarters[i]] = pe_value

            break  # Found the right table, no need to check others

    if not records:
        raise RuntimeError("Failed to locate PE ratio data on the Wisesheets page")

    return records


def _parse_date(date_text: str) -> Optional[str]:
    """Parse date text into ISO format string."""
    date_text = date_text.strip()

    # Try to parse as year only (e.g., "2024")
    try:
        year = int(date_text)
        return f"{year}-12-31"  # Default to end of year
    except ValueError:
        pass

    # Try to parse as year-month (e.g., "2024-03", "2024-Q1")
    if "-" in date_text:
        parts = date_text.split("-")
        if len(parts) == 2:
            try:
                year = int(parts[0])
                month_part = parts[1]

                # Handle quarter format (Q1, Q2, Q3, Q4)
                if month_part.startswith("Q"):
                    quarter = int(month_part[1:])
                    if quarter == 1:
                        return f"{year}-03-31"
                    elif quarter == 2:
                        return f"{year}-06-30"
                    elif quarter == 3:
                        return f"{year}-09-30"
                    elif quarter == 4:
                        return f"{year}-12-31"

                # Handle month format
                month = int(month_part)
                if 1 <= month <= 12:
                    # Use end of month
                    if month in [1, 3, 5, 7, 8, 10, 12]:
                        return f"{year}-{month:02d}-31"
                    elif month in [4, 6, 9, 11]:
                        return f"{year}-{month:02d}-30"
                    else:  # February
                        return f"{year}-02-28"  # Assume non-leap year
            except ValueError:
                pass

    # Try to parse as full date (e.g., "2024-03-15")
    try:
        dt = datetime.strptime(date_text, "%Y-%m-%d")
        return dt.date().isoformat()
    except ValueError:
        pass

    return None


def update_pe_data(limit: int = 100) -> None:
    """Update PE ratio historical data from Wisesheets for instruments with oldest PE data."""
    with get_session() as session:
        # Sort tickers by pe date, old first, get pe for the first N tickers
        last_pe_date_expr = (
            select(func.max(sql_text("key")))
            .select_from(func.jsonb_object_keys(InstrumentYahoo.pes).alias("key"))
            .correlate(InstrumentYahoo)
            .scalar_subquery()
        )

        query = (
            select(InstrumentYahoo)
            .where(InstrumentYahoo.pes != {})
            .order_by(last_pe_date_expr.nulls_first())
            .limit(limit)
            .options(selectinload(InstrumentYahoo.instrument))
        )

        rows = session.execute(query).scalars().all()
        tickers = [row.instrument.yahoo_symbol for row in rows][:10]
        logger.info(f"Found {len(rows)} instruments to update: {tickers},...")

        for row in rows:
            ticker = row.instrument.yahoo_symbol
            url = build_url(ticker)

            try:
                html = fetch_html(url)
                pe_data = parse_pe_data(html)

                # Convert to the same format as Macrotrends scraper
                formatted_data = {}
                for date, pe_value in pe_data.items():
                    formatted_data[date] = {
                        "stock_price": None,  # Not available from Wisesheets
                        "ttm_eps": None,  # Not available from Wisesheets
                        "pe_ratio": pe_value,
                    }

                row.pes = formatted_data
                row.updated_at = datetime.now(TIMEZONE)
                session.commit()

            except Exception as e:
                logger.error(f"Error scraping {ticker} (instrument_id={row.instrument_id}): {e}", exc_info=True)
                session.rollback()
                sleep(20)
                continue

            sleep(15)  # Be respectful to the server


if __name__ == "__main__":
    update_pe_data(limit=10)
