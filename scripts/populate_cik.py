"""
Populate CIK Script
===================
Fetches the official SEC company tickers JSON and fills missing CIKs on Instruments.
US tickers only (yahoo_symbol matched to SEC tickers; UK .L symbols stay NULL).
"""

import requests
from sqlalchemy import select

from config import SEC_USER_AGENT, logger
from models import Instrument
from scripts.update_data import get_session

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch_sec_ticker_map() -> dict[str, str]:
    """
    Returns a dict mapping {TICKER: CIK_STRING}
    e.g. {'AAPL': '0000320193'}
    """
    logger.info("Fetching SEC tickers...")
    headers = {"User-Agent": SEC_USER_AGENT}
    resp = requests.get(SEC_TICKERS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Data structure: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    result = {}
    for entry in data.values():
        val_cik = str(entry["cik_str"]).zfill(10)
        val_ticker = entry["ticker"].upper()
        result[val_ticker] = val_cik

    logger.info("Loaded %s tickers from SEC.", len(result))
    return result


def populate_ciks():
    """Fill missing CIKs on instruments using SEC ticker data."""
    ticker_map = fetch_sec_ticker_map()

    with get_session() as session:
        instruments = session.scalars(select(Instrument).where(Instrument.cik.is_(None))).all()
        updated_count = 0
        no_sec_match = 0

        for inst in instruments:
            # Yahoo format: "AAPL", "BRK-B" or "BRK.B"
            # SEC format: "AAPL", "BRK-B"
            candidate = inst.yahoo_symbol.upper()

            # Handle "BRK.B" -> "BRK-B" normalization
            candidate_normalized = candidate.replace(".", "-")

            found_cik = ticker_map.get(candidate) or ticker_map.get(candidate_normalized)

            if not found_cik:
                no_sec_match += 1
                continue

            if inst.cik != found_cik:
                inst.cik = found_cik
                updated_count += 1

        logger.info(
            "CIK populate: candidates=%s, updated=%s, no SEC ticker match=%s",
            len(instruments),
            updated_count,
            no_sec_match,
        )


if __name__ == "__main__":
    populate_ciks()
