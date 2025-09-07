"""
Currency conversion service for Trading212 Portfolio Manager
==========================================================
Handles currency conversion with database caching.
"""

import logging
import requests
from datetime import date
from typing import Dict, Optional

from database import get_db_service
from config import FALLBACK_RATES

logger = logging.getLogger(__name__)


class CurrencyService:
    """Handles currency conversion with database caching."""

    def __init__(self):
        self.db_service = get_db_service()
        self.fallback_rates = FALLBACK_RATES

    def get_currency_table(self) -> Dict[str, float]:
        """
        Get GBP conversion rates for all supported currencies.
        Uses database cache with fallback to API and hardcoded rates.
        """
        today = date.today()
        table = {"GBX": 0.01, "GBP": 1.0}  # Base currencies

        # Try to get rates from database first
        currencies_to_fetch = []
        for currency in ["USD", "EUR", "CAD"]:
            cached_rate = self.db_service.get_currency_rate(currency, "GBP", today)
            if cached_rate:
                table[currency] = cached_rate
            else:
                currencies_to_fetch.append(currency)

        # Fetch missing rates from API in batch
        if currencies_to_fetch:
            try:
                rates = self._fetch_rates_batch(currencies_to_fetch)
                for currency, rate in rates.items():
                    if rate:
                        self.db_service.save_currency_rate(currency, "GBP", rate, today)
                        table[currency] = rate
                        logger.info("Fetched fresh rate for %s/GBP: %f", currency, rate)
                    else:
                        table[currency] = self.fallback_rates[currency]
                        logger.warning("Using fallback rate for %s: %f", currency, self.fallback_rates[currency])
            except Exception as exc:
                logger.warning("Currency API failed, using fallback rates: %s", exc)
                for currency in currencies_to_fetch:
                    table[currency] = self.fallback_rates[currency]

        return table

    def _fetch_rates_batch(self, currencies: list) -> Dict[str, float]:
        """Fetch currency rates from a reliable API in batch."""
        rates = {}

        # Use exchangerate-api.com (free tier available)
        try:
            url = "https://api.exchangerate-api.com/v4/latest/GBP"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Convert from GBP to other currencies (invert the rates)
            for currency in currencies:
                if currency in data['rates']:
                    # Invert the rate since we want TO GBP, not FROM GBP
                    rates[currency] = 1.0 / data['rates'][currency]
                else:
                    rates[currency] = None

        except Exception as e:
            logger.warning("exchangerate-api.com failed: %s", e)

            # Fallback to a simpler API
            try:
                for currency in currencies:
                    url = f"https://api.exchangerate.host/latest?base={currency}&symbols=GBP"
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    if 'rates' in data and 'GBP' in data['rates']:
                        rates[currency] = data['rates']['GBP']
                    else:
                        rates[currency] = None

            except Exception as e2:
                logger.warning("exchangerate.host also failed: %s", e2)
                for currency in currencies:
                    rates[currency] = None

        return rates

    def convert_to_gbp(self, amount: float, from_currency: str) -> float:
        """Convert amount from given currency to GBP."""
        if from_currency == "GBP":
            return amount
        elif from_currency == "GBX":
            return amount * 0.01

        rates = self.get_currency_table()
        rate = rates.get(from_currency)

        if rate is None:
            logger.error("No conversion rate available for %s", from_currency)
            return amount  # Return original amount as fallback

        return amount * rate


# Global currency service instance
_currency_service: Optional[CurrencyService] = None


def get_currency_service() -> CurrencyService:
    """Get or create the global currency service instance."""
    global _currency_service
    if _currency_service is None:
        _currency_service = CurrencyService()
    return _currency_service
