"""
Currency conversion service for Trading212 Portfolio Manager
==========================================================
Handles currency conversion with database caching.
"""

import logging
from datetime import date
from typing import Dict, Optional

from forex_python.converter import CurrencyRates
from database import get_db_service
from config import FALLBACK_RATES

logger = logging.getLogger(__name__)


class CurrencyService:
    """Handles currency conversion with database caching."""
    
    def __init__(self):
        self.db_service = get_db_service()
        self.converter = CurrencyRates()
        self.fallback_rates = FALLBACK_RATES
    
    def get_currency_table(self) -> Dict[str, float]:
        """
        Get GBP conversion rates for all supported currencies.
        Uses database cache with fallback to API and hardcoded rates.
        """
        today = date.today()
        table = {"GBX": 0.01, "GBP": 1.0}  # Base currencies
        currency_source_ready = True
        
        # Try to get rates from database first
        for currency in ["USD", "EUR", "CAD"]:
            cached_rate = self.db_service.get_currency_rate(currency, "GBP", today)
            if cached_rate:
                table[currency] = cached_rate
                continue
            
            # Try to fetch from API
            rate = None
            if currency_source_ready:
                try:
                    rate = self.converter.get_rate(currency, "GBP")
                    self.db_service.save_currency_rate(currency, "GBP", rate, today)
                    table[currency] = rate
                    logger.info("Fetched fresh rate for %s/GBP: %f", currency, rate)
                except Exception as exc:
                    currency_source_ready = False
                    logger.warning("Currency API failed for %s, using fallback rate %f: %s",
                                   currency, self.fallback_rates[currency], exc)

            if not rate:
                # Use fallback rate
                table[currency] = self.fallback_rates[currency]

        return table
    
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
    
    def get_rate(self, from_currency: str, to_currency: str = "GBP") -> Optional[float]:
        """Get conversion rate between currencies."""
        if from_currency == to_currency:
            return 1.0
        
        if to_currency != "GBP":
            logger.warning("Only GBP conversion is currently supported")
            return None
        
        rates = self.get_currency_table()
        return rates.get(from_currency)
    
    def refresh_rates(self) -> Dict[str, float]:
        """Force refresh of all currency rates from API."""
        today = date.today()
        table = {"GBX": 0.01, "GBP": 1.0}
        
        for currency in ["USD", "EUR", "CAD"]:
            try:
                rate = self.converter.get_rate(currency, "GBP")
                self.db_service.save_currency_rate(currency, "GBP", rate, today)
                table[currency] = rate
                logger.info("Refreshed rate for %s/GBP: %f", currency, rate)
            except Exception as exc:
                fallback_rate = self.fallback_rates.get(currency)
                if fallback_rate:
                    table[currency] = fallback_rate
                    logger.warning("Failed to refresh %s rate, using fallback: %s", currency, exc)
                else:
                    logger.error("No fallback rate for %s", currency)
        
        return table


# Global currency service instance
_currency_service: Optional[CurrencyService] = None


def get_currency_service() -> CurrencyService:
    """Get or create the global currency service instance."""
    global _currency_service
    if _currency_service is None:
        _currency_service = CurrencyService()
    return _currency_service


def currency_table() -> Dict[str, float]:
    """Legacy function for backward compatibility."""
    return get_currency_service().get_currency_table()
