"""
Configuration for Trading212 Portfolio Manager
==============================================
Centralized configuration settings.
"""

import os

# API Configuration
TRADING212_API_KEY = os.environ["TRADING212_API_KEY"]

# Database Configuration
DB_NAME = os.getenv("DB_NAME", "trading212_portfolio")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Yahoo Finance Configuration
PRICE_FIELD = "Adj Close"  # or "close_price" if you prefer raw closes
BATCH_SIZE_YF = 25  # tickers per yahoo request
REQUEST_RETRY = 5
REQUEST_SLEEP = 1.0  # polite pause between Yahoo calls
HISTORY_YEARS = 10

# Portfolio Configuration
BENCH = "VUAG.L"  # S&P500
RISK_FREE = 0.045  # annual risk-free guess (4.5 %); replace with 3-month T-bill

# Display Configuration
MAX_HOLDINGS_DISPLAY = 20
MAX_COUNTRIES_DISPLAY = 10
MAX_SECTORS_DISPLAY = 10
MAX_PERFORMERS_DISPLAY = 5

# Currency Configuration
SUPPORTED_CURRENCIES = ["GBX", "GBP", "USD", "EUR", "CAD"]
FALLBACK_RATES = {
    "GBX": 0.01,
    "GBP": 1.0,
    "USD": 0.74,
    "EUR": 0.87,
    "CAD": 0.54,
}

# Color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Validation
def validate_config():
    """Validate that all required configuration is present."""
    if not TRADING212_API_KEY:
        raise ValueError("TRADING212_API_KEY environment variable is required")

    if not DB_PASSWORD:
        raise ValueError("DB_PASSWORD environment variable is required")

    return True
