"""
Configuration for Trading212 Portfolio Manager
==============================================
Centralized configuration settings.
"""

import os
import re

from datetime import timezone

TIMEZONE = timezone.utc

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
BATCH_SIZE_YF = 50  # tickers per yahoo request
REQUEST_RETRY = 5
HISTORY_YEARS = 10

# Portfolio Configuration
PATTERN_MULTI = re.compile(r"^(?P<sym>.+?)_(?P<tag>[A-Z]{2,3})$")
BENCH = "VUAG.L"  # S&P500
VIX = "^VIX"  # VIX index

# Currency Configuration
CURRENCIES = ("USD", "EUR", "CAD")
