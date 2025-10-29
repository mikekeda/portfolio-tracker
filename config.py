"""
Configuration for Trading212 Portfolio Manager
==============================================
Centralized configuration settings.
"""

import os
import re
from datetime import timezone

import requests

SITE_ENV_PREFIX = "T212"


def get_env_var(name: str, default: str = "") -> str:
    """Get all sensitive data from google vm custom metadata."""
    try:
        name = f"{SITE_ENV_PREFIX}_{name}"
        env_var = os.environ.get(name)
        if env_var is not None:
            # Check env variable (Jenkins build).
            return env_var
        else:
            res = requests.get(
                f"http://metadata.google.internal/computeMetadata/v1/instance/attributes/{name}",
                headers={"Metadata-Flavor": "Google"},
            )
            if res.status_code == 200:
                return res.text
    except requests.exceptions.ConnectionError:
        pass

    return default


TIMEZONE = timezone.utc

# API Configuration
TRADING212_API_KEY = get_env_var("TRADING212_API_KEY")

# Database Configuration
DB_NAME = get_env_var("DB_NAME", "trading212_portfolio")
DB_PASSWORD = get_env_var("DB_PASSWORD")
DB_USER = get_env_var("DB_USER", "postgres")
DB_HOST = get_env_var("DB_HOST", "localhost")
DB_PORT = get_env_var("DB_PORT", "5432")

API_TOKEN = get_env_var("API_TOKEN")
DOMAIN = get_env_var("DOMAIN", "http://localhost:3000")
FRED_API_KEY = get_env_var("FRED_API_KEY")

# Yahoo Finance Configuration
PRICE_FIELD = "Adj Close"  # or "close_price" if you prefer raw closes
BATCH_SIZE_YF = 50  # tickers per yahoo request
REQUEST_RETRY = 5
HISTORY_YEARS = 10

# Portfolio Configuration
PATTERN_MULTI = re.compile(r"^(?P<sym>.+?)_(?P<tag>[A-Z]{2,3})$")
SPY = "VUAG.L"
BENCHES = (SPY, "XNAS.L")  # VUAG.L - S&P500, XNAS.L - QQQ
VIX = "^VIX"  # VIX index

# Currency Configuration
CURRENCIES = ("USD", "EUR", "CAD")
