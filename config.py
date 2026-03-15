"""
Configuration for Trading212 Portfolio Manager
==============================================
Centralized configuration settings.
"""

import logging
import os
import re
from datetime import timezone as dt_timezone

import requests

SITE_ENV_PREFIX = "T212"

# Configure logging
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_env_var(name: str, default: str = "") -> str:
    """Get sensitive data from env vars, Oracle Cloud IMDS, or Google Cloud metadata."""
    name = f"{SITE_ENV_PREFIX}_{name}"

    env_var = os.environ.get(name)
    if env_var is not None:
        return env_var

    # Try Oracle Cloud IMDS (only reachable on OCI instances)
    try:
        res = requests.get(
            f"http://169.254.169.254/opc/v2/instance/metadata/{name}",
            headers={"Authorization": "Bearer Oracle"},
            timeout=2,
        )
        if res.status_code == 200:
            return res.text.strip()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass

    # Try Google Cloud metadata (only reachable on GCP instances)
    try:
        res = requests.get(
            f"http://metadata.google.internal/computeMetadata/v1/instance/attributes/{name}",
            headers={"Metadata-Flavor": "Google"},
            timeout=2,
        )
        if res.status_code == 200:
            return res.text.strip()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass

    return default


TIMEZONE = dt_timezone.utc

DEBUG = get_env_var("DEBUG")
SOCKET_FILE = get_env_var("SOCKET_FILE", "/temp/site.sock")

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
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")

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
# Equity Risk Premium (Historical average ~5%)
EQUITY_RISK_PREMIUM = 0.05

# Currency Configuration
CURRENCIES = ("USD", "EUR", "CAD")

# CELERY STUFF
CELERY_BROKER_URL = "redis://localhost:6379/11"
CELERY_result_backend = "redis://localhost:6379/11"
CELERY_accept_content = ["application/json"]
CELERY_task_serializer = "json"
CELERY_result_serializer = "json"
CELERY_timezone = "UTC"
