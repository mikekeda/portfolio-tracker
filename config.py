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
TRADING212_API_BASE = "https://live.trading212.com"
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

# SEC EDGAR requires User-Agent in the form "Company Name email@example.com".
SEC_USER_AGENT = "PortfolioTracker/1.0 (admin@example.com)"

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
# Equity Risk Premium — extra return equity investors demand over risk-free.
# Damodaran implied ERP (Jan 2025): ~4.6%. Historical arithmetic avg ~6.4%, geometric ~4.2%.
# Reference: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/implprem.html
EQUITY_RISK_PREMIUM = 0.046
# GBP risk-free rate — approximate UK 1yr gilt yield. Update when BoE base rate changes materially.
# BoE cut base rate to 4.25% on 8 May 2025; 1yr gilt was ~4.1–4.2% at that time.
# Verify current yield: https://www.bankofengland.co.uk/statistics/yield-curves (Short Sterling, 1Y row)
RISK_FREE_RATE = 0.04  # ~4.0% — verify against current gilt yield
# Terminal growth rate for DCF — long-run nominal GDP growth. Update if macro regime shifts.
# Reference: https://www.imf.org/en/Publications/WEO (World Economic Outlook, long-run GDP)
TERMINAL_GROWTH_RATE = 0.025  # 2.5%
# Soft caps (% of portfolio) for tag-cluster exposure — advisory review triggers
# for correlated-theme concentration, not hard limits. Tags overlap by design
# (NVDA is semiconductor+ai+growth), so rows don't sum to 100%. Served to the
# frontend via /api/portfolio/allocations; update as risk appetite changes.
# growth is the declared strategy, not a risk cluster: its cap guards the ~25%
# non-growth ballast floor (banks, defense primes, UKDV, gold), so the alarm
# only rings when diversifiers get sold down, not when growth rallies.
# Tested against value weight by _cluster_headroom_gbp, which exempts ETFs from
# every entry but `etf`. Risk share is a separate question — see TAG_RISK_ALERT_PCT.
TAG_CLUSTER_SOFT_CAPS: dict[str, float] = {
    "ai": 45,
    "semiconductor": 35,
    "growth": 75,
    "defense": 20,
    "space": 15,
    "speculative": 20,
    "software": 15,
    "cloud": 20,
    "healthcare": 12,
    "financial": 12,
    "EU": 30,
    # Deliberately slack: the glide path targets an index sleeve past 50% (STRATEGY.md §6.1).
    "etf": 80,
    "commodity": 5,
}

# Risk-share alarms for the Risk page — counts ETFs and never blocks a trade.
# Set above current readings so an alert means deterioration, except `speculative`
# and `space`, which fire today on ~22%/18% of risk for ~12%/9% of the money.
TAG_RISK_ALERT_PCT: dict[str, float] = {
    "ai": 60,
    "semiconductor": 55,
    "growth": 80,
    "defense": 25,
    "space": 15,
    "speculative": 20,
    "software": 20,
    "cloud": 25,
    "healthcare": 15,
    "financial": 15,
    "EU": 35,
    "etf": 40,
    "commodity": 8,
}

# MSCI World long-run reference figures used on the Projection page (1970–present, rough expectations).
# Nominal 8.5% / inflation ~2.5% ≈ real 6.5%. σ ~16% p.a. from monthly returns.
# Reference: https://www.msci.com/documents/10199/178e6643-6ae6-47b9-82be-e1fc565ededb
PROJECTION_BENCHMARK_NOMINAL_RETURN = 0.085
PROJECTION_BENCHMARK_REAL_RETURN = 0.065
PROJECTION_BENCHMARK_VOLATILITY = 0.16
# UK ISA annual subscription limit — set by HMRC each tax year.
# Reference: https://www.gov.uk/individual-savings-accounts
ISA_ANNUAL_ALLOWANCE = 20_000.0  # £20,000 for 2025/26 (frozen since 2017/18)
# Trade-suggestion agent (backend/agent) — deterministic safety caps applied to
# every strategy's output, and by the backtest engine when simulating fills.
# FX fee reference: https://www.trading212.com/terms/invest (0.15%, no commission)
T212_FX_FEE = 0.0015  # charged on non-GBP notional per trade
# UK Stamp Duty Reserve Tax — 0.5% on purchases of UK shares; LSE-listed ETFs
# (Irish-domiciled) are exempt. Confirmed against transaction_history fees.
# Reference: https://www.gov.uk/tax-buy-shares
UK_STAMP_DUTY = 0.005
# French financial transaction tax on purchases of large-cap French (.PA) shares —
# 0.3% until 2025, 0.4% since (both rates observed in transaction_history).
FRENCH_FTT = 0.004
AGENT_MAX_DAILY_TURNOVER = 0.05  # per side (sells and buys each), fraction of portfolio value
AGENT_MIN_HOLDINGS = 5
AGENT_MAX_POSITION_WEIGHT = 0.16  # NVDA is ~15% today; cap must not force an immediate trim
AGENT_MIN_TRADE_GBP = 150.0
# Days per year constants for annualisation
DAYS_PER_YEAR = 365.25  # Gregorian calendar average; used for TWRR/MWRR/alpha annualisation
TRADING_DAYS_PER_YEAR = 252  # US/UK equity market trading days; used for volatility scaling

# USD/EUR/CAD are instrument listing currencies; the rest are ADR reporting
# currencies, needed wherever a statement figure meets a market value.
CURRENCIES = ("USD", "EUR", "CAD", "SEK", "DKK", "JPY", "CNY", "TWD", "BRL", "MXN", "INR", "KRW", "CHF", "PLN")

# CELERY STUFF
CELERY_BROKER_URL = "redis://localhost:6379/11"
CELERY_result_backend = "redis://localhost:6379/11"
CELERY_accept_content = ["application/json"]
CELERY_task_serializer = "json"
CELERY_result_serializer = "json"
CELERY_timezone = "UTC"
