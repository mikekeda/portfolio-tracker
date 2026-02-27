import asyncio
from celery_tasks.celery_app import app
from scripts.backfill_portfolio_daily import backfill_portfolio_daily
from scripts.get_earnings_reports import get_earnings_reports
from scripts.scrape_13f import main as scrape_13f_main
from scripts.scrape_wisesheets_pe import update_pe_data
from scripts.update_data import update_data
from scripts.update_market_metrics import update_market_metrics
from scripts.update_pies import update_pies


@app.task
def calculate_portfolio_returns_task():
    backfill_portfolio_daily(rebuild=False)


@app.task
def update_data_task():
    update_data()


@app.task
def update_market_metrics_task():
    asyncio.run(update_market_metrics())


@app.task
def update_pe_data_task():
    """Update PE ratio historical data from Macrotrends for instruments with oldest PE data."""
    update_pe_data(limit=100)


@app.task
def update_pies_task():
    """Update Trading212 pies data from API."""
    update_pies()


@app.task
def fetch_earnings_reports_task(limit: int = 100):
    """Fetch and process earnings reports from SEC EDGAR API."""
    get_earnings_reports(limit=limit)


@app.task
def scrape_13f_task():
    """Scrape SEC 13F institutional holdings for all configured investors.

    Runs weekly — 13F filings are quarterly (deadlines: May 15, Aug 14, Nov 14, Feb 14).
    The script fetches only missing quarters, so this is a no-op when already up to date.
    """
    scrape_13f_main()
