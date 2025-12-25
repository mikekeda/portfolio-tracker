import asyncio
from celery_tasks.celery_app import app
from scripts.backfill_portfolio_daily import backfill_portfolio_daily
from scripts.scrape_macrotrends_pe import update_pe_data
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
