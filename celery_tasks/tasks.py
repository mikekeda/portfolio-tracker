from celery_tasks.celery_app import app
from scripts.backfill_portfolio_daily import backfill_portfolio_daily
from scripts.update_data import update_data
from scripts.update_market_metrics import update_market_metrics


@app.task
def calculate_portfolio_returns_task():
    backfill_portfolio_daily(rebuild=False)


@app.task
def update_data_task():
    update_data()


@app.task
def update_market_metrics_task():
    update_market_metrics()
