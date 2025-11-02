from celery_tasks.celery_app import app
from scripts.backfill_portfolio_daily import backfill_portfolio_daily
from scripts.update_data import update_data


@app.task
def calculate_portfolio_returns_task():
    backfill_portfolio_daily()


@app.task
def update_data_task():
    update_data()
