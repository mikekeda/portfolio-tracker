import sys
import os
from celery import Celery
from celery.schedules import crontab

sys.path.insert(1, os.getcwd())

app = Celery("T212")
app.config_from_object("config", namespace="CELERY")

app.autodiscover_tasks(["celery_tasks"])

# Market-aware scheduling:
# - Market hours: 08:00-20:00 UTC (covers LSE 08:00-16:30 and US markets 13:30-20:00)
# - Midnight updates (00:00) added to prevent errors on pages when data is missing after midnight
# - Tasks run during market hours on weekdays and during market hours time on weekends
#
# update_data_task:
#   - Weekdays: Every 2 hours during market hours + midnight (0:00, 8:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00)
#   - Weekends: Every 4 hours + midnight (0:00, 8:00, 12:00, 16:00, 20:00)
#
# calculate_portfolio_returns_task:
#   - Weekdays: Every 4 hours, 5 min after update_data (0:05, 8:05, 12:05, 16:05, 20:05)
#   - Weekends: Every 8 hours, 5 min after update_data (0:05, 8:05, 16:05)
# When schedules overlap, calculate_portfolio_returns runs 5 minutes after update_data

app.conf.beat_schedule = {
    # Update data during market hours (Mon-Fri): Every 2 hours + midnight to prevent missing data errors
    # Runs at 0:00, 8:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00
    "update_data_market_hours": {
        "task": "celery_tasks.tasks.update_data_task",
        "schedule": crontab(minute=0, hour="0,8,10,12,14,16,18,20,22", day_of_week="mon-fri"),
        "args": (),
    },
    # Update data weekends: Every 4 hours + midnight to prevent missing data errors
    # Runs at 0:00, 8:00, 12:00, 16:00, 20:00
    "update_data_weekends": {
        "task": "celery_tasks.tasks.update_data_task",
        "schedule": crontab(minute=0, hour="0,8,12,16,20", day_of_week="sat-sun"),
        "args": (),
    },
    # Calculate portfolio returns during market hours: Every 4 hours, staggered 5 min after update_data
    # Runs at 0:05, 8:05, 12:05, 16:05, 20:05 (5 minutes after update_data at 0:00, 8:00, 12:00, 16:00, 20:00)
    "calculate_portfolio_returns_market_hours": {
        "task": "celery_tasks.tasks.calculate_portfolio_returns_task",
        "schedule": crontab(minute=5, hour="0,8,12,16,20", day_of_week="mon-fri"),
        "args": (),
    },
    # Calculate portfolio returns weekends: Every 8 hours, staggered 5 min after update_data
    # Runs at 0:05, 8:05, 16:05 (5 minutes after update_data at 0:00, 8:00, 16:00)
    "calculate_portfolio_returns_weekends": {
        "task": "celery_tasks.tasks.calculate_portfolio_returns_task",
        "schedule": crontab(minute=5, hour="0,8,16", day_of_week="sat-sun"),
        "args": (),
    },
    # Update market metrics during market hours: Every 4 hours
    # Runs at 0:30, 8:30, 12:30, 16:30, 20:30 (30 minutes after update_data)
    "update_market_metrics_market_hours": {
        "task": "celery_tasks.tasks.update_market_metrics_task",
        "schedule": crontab(minute=30, hour="0,8,12,16,20", day_of_week="mon-fri"),
        "args": (),
    },
    # Update market metrics weekends: Every 12 hours
    # Runs at 0:30, 12:30
    "update_market_metrics_weekends": {
        "task": "celery_tasks.tasks.update_market_metrics_task",
        "schedule": crontab(minute=30, hour="0,12", day_of_week="sat-sun"),
        "args": (),
    },
    # Update PE data from Macrotrends: Daily at 2 AM UTC (night time)
    # Runs every day at 2:00 AM UTC to update PE history for instruments with oldest data
    "update_pe_data_nightly": {
        "task": "celery_tasks.tasks.update_pe_data_task",
        "schedule": crontab(minute=0, hour=2),
        "args": (),
    },
    # Update Trading212 pies: Daily at 3 AM UTC (night time)
    # Runs every day at 3:00 AM UTC to update pie data from Trading212 API
    "update_pies_nightly": {
        "task": "celery_tasks.tasks.update_pies_task",
        "schedule": crontab(minute=0, hour=3),
        "args": (),
    },
    # Fetch earnings reports from SEC EDGAR: Daily at 4 AM UTC (night time)
    # Runs every day at 4:00 AM UTC to fetch latest 10-Q/10-K filings and generate LLM summaries
    "fetch_earnings_reports_nightly": {
        "task": "celery_tasks.tasks.fetch_earnings_reports_task",
        "schedule": crontab(minute=0, hour=4),
        "args": (100,),
    },
    # Fetch UK RNS earnings reports from Investegate: Daily at 4:30 AM UTC
    # Staggered 30 min after the SEC run to avoid overlapping LLM API quota and
    # to keep the two pipelines' logs interleaved-but-separable. Targets .L
    # instruments without a populated CIK.
    "fetch_uk_earnings_reports_nightly": {
        "task": "celery_tasks.tasks.fetch_uk_earnings_reports_task",
        "schedule": crontab(minute=30, hour=4),
        "args": (100,),
    },
    # Scrape SEC 13F institutional holdings: Weekly on Monday at 1 AM UTC
    # 13F filings are due quarterly (~45 days after quarter end: May 15, Aug 14, Nov 14, Feb 14).
    # Weekly cadence catches new filings within days of publication; no-op when already up to date.
    "scrape_13f_weekly": {
        "task": "celery_tasks.tasks.scrape_13f_task",
        "schedule": crontab(minute=0, hour=1, day_of_week="mon"),
        "args": (),
    },
    # Sync transaction + dividend history from Trading212 API: Daily at 5 AM UTC.
    # Upserts by reference ID — idempotent, safe to re-run.
    # Runs after other nightly tasks (PE at 2am, pies at 3am, earnings at 4am).
    "sync_transactions_nightly": {
        "task": "celery_tasks.tasks.sync_transactions_task",
        "schedule": crontab(minute=0, hour=5),
        "args": (),
    },
}
