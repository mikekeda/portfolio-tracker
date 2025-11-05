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
# - Tasks only run during market hours on weekdays and during market hours time on weekends
#
# update_data_task:
#   - Weekdays: Every 2 hours during market hours (8:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00)
#   - Weekends: Every 4 hours during market hours time (8:00, 12:00, 16:00, 20:00)
#
# calculate_portfolio_returns_task:
#   - Weekdays: Every 4 hours during market hours, 5 min after update_data (8:05, 12:05, 16:05)
#   - Weekends: Every 8 hours, 5 min after update_data (8:05, 16:05)
# When schedules overlap, calculate_portfolio_returns runs 5 minutes after update_data

app.conf.beat_schedule = {
    # Update data during market hours (Mon-Fri, 08:00-20:00 UTC): Every 2 hours
    "update_data_market_hours": {
        "task": "celery_tasks.tasks.update_data_task",
        "schedule": crontab(minute=0, hour="8,10,12,14,16,18,20", day_of_week="mon-fri"),
        "args": (),
    },
    # Update data weekends (Sat-Sun): Every 4 hours during market hours time
    "update_data_weekends": {
        "task": "celery_tasks.tasks.update_data_task",
        "schedule": crontab(minute=0, hour="8,12,16,20", day_of_week="sat-sun"),
        "args": (),
    },
    # Calculate portfolio returns during market hours: Every 4 hours, staggered 5 min after update_data
    # Runs at 8:05, 12:05, 16:05 (5 minutes after update_data at 8:00, 12:00, 16:00)
    "calculate_portfolio_returns_market_hours": {
        "task": "celery_tasks.tasks.calculate_portfolio_returns_task",
        "schedule": crontab(minute=5, hour="8,12,16", day_of_week="mon-fri"),
        "args": (),
    },
    # Calculate portfolio returns weekends: Every 8 hours, staggered 5 min after update_data
    # Runs at 8:05, 16:05 (5 minutes after update_data at 8:00, 16:00)
    "calculate_portfolio_returns_weekends": {
        "task": "celery_tasks.tasks.calculate_portfolio_returns_task",
        "schedule": crontab(minute=5, hour="8,16", day_of_week="sat-sun"),
        "args": (),
    },
}
