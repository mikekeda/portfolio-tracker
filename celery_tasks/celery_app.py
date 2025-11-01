import sys
import os
from celery import Celery

sys.path.insert(1, os.getcwd())

app = Celery("T212")
app.config_from_object("config", namespace="CELERY")

app.autodiscover_tasks(["celery_tasks"])

app.conf.beat_schedule = {
    "every-hour": {
        "task": "celery_tasks.tasks.update_data_task",
        "schedule": 14400.0,  # every 4h
        "args": (),
    },
}
