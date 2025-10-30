from celery import Celery

app = Celery("T212")
app.config_from_object("config", namespace="CELERY")

app.autodiscover_tasks(["celery_tasks"])

app.conf.beat_schedule = {
    "every-hour": {
        "task": "celery_tasks.update_data.update_data_task",
        "schedule": 14400.0,  # every 4h
        "args": (),
    },
}
