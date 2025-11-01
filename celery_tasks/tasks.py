from celery_tasks.celery_app import app
from scripts.update_data import update_data


@app.task
def update_data_task():
    update_data()
