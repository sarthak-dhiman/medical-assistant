from celery import Celery
from .config import REDIS_URL

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["web_app.backend.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker optimization
    worker_concurrency=1, # One worker since models are heavy and single-threaded usually better for stability
    worker_prefetch_multiplier=1
)
