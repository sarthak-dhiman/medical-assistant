from celery import Celery
from .config import settings

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["web_app.backend.tasks"]
)

from kombu import Queue

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    # Task Routing - Explicit Queues
    task_default_queue='celery',
    task_queues=(
        Queue('celery', routing_key='celery'),
        Queue('q_lightweight', routing_key='q_lightweight'),
        Queue('q_heavy_cv', routing_key='q_heavy_cv'),
    ),
    # Worker optimization
    worker_concurrency=1, 
    worker_prefetch_multiplier=1
)
