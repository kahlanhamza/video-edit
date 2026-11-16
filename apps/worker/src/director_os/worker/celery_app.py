"""Celery app — Redis broker/backend (ADR-0004).

Task signatures are written so a Temporal workflow can wrap them unchanged in M2/M5.
"""

from __future__ import annotations

from director_os.core.config import get_settings


def make_app(broker: str | None = None, backend: str | None = None):
    settings = get_settings()
    url = broker or settings.redis_url
    from celery import Celery

    app = Celery(
        "directoros",
        broker=url,
        backend=backend or url,
        include=["director_os.worker.tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_default_queue="directoros",
        worker_max_tasks_per_child=20,  # ffmpeg child procs like a fresh worker
    )
    return app


celery_app = make_app()
