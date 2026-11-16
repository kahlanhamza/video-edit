"""API gateway health + projects (FastAPI TestClient) and Celery worker tasks."""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    get_settings.cache_clear()
    api_main._engine = None  # force re-bind to the tmp database
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "directoros-api"


def test_project_create_and_get(api_client):
    r = api_client.post("/projects", json={"title": "Case 1"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert r.json()["status"] == "created"
    r2 = api_client.get(f"/projects/{pid}")
    assert r2.status_code == 200
    assert r2.json()["title"] == "Case 1"
    assert api_client.get("/projects/nope").status_code == 404


def test_worker_tasks_eager():
    from director_os.worker.celery_app import celery_app
    from director_os.worker.tasks import ping, run_ffmpeg

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    assert ping.apply().get() == "pong"
    result = run_ffmpeg.apply(args=[["ffmpeg", "-version"]]).get()
    assert result["rc"] == 0
    with pytest.raises(ValueError):
        run_ffmpeg.apply(args=[["rm", "-rf", "/"]])
