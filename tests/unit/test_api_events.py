"""SSE job progress — streaming endpoint contract (FastAPI TestClient)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_events.db")
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


def _make_project_job(api_client, tmp_path, *, progress: float, error: str | None = None):
    import director_os.api.main as api_main
    from director_os.core.db import JobRow

    r = api_client.post("/projects", json={"title": "SSE case"})
    pid = r.json()["id"]
    s = api_main.session()
    job = JobRow(project_id=pid, current_stage="render", progress_pct=progress)
    job.error_message = error
    s.add(job)
    s.commit()
    return pid, job.id


def _stream(api_client, url: str, **params) -> str:
    with api_client.stream("GET", url, params=params) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        return "".join(chunk for chunk in r.iter_text())


def test_sse_terminal_job_emits_progress_then_done(api_client, tmp_path):
    pid, jid = _make_project_job(api_client, tmp_path, progress=100.0)
    text = _stream(api_client, f"/projects/{pid}/jobs/{jid}/events", max_seconds=1.0)
    assert "event: progress" in text
    assert "event: done" in text
    assert '"stage":"render"' in text.replace(" ", "")
    assert '"progress_pct":100.0' in text


def test_sse_error_is_terminal(api_client, tmp_path):
    pid, jid = _make_project_job(api_client, tmp_path, progress=90.0, error="qa_failed: loudness")
    text = _stream(api_client, f"/projects/{pid}/jobs/{jid}/events", max_seconds=1.0)
    assert "event: done" in text
    assert "qa_failed" in text


def test_sse_non_terminal_job_updates_then_times_out(api_client, tmp_path):
    pid, jid = _make_project_job(api_client, tmp_path, progress=60.0)
    text = _stream(
        api_client,
        f"/projects/{pid}/jobs/{jid}/events",
        poll_s=0.05,
        max_seconds=0.4,
    )
    assert "event: progress" in text
    # hard cap always closes the stream with a timeout frame
    assert "event: done" in text and '"reason":"timeout"' in text
    # progress stayed at 60 (nothing advanced it) but the stream still closed
    assert '"progress_pct":60.0' in text


def test_sse_unknown_job_404_before_stream(api_client, tmp_path):
    pid, _ = _make_project_job(api_client, tmp_path, progress=0.0)
    r = api_client.get(f"/projects/{pid}/jobs/does-not-exist/events")
    assert r.status_code == 404


def test_sse_job_of_other_project_404(api_client, tmp_path):
    pid_a, jid = _make_project_job(api_client, tmp_path, progress=100.0)
    pid_b = api_client.post("/projects", json={"title": "Other"}).json()["id"]
    r = api_client.get(f"/projects/{pid_b}/jobs/{jid}/events")
    assert r.status_code == 404
