"""M3 paste-a-URL ingest — hermetic API tests (no network, no broker).

The 2026-11-23 dispatch law: the gateway NEVER downloads inside the request
(the old in-request fetch made the browser hang for the length of a 469 MB
bodycam — the "Creating…" bug). POST /ingest registers the job, writes the
dispatch event (the resume contract) and queues the fleet task; the WORKER
downloads with byte progress. The wire contract + job-row semantics are
pinned here; the run-endpoint resume law lives in test_run_endpoint.py.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None  # force re-bind to the tmp database
    api_main._in_flight.clear()
    with TestClient(api_main.app) as client:
        yield client
    api_main._in_flight.clear()
    get_settings.cache_clear()


@pytest.fixture()
def recorded_enqueue(monkeypatch):
    """Capture celery send_task calls — the worker owns the download now."""
    sent: list[dict] = []

    def fake_send_task(name, args=None, **kwargs):
        sent.append({"name": name, "args": args, "kwargs": kwargs})
        return None

    monkeypatch.setattr("director_os.worker.celery_app.celery_app.send_task", fake_send_task)
    return sent


def _make_project(api_client) -> str:
    return api_client.post("/projects", json={"title": "Ingest case"}).json()["id"]


def test_ingest_url_registers_job_dispatches_url_and_enqueues(api_client, recorded_enqueue):
    """The URL rides the dispatch; nothing blocks on the network here."""
    pid = _make_project(api_client)
    r = api_client.post(
        f"/projects/{pid}/ingest",
        json={
            "url": "https://foia.example/eustis-bodycam.mp4",
            "custom_headers": {"Referer": "https://portal.example/case"},
        },
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["enqueued"] is True
    assert body["source_path"].endswith("eustis-bodycam.mp4")
    assert len(recorded_enqueue) == 1
    assert recorded_enqueue[0]["name"] == "directoros.advance_job"
    job_id, proj, src, _db, _headers = recorded_enqueue[0]["args"]
    assert proj == pid
    assert src == "https://foia.example/eustis-bodycam.mp4"  # the URL, not a path

    # a queued job row exists and shows in /jobs (the SSE feed reads the same row)
    jobs = api_client.get(f"/projects/{pid}/jobs").json()
    assert [j["id"] for j in jobs] == [job_id]
    assert jobs[0]["stage"] == "ingest" and jobs[0]["progress_pct"] == 0.0

    # the dispatch event is on record (crashed-ingest resume contract)
    import director_os.api.main as api_main
    from director_os.core.db import JobEventRow

    s = api_main.session()
    try:
        row = (
            s.query(JobEventRow)
            .filter(JobEventRow.job_id == job_id, JobEventRow.kind == "dispatch")
            .one_or_none()
        )
        assert row is not None
        assert row.payload["source"] == "https://foia.example/eustis-bodycam.mp4"
        assert row.payload["custom_headers"] == {"Referer": "https://portal.example/case"}
    finally:
        s.close()


def test_ingest_url_rejects_non_http_scheme(api_client, recorded_enqueue):
    pid = _make_project(api_client)
    r = api_client.post(f"/projects/{pid}/ingest", json={"url": "ftp://x/y.mp4"})
    assert r.status_code == 422
    assert recorded_enqueue == []


def test_ingest_url_404_unknown_project(api_client, recorded_enqueue):
    r = api_client.post("/projects/nope/ingest", json={"url": "https://x.example/v.mp4"})
    assert r.status_code == 404
    assert recorded_enqueue == []
