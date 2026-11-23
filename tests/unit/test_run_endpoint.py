"""The Run button contract — regressions for the two shipped bugs.

CRIT-2026-11-23 findings locked here:
- CR-A ("Creating..." forever): POST /run with a video_url used to download
  the source INSIDE the request — the Studio sat on "Creating..." for the
  length of a 469 MB bodycam fetch with zero feedback. Now the job registers
  instantly, the dispatch carries the URL, and the WORKER downloads.
- CR-B1 ("Run does nothing"): Run during an active ingest used to 422
  ("nothing to run yet") because the media_assets row only exists after
  probe+proxy — the button lied while the pipeline was actually running.
- CR-guard (double dispatch): two Run clicks queued the same job twice on
  the fleet path (the old in-flight guard covered only the inline thread) —
  two workers raced one pipeline (duplicate ASR spend, proxy build races).
- CR-ingest: /ingest used to download in-request too; now it refuses to
  start a second pipeline on a running case (one case = one run, CR-006).
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_run.db")
    get_settings.cache_clear()
    api_main._engine = None  # force re-bind to the tmp database
    api_main._in_flight.clear()  # claims are process-global — isolate per test
    with TestClient(api_main.app) as client:
        yield client
    api_main._in_flight.clear()
    get_settings.cache_clear()


@pytest.fixture()
def client(api_client):
    """The whole-file alias: tests below read `client`, the app fixture is
    `api_client` — without this bridge the suite errors at fixture setup."""
    return api_client


@pytest.fixture()
def dispatch_recorder(api_client, monkeypatch):
    """Capture celery send_task calls — no broker, no inline pipeline thread."""
    calls: list[tuple[str, list]] = []

    def fake_send_task(task, args=None, **kwargs):
        calls.append((task, list(args or [])))
        return None

    monkeypatch.setattr("director_os.worker.celery_app.celery_app.send_task", fake_send_task)
    return calls


def _create_case(client, title: str = "Run contract case") -> str:
    r = client.post("/projects", json={"title": title})
    assert r.status_code == 201
    return r.json()["id"]


def test_run_with_url_registers_job_and_dispatches_url(client, dispatch_recorder):
    """CR-A: the URL rides the dispatch; no in-request download happens."""
    pid = _create_case(client)
    r = client.post(f"/projects/{pid}/run", json={"video_url": "https://example.com/bodycam.mp4"})
    assert r.status_code == 202
    body = r.json()
    assert body["reused"] is False
    assert body["stage"] == "ingest"
    assert len(dispatch_recorder) == 1

    # the dispatch event is the resume contract: the URL is on record
    import director_os.api.main as api_main
    from director_os.core.db import JobEventRow

    s = api_main.session()
    try:
        row = (
            s.query(JobEventRow)
            .filter(JobEventRow.job_id == body["job_id"], JobEventRow.kind == "dispatch")
            .one_or_none()
        )
        assert row is not None
        assert row.payload["source"] == "https://example.com/bodycam.mp4"
    finally:
        s.close()


def test_run_with_url_dispatch_args_shape(client, dispatch_recorder):
    pid = _create_case(client)
    r = client.post(f"/projects/{pid}/run", json={"video_url": "https://example.com/v.mp4"})
    assert r.status_code == 202
    ((task, args),) = dispatch_recorder
    assert task == "directoros.advance_job"
    assert args[0] == r.json()["job_id"]
    assert args[1] == pid
    assert args[2] == "https://example.com/v.mp4"  # the URL, not a local path
    assert isinstance(args[3], str) and "sqlite" in args[3]


def test_run_with_url_while_job_running_returns_running_job(client, dispatch_recorder):
    """One case = one run: a second Run click cannot start a parallel pipeline."""
    pid = _create_case(client)
    first = client.post(
        f"/projects/{pid}/run", json={"video_url": "https://example.com/a.mp4"}
    ).json()
    r = client.post(f"/projects/{pid}/run", json={"video_url": "https://example.com/b.mp4"})
    assert r.status_code == 202
    body = r.json()
    assert body["reused"] is True
    assert body["job_id"] == first["job_id"]
    assert "already running" in body["detail"]
    # still exactly ONE job and ONE dispatch
    assert len(dispatch_recorder) == 1


def test_run_during_ingest_returns_running_job_not_422(client, dispatch_recorder):
    """CR-B1, the exact reported bug: Run while ingest is in flight."""
    pid = _create_case(client)
    first = client.post(
        f"/projects/{pid}/run", json={"video_url": "https://example.com/big.mp4"}
    ).json()
    # the asset row does NOT exist yet (worker still downloading) — Run must
    # return the live job, not "nothing to run yet"
    r = client.post(f"/projects/{pid}/run", json={"video_url": None, "custom_headers": None})
    assert r.status_code == 202
    body = r.json()
    assert body["reused"] is True
    assert body["job_id"] == first["job_id"]
    assert body["dispatched"] is True
    assert "already running" in body["detail"]


def test_run_resume_after_gateway_restart_uses_dispatch_url(client, dispatch_recorder):
    """A crashed ingest (no asset row) is resumable from the dispatch event."""
    import director_os.api.main as api_main

    pid = _create_case(client)
    first = client.post(
        f"/projects/{pid}/run", json={"video_url": "https://example.com/dead.mp4"}
    ).json()
    api_main._release_job(first["job_id"])  # simulate gateway restart / TTL

    r = client.post(f"/projects/{pid}/run", json={})
    assert r.status_code == 202
    body = r.json()
    assert body["reused"] is True
    assert body["job_id"] == first["job_id"]
    assert len(dispatch_recorder) == 2  # re-dispatched
    ((task, args),) = dispatch_recorder[1:2]
    assert args[2] == "https://example.com/dead.mp4"


def test_run_fresh_project_without_media_is_422(client, dispatch_recorder):
    pid = _create_case(client)
    r = client.post(f"/projects/{pid}/run", json={})
    assert r.status_code == 422
    assert "nothing to run yet" in r.json()["detail"]
    assert dispatch_recorder == []


def test_ingest_conflicts_while_pipeline_running(client, dispatch_recorder):
    pid = _create_case(client)
    client.post(f"/projects/{pid}/run", json={"video_url": "https://example.com/a.mp4"})
    r = client.post(f"/projects/{pid}/ingest", json={"url": "https://example.com/b.mp4"})
    assert r.status_code == 409
    assert "already running" in r.json()["detail"]
    assert len(dispatch_recorder) == 1


def test_ingest_registers_job_and_dispatches_url(client, dispatch_recorder):
    """CR-ingest: /ingest no longer downloads in-request either."""
    pid = _create_case(client)
    r = client.post(f"/projects/{pid}/ingest", json={"url": "https://example.com/c.mp4"})
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"]
    assert body["enqueued"] is True
    ((task, args),) = dispatch_recorder
    assert task == "directoros.advance_job"
    assert args[2] == "https://example.com/c.mp4"


def test_stop_releases_claim_and_run_redispatches(client, dispatch_recorder):
    """Stop -> Run resume law: the parked job must be resumable immediately."""
    pid = _create_case(client)
    client.post(f"/projects/{pid}/run", json={"video_url": "https://example.com/s.mp4"})
    r = client.post(f"/projects/{pid}/stop")
    assert r.status_code == 202
    r2 = client.post(f"/projects/{pid}/run", json={})
    assert r2.status_code == 202
    assert len(dispatch_recorder) == 2  # the claim did not eat the resume


def test_dispatch_guard_blocks_double_send(api_client, dispatch_recorder):
    """CR-guard: the same job id can never be queued twice while claimed."""
    import director_os.api.main as api_main

    ok1, detail1 = api_main._dispatch_advance("job-guard-1", "proj", "src")
    ok2, detail2 = api_main._dispatch_advance("job-guard-1", "proj", "src")
    assert ok1 and ok2  # both succeed from the caller's view
    assert dispatch_recorder and len(dispatch_recorder) == 1
    assert "already running" in detail2


def test_failed_download_is_resumable_via_dispatch_event(client, dispatch_recorder):
    """A job that failed before the asset landed resumes from the event URL."""
    import director_os.api.main as api_main
    from director_os.core.db import JobRow

    pid = _create_case(client)
    first = client.post(
        f"/projects/{pid}/run", json={"video_url": "https://example.com/died.mp4"}
    ).json()
    # simulate the worker dying before probe: mark the job failed
    api_main._release_job(first["job_id"])
    s = api_main.session()
    try:
        job = s.get(JobRow, first["job_id"])
        job.error_message = "pipeline failed at ingest: download died"
        s.commit()
    finally:
        s.close()

    r = client.post(f"/projects/{pid}/run", json={})
    assert r.status_code == 202
    body = r.json()
    assert body["reused"] is True
    assert body["job_id"] == first["job_id"]
    ((task, args),) = dispatch_recorder[1:2]
    assert args[2] == "https://example.com/died.mp4"
