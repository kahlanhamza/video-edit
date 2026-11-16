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


def _seed_timeline(api_client, pid: str) -> int:
    import director_os.api.main as api_main
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.otio_engine.contract import RawClipMeta
    from director_os.otio_engine.timeline import ClipSpec, TimelineSpec, TrackSpec, build_timeline

    timeline = build_timeline(
        TimelineSpec(
            name="studio_case",
            fps=30,
            width=1920,
            height=1080,
            tracks=[
                TrackSpec(
                    name="V1_master",
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=4.0,
                            meta=RawClipMeta(asset_id="a"),
                        )
                    ],
                )
            ],
        )
    )
    store = CheckpointStore(api_main._session_factory)
    return store.checkpoint(timeline, pid, note="studio seed", created_by="test")


def test_list_projects_and_jobs(api_client, tmp_path):
    ids = [api_client.post("/projects", json={"title": f"P{i}"}).json()["id"] for i in range(2)]
    r = api_client.get("/projects")
    assert r.status_code == 200
    assert {p["id"] for p in r.json()} == set(ids)

    pid = ids[0]
    import director_os.api.main as api_main
    from director_os.core.db import JobRow

    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        s.add(JobRow(project_id=pid, current_stage="qa", progress_pct=90.0))
        s.commit()
    r = api_client.get(f"/projects/{pid}/jobs")
    assert r.status_code == 200
    jobs = r.json()
    assert len(jobs) == 1 and jobs[0]["stage"] == "qa" and jobs[0]["progress_pct"] == 90.0
    assert api_client.get("/projects/nope/jobs").status_code == 404


def test_timeline_state_endpoint(api_client, tmp_path):
    pid = api_client.post("/projects", json={"title": "Timeline case"}).json()["id"]
    v = _seed_timeline(api_client, pid)

    import director_os.api.main as api_main
    from director_os.core.db import ProjectRow

    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        s.get(ProjectRow, pid).active_otio_version_id = v
        s.commit()

    r = api_client.get(f"/projects/{pid}/timeline")
    assert r.status_code == 200
    body = r.json()
    assert body["active_version"] == v
    assert len(body["versions"]) == 1
    assert body["versions"][0]["checkpoint_note"] == "studio seed"
    assert api_client.get("/projects/nope/timeline").status_code == 404
