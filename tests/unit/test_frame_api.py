"""Frame endpoint + run-resume semantics (owner laws 2026-11-23).

- GET /assets/{id}/frame serves a frame-accurate JPEG off the 480p proxy
  (src=full decodes the original), LRU-cached per asset
- POST /projects/{id}/run on a FAILED job reuses the job AND clears the
  terminal error before dispatch — the SSE stream can never insta-close on
  a stale error, so Run-again is always visible
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/frame_api.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture()
def asset_with_proxy(api_client, tmp_path, fixture_video):
    """Project + asset row whose proxy bundle was really built (ingest path)."""
    from director_os.core.db import MediaAssetRow
    from director_os.core.proxy import build_proxies

    pid = api_client.post("/projects", json={"title": "frame case"}).json()["id"]

    import director_os.api.main as api_main

    bundle = build_proxies(fixture_video, tmp_path / "assets" / "a1")
    with api_main.session() as s:
        s.add(
            MediaAssetRow(
                id="a1",
                project_id=pid,
                source_url=str(fixture_video),
                duration_s=12.0,
                fps=30.0,
                width=320,
                height=240,
                r2_proxy_key=str(bundle.proxy_path),
                r2_audio_key=str(bundle.audio_path),
            )
        )
        s.commit()
    return pid, "a1"


def test_frame_endpoint_serves_jpeg_from_proxy(api_client, asset_with_proxy, tmp_path):
    pid, aid = asset_with_proxy
    r = api_client.get(f"/assets/{aid}/frame", params={"t": 1.5, "w": 320})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    assert r.content[:3] == b"\xff\xd8\xff"
    assert len(r.content) > 1000

    # the frame landed in the per-asset LRU cache dir
    frames_dir = tmp_path / "output" / "pipeline" / pid / "assets" / aid / "frames"
    assert any(frames_dir.glob("*.jpg"))


def test_frame_endpoint_full_res_and_errors(api_client, asset_with_proxy):
    _pid, aid = asset_with_proxy
    full = api_client.get(f"/assets/{aid}/frame", params={"t": 2.0, "src": "full"})
    assert full.status_code == 200
    assert full.content[:3] == b"\xff\xd8\xff"

    assert api_client.get(f"/assets/{aid}/frame", params={"t": -1}).status_code == 422
    assert api_client.get(f"/assets/{aid}/frame", params={"w": 10}).status_code == 422
    assert api_client.get("/assets/nope/frame").status_code == 404


def test_run_again_clears_the_failed_job_error(api_client, monkeypatch, tmp_path):
    """Run on a failed job must reuse it, clear the error, and dispatch."""
    import director_os.api.main as api_main
    from director_os.core.db import JobRow, MediaAssetRow

    pid = api_client.post("/projects", json={"title": "resume case"}).json()["id"]
    src = tmp_path / "bodycam.mp4"
    src.write_bytes(b"fake-media")  # dispatch is faked; bytes never read

    with api_main.session() as s:
        s.add(
            MediaAssetRow(
                id="a1",
                project_id=pid,
                source_url=str(src),
                duration_s=12.0,
                fps=30.0,
                width=320,
                height=240,
            )
        )
        s.add(
            JobRow(
                id="job-failed",
                project_id=pid,
                current_stage="perception",
                progress_pct=22.0,
                error_message="no transcribe callable provided (Gladia not configured?)",
            )
        )
        s.commit()

    dispatched: list[tuple] = []
    monkeypatch.setattr(
        api_main, "_dispatch_advance", lambda *a, **kw: dispatched.append(a) or (True, "test")
    )

    r = api_client.post(f"/projects/{pid}/run")
    assert r.status_code == 202
    body = r.json()
    assert body["reused"] is True
    assert body["job_id"] == "job-failed"
    assert dispatched, "the resume path must dispatch the advance task"

    with api_main.session() as s:
        job = s.get(JobRow, "job-failed")
        assert job.error_message is None, "terminal error cleared before dispatch"
        events = [e for e in s.query(api_main.JobEventRow).all() if e.job_id == "job-failed"]
    assert any(e.payload.get("resumed") for e in events), "a resumed event was emitted"
