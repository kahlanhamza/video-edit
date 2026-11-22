"""Approve & Render gate — human-gate endpoint + dirty-segment re-render law (M3).

Pins:
- approve only opens at `awaiting_review` (409 otherwise) and needs a timeline
  (422) + an ingested asset (404) — the gate is a STATE, not a button,
- approve flips the project to `approved`, queues a render-stage JobRow and
  hands `directoros.advance_job` to the fleet with the ORIGINAL source path —
  the idempotent pipeline + content-addressed segment cache make "only dirty
  segments re-render" true by construction (cache hits are free),
- broker-down degradation mirrors ingest: 202 with enqueued=False, job stays
  queued for the fleet to pick up.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/approve.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture()
def reviewable_project(api_client, tmp_path):
    """Project at `awaiting_review` with an active timeline + raw asset row."""
    import director_os.api.main as api_main
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.core.db import MediaAssetRow, ProjectRow
    from director_os.core.models import ProjectStatus
    from director_os.otio_engine.contract import RawClipMeta
    from director_os.otio_engine.timeline import (
        ClipSpec,
        GapSpec,
        TimelineSpec,
        TrackSpec,
        build_timeline,
    )

    source = tmp_path / "raw_bodycam.mp4"
    source.write_bytes(b"\x00\x01fake-mp4")
    pid = api_client.post("/projects", json={"title": "Approve case"}).json()["id"]
    timeline = build_timeline(
        TimelineSpec(
            name="approve_case",
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
                            dur_s=5.0,
                            meta=RawClipMeta(asset_id="a_approve"),
                        )
                    ],
                ),
                TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=5.0)]),
            ],
        )
    )
    store = CheckpointStore(api_main._session_factory)
    version = store.checkpoint(timeline, pid, note="approve seed", created_by="test")
    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        p = s.get(ProjectRow, pid)
        p.active_otio_version_id = version
        p.status = ProjectStatus.awaiting_review.value
        s.add(
            MediaAssetRow(
                id="a_approve",
                project_id=pid,
                source_url=str(source),
                role="raw_bwc",
                duration_s=5.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
        )
        s.commit()
    return pid, version, source


def _patch_fleet(monkeypatch) -> list[list]:
    """Capture celery send_task args — hermetic, no broker."""
    from director_os.worker.celery_app import celery_app

    sent: list[list] = []

    def _fake_send(name: str, args: list | None = None, **kw) -> list:
        sent.append(args or [])
        return sent

    monkeypatch.setattr(celery_app, "send_task", _fake_send)
    return sent


def test_approve_happy_path_flips_status_and_queues_render(
    api_client, reviewable_project, monkeypatch
):
    from director_os.core.db import JobRow, ProjectRow

    pid, version, source = reviewable_project
    sent = _patch_fleet(monkeypatch)

    r = api_client.post(f"/projects/{pid}/approve")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "approved"
    assert body["version"] == version
    assert body["enqueued"] is True
    assert "cache hits are free" in body["detail"]

    # the fleet received exactly the advance_job payload with the ORIGINAL source
    assert len(sent) == 1
    job_id, queued_pid, queued_source, db_url = sent[0]
    assert queued_pid == pid
    assert queued_source == str(source)

    # durable state: project approved + a render-stage job the browser can watch
    import director_os.api.main as api_main

    with api_main._session_factory() as s:  # type: ignore[union-attr]
        assert s.get(ProjectRow, pid).status == "approved"
        job = s.get(JobRow, job_id)
        assert job.current_stage == "render"
        assert job.progress_pct == 0.0


def test_approve_degrades_cleanly_when_broker_down(api_client, reviewable_project, monkeypatch):
    """Broker down no longer strands the approval: the gateway dispatches the
    re-render inline (same advance_job entrypoint the fleet runs), so the
    human decision is durable AND the render actually happens without a CLI."""
    from director_os.worker.celery_app import celery_app

    def boom(name, args=None, **kw):
        raise ConnectionError("redis down")

    monkeypatch.setattr(celery_app, "send_task", boom)
    pid, _, _ = reviewable_project
    r = api_client.post(f"/projects/{pid}/approve")
    assert r.status_code == 202
    body = r.json()
    assert body["enqueued"] is True  # dispatched — inline, not stranded
    assert "inline" in body["detail"]
    assert body["ok"] is True  # the human decision is durable even without a fleet


def test_approve_409_before_qa_pass(api_client, reviewable_project):
    import director_os.api.main as api_main
    from director_os.core.db import ProjectRow

    pid, _, _ = reviewable_project
    with api_main._session_factory() as s:  # type: ignore[union-attr]
        s.get(ProjectRow, pid).status = "planned"
        s.commit()
    r = api_client.post(f"/projects/{pid}/approve")
    assert r.status_code == 409
    assert "awaiting_review" in r.json()["detail"]


def test_approve_422_without_timeline(api_client):
    pid = api_client.post("/projects", json={"title": "No timeline"}).json()["id"]
    r = api_client.post(f"/projects/{pid}/approve")
    assert r.status_code == 422


def test_approve_404_unknown_project(api_client):
    assert api_client.post("/projects/ghost/approve").status_code == 404
