"""Agent tool-dispatch endpoint — checkpoint-first HTTP dispatch (M2)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_dispatch.db")
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture()
def project_with_timeline(api_client, tmp_path):
    """A project whose active timeline has one 5s raw clip on the master track."""
    import director_os.api.main as api_main
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.core.db import ProjectRow
    from director_os.otio_engine.contract import RawClipMeta
    from director_os.otio_engine.timeline import (
        ClipSpec,
        GapSpec,
        TimelineSpec,
        TrackSpec,
        build_timeline,
    )

    pid = api_client.post("/projects", json={"title": "Dispatch case"}).json()["id"]
    timeline = build_timeline(
        TimelineSpec(
            name="dispatch_case",
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
                            meta=RawClipMeta(asset_id="asset_x"),
                        )
                    ],
                ),
                TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=5.0)]),
            ],
        )
    )
    store = CheckpointStore(api_main._session_factory)
    version = store.checkpoint(timeline, pid, note="dispatch test seed", created_by="test")
    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        p = s.get(ProjectRow, pid)
        p.active_otio_version_id = version
        s.commit()
    return pid, version


def _dispatch(api_client, pid: str, tool: str, args: dict, note: str = "") -> dict:
    r = api_client.post(
        f"/projects/{pid}/agent/dispatch",
        json={"tool": tool, "args": args, "checkpoint_note": note},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_dispatch_trim_advances_pointer(api_client, project_with_timeline):
    pid, v0 = project_with_timeline
    body = _dispatch(
        api_client,
        pid,
        "timeline.trim_clip",
        {
            "item_index": 0,
            "new_start_s": 1.0,
            "new_dur_s": 3.0,
        },
        note="tighten the opening",
    )
    assert body["ok"] is True
    assert body["version"] == v0 + 3  # plan start + before-step + plan complete
    assert body["version_before"] == v0 + 2
    assert "trimmed" in body["detail"]

    # pointer moved; a second dispatch builds on the trimmed state
    body2 = _dispatch(api_client, pid, "timeline.split_clip", {"item_index": 0, "at_local_s": 1.5})
    assert body2["ok"] is True
    assert body2["version"] > body["version"]

    import director_os.api.main as api_main
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.core.db import ProjectRow

    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        assert s.get(ProjectRow, pid).active_otio_version_id == body2["version"]
    store = CheckpointStore(api_main._session_factory)
    notes = [v["checkpoint_note"] for v in store.list_versions(pid)]
    assert any("tighten the opening" in n for n in notes)


def test_dispatch_semantic_failure_rolls_back(api_client, project_with_timeline):
    pid, v0 = project_with_timeline
    body = _dispatch(
        api_client,
        pid,
        "timeline.trim_clip",
        {
            "item_index": 7,
            "new_start_s": 1.0,
            "new_dur_s": 3.0,  # index out of range
        },
    )
    assert body["ok"] is False
    assert body["rolled_back_to"] is not None  # durable rollback happened
    assert "out of range" in body["detail"]

    import director_os.api.main as api_main
    from director_os.core.db import ProjectRow

    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        assert s.get(ProjectRow, pid).active_otio_version_id == body["rolled_back_to"]


def test_dispatch_unknown_tool_rejected_and_rolled_back(api_client, project_with_timeline):
    pid, _ = project_with_timeline
    body = _dispatch(api_client, pid, "timeline.nuke_everything", {})
    assert body["ok"] is False
    assert body["rolled_back_to"] is not None


def test_dispatch_unknown_project_404(api_client):
    r = api_client.post(
        "/projects/nope/agent/dispatch", json={"tool": "timeline.trim_clip", "args": {}}
    )
    assert r.status_code == 404


def test_dispatch_without_timeline_422(api_client):
    pid = api_client.post("/projects", json={"title": "No timeline"}).json()["id"]
    r = api_client.post(
        f"/projects/{pid}/agent/dispatch", json={"tool": "timeline.trim_clip", "args": {}}
    )
    assert r.status_code == 422
    assert "no timeline" in r.json()["detail"]


def test_dispatch_logs_tool_row(api_client, project_with_timeline):
    pid, _ = project_with_timeline
    _dispatch(
        api_client,
        pid,
        "timeline.trim_clip",
        {
            "item_index": 0,
            "new_start_s": 0.5,
            "new_dur_s": 4.0,
        },
    )
    import director_os.api.main as api_main
    from director_os.agent_runtime.checkpoint import CheckpointStore

    logs = CheckpointStore(api_main._session_factory).recent_logs(pid)
    assert any(entry["tool"] == "timeline.trim_clip" for entry in logs)
    assert any(
        entry["tool"] == "llm.propose" or entry["tool"] == "timeline.trim_clip" for entry in logs
    )
