"""Op log API — CRUD + fold + SSE op stream + legacy import (op log v1)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/ops_api.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


def _project(api_client, *, with_asset: bool = False) -> str:
    pid = api_client.post("/projects", json={"title": "Op log case"}).json()["id"]
    if with_asset:
        import director_os.api.main as api_main
        from director_os.core.db import MediaAssetRow

        s = api_main.session()
        s.add(
            MediaAssetRow(
                project_id=pid,
                duration_s=100.0,
                fps=30.0,
                width=1920,
                height=1080,
            )
        )
        s.commit()
        s.close()
    return pid


# ---------------------------------------------------------------- CRUD laws
def test_create_and_list_ops(api_client):
    pid = _project(api_client)
    r = api_client.post(
        f"/projects/{pid}/ops",
        json={"kind": "blur", "params": {"x": 10, "y": 20, "w": 300, "h": 120, "sigma": 24}},
    )
    assert r.status_code == 201
    op = r.json()
    assert op["rev"] == 1 and op["kind"] == "blur" and op["origin"] == "human"
    assert op["revoked"] is False

    state = api_client.get(f"/projects/{pid}/ops").json()
    assert state["active_ops"] == 1 and state["rev"] == 1
    assert len(state["fold"]["effects"]) == 1
    assert state["fold"]["effects"][0]["op_id"] == op["id"]


def test_unknown_kind_rejected(api_client):
    pid = _project(api_client)
    r = api_client.post(f"/projects/{pid}/ops", json={"kind": "teleport"})
    assert r.status_code == 422
    assert "known" in r.json()["detail"]


def test_agent_op_requires_evidence(api_client):
    pid = _project(api_client)
    r = api_client.post(
        f"/projects/{pid}/ops",
        json={"kind": "blur", "origin": "agent", "params": {"x": 1}},
    )
    assert r.status_code == 422
    assert "evidence" in r.json()["detail"]
    ok = api_client.post(
        f"/projects/{pid}/ops",
        json={
            "kind": "blur",
            "origin": "agent",
            "evidence": {"frames": [4201], "note": "face leak visible"},
            "params": {"x": 1, "y": 2, "w": 3, "h": 4},
        },
    )
    assert ok.status_code == 201
    assert ok.json()["origin"] == "agent"


def test_frame_window_validated_against_tape(api_client):
    pid = _project(api_client, with_asset=True)
    bad = api_client.post(
        f"/projects/{pid}/ops", json={"kind": "zoom", "frame_start": 5000, "frame_end": 5100}
    )
    assert bad.status_code == 422 and "beyond tape" in bad.json()["detail"]
    inverted = api_client.post(
        f"/projects/{pid}/ops", json={"kind": "zoom", "frame_start": 100, "frame_end": 50}
    )
    assert inverted.status_code == 422
    ok = api_client.post(
        f"/projects/{pid}/ops", json={"kind": "zoom", "frame_start": 0, "frame_end": 3000}
    )
    assert ok.status_code == 201


def test_404_for_missing_project(api_client):
    assert api_client.get("/projects/ghost/ops").status_code == 404
    assert api_client.post("/projects/ghost/ops", json={"kind": "note"}).status_code == 404


# ------------------------------------------------- dual-write + legacy import
def test_blur_op_dual_writes_legacy_region(api_client):
    pid = _project(api_client, with_asset=True)
    op = api_client.post(
        f"/projects/{pid}/ops",
        json={"kind": "blur", "params": {"x": 4, "y": 4, "w": 40, "h": 40, "sigma": 18}},
    ).json()
    regions = api_client.get(f"/projects/{pid}/redactions").json()
    assert len(regions) == 1 and regions[0]["id"] == op["id"], "render path keeps working"
    assert regions[0]["created_by"] == "human_edit"


def test_import_legacy_regions_idempotent(api_client):
    pid = _project(api_client, with_asset=True)
    region = api_client.post(
        f"/projects/{pid}/redactions",
        json={"x": 10, "y": 10, "w": 100, "h": 80, "sigma": 24, "label": "plate"},
    ).json()
    first = api_client.post(f"/projects/{pid}/ops/import-legacy").json()
    assert first == {"imported": 1, "skipped": 0, "total_regions": 1}
    again = api_client.post(f"/projects/{pid}/ops/import-legacy").json()
    assert again == {"imported": 0, "skipped": 1, "total_regions": 1}

    state = api_client.get(f"/projects/{pid}/ops").json()
    assert state["active_ops"] == 1
    effects = state["fold"]["effects"]
    assert len(effects) == 1, "import must never double-render the region"
    assert effects[0]["x"] == region["x"] and effects[0]["op_id"]
    assert effects[0]["params" if "params" in effects[0] else "label"]


def test_revoke_drops_region_and_fold(api_client):
    pid = _project(api_client, with_asset=True)
    op = api_client.post(
        f"/projects/{pid}/ops",
        json={"kind": "blur", "params": {"x": 4, "y": 4, "w": 40, "h": 40}},
    ).json()
    r = api_client.post(f"/projects/{pid}/ops/{op['id']}/revoke", json={"note": "wrong box"})
    assert r.status_code == 200 and r.json()["revoked"] is True
    assert api_client.get(f"/projects/{pid}/redactions").json() == []

    state = api_client.get(f"/projects/{pid}/ops").json()
    assert state["active_ops"] == 0 and state["revoked_ops"] == 1
    assert state["fold"]["effects"] == []

    again = api_client.post(f"/projects/{pid}/ops/{op['id']}/revoke")
    assert again.status_code == 409
    assert api_client.post(f"/projects/{pid}/ops/nope/revoke").status_code == 404


# ---------------------------------------------------------------- NLE fold
def test_nle_snapshot_renders_effects_from_fold(api_client, tmp_path):
    pid = _project(api_client, with_asset=True)
    agent_op = api_client.post(
        f"/projects/{pid}/ops",
        json={
            "kind": "blur",
            "origin": "agent",
            "evidence": {"frames": [120, 121]},
            "params": {"x": 1, "y": 2, "w": 3, "h": 4, "sigma": 30},
        },
    ).json()
    api_client.post(
        f"/projects/{pid}/redactions",
        json={"x": 50, "y": 50, "w": 60, "h": 60, "sigma": 24, "label": "legacy"},
    )
    nle = api_client.get(f"/projects/{pid}/timeline/nle").json()
    assert nle["ops_rev"] >= 1
    assert nle["ops_counts"]["active"] == 1
    assert len(nle["effects"]) == 2
    op_fx = next(e for e in nle["effects"] if e["op_id"] == agent_op["id"])
    assert op_fx["origin"] == "agent" and op_fx["sigma"] == 30
    legacy_fx = next(e for e in nle["effects"] if e["op_id"] is None)
    assert legacy_fx["label"] == "legacy" and legacy_fx["origin"] == "human"


def test_caption_edit_records_op(api_client, tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from director_os.subtitles.ass_builder import CaptionBeat
    from director_os.subtitles.store import save_cues

    pid = _project(api_client)
    save_cues(
        get_settings().output_dir / "pipeline" / pid,
        [CaptionBeat(index=0, text="raw text", speaker="", start_ms=0, end_ms=900, words=[])],
    )
    r = api_client.patch(
        f"/projects/{pid}/subtitles/0", json={"text": "proofread text", "speaker": "OFFICER"}
    )
    assert r.status_code == 200
    state = api_client.get(f"/projects/{pid}/ops").json()
    assert state["active_ops"] == 1
    assert state["fold"]["caption_overrides"]["0"]["text"] == "proofread text"
    assert state["fold"]["caption_overrides"]["0"]["speaker"] == "OFFICER"
    del api_main  # imported for session side effects


# ---------------------------------------------------------------- SSE stream
def test_ops_stream_pushes_create_and_revoke(api_client):
    pid = _project(api_client)
    op = api_client.post(
        f"/projects/{pid}/ops",
        json={"kind": "blur", "params": {"x": 1, "y": 2, "w": 3, "h": 4}},
    ).json()
    api_client.post(f"/projects/{pid}/ops/{op['id']}/revoke")

    with api_client.stream("GET", f"/projects/{pid}/ops/stream", params={"max_seconds": 1.0}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        text = "".join(chunk for chunk in r.iter_text())
    assert "event: hello" in text
    assert "event: op" in text
    assert '"kind":"blur"' in text.replace(" ", "")
    assert '"kind":"revoke"' in text.replace(" ", "")
    assert "event: done" in text
    compact = text.replace(" ", "")
    assert '"revokes_op_id":"' + op["id"] + '"' in compact
