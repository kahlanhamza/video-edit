"""Export gate — preview is overlay-only, Export is the only render.

- GET /projects/{id}/export/gate: never-exported -> dirty with exported_once
  false; after the render stamps baked_ops_rev -> clean; a later op event ->
  dirty again with pending_ops counted (revoked tombstones never count).
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/gate.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture()
def project_id(api_client):
    return api_client.post("/projects", json={"title": "gate case"}).json()["id"]


def test_gate_never_exported_is_dirty(api_client, project_id):
    r = api_client.get(f"/projects/{project_id}/export/gate")
    assert r.status_code == 200
    out = r.json()
    assert out["dirty"] is True
    assert out["exported_once"] is False
    assert out["ops_rev"] == 0
    assert out["baked_ops_rev"] is None


def test_gate_reports_dirty_after_ops(api_client, project_id):
    api_client.post(
        f"/projects/{project_id}/ops",
        json={"kind": "blur", "params": {"x": 1, "y": 1, "w": 4, "h": 4}, "evidence": {"f": 1}},
    )
    out = api_client.get(f"/projects/{project_id}/export/gate").json()
    assert out["ops_rev"] == 1
    assert out["dirty"] is True
    assert out["pending_ops"] == 0  # never exported: pending is undefined, not 1


def test_gate_clean_after_stamp_then_dirty_after_edit(api_client, project_id):
    import director_os.api.main as api_main
    from director_os.core.db import MediaAssetRow, PerceptionIndexRow

    api_client.post(
        f"/projects/{project_id}/ops",
        json={"kind": "blur", "params": {"x": 1, "y": 1, "w": 4, "h": 4}, "evidence": {"f": 1}},
    )
    with api_main.session() as s:
        s.add(
            MediaAssetRow(
                id="a1", project_id=project_id, duration_s=4.0, fps=30.0, width=320, height=240
            )
        )
        s.commit()
    # the RENDER stamps the cursor (server-side, at bake time)
    with api_main.session() as s:
        s.add(PerceptionIndexRow(asset_id="a1", redaction_tracks_json={"baked_ops_rev": 1}))
        s.commit()

    out = api_client.get(f"/projects/{project_id}/export/gate").json()
    assert out["exported_once"] is True
    assert out["dirty"] is False
    assert out["pending_ops"] == 0

    # a new op event -> dirty again, tombstones excluded from pending
    # (caption ops write through to the cue store — seed it first)
    from director_os.core.config import get_settings
    from director_os.subtitles.ass_builder import CaptionBeat
    from director_os.subtitles.store import save_cues

    save_cues(
        get_settings().output_dir / "pipeline" / project_id,
        [CaptionBeat(index=0, text="raw", speaker="", start_ms=0, end_ms=900, words=[])],
    )
    api_client.post(
        f"/projects/{project_id}/ops",
        json={"kind": "caption", "params": {"cue_index": 0, "text": "fixed"}},
    )
    out = api_client.get(f"/projects/{project_id}/export/gate").json()
    assert out["dirty"] is True
    assert out["pending_ops"] == 1


def test_gate_revoked_op_counts_as_pending_event(api_client, project_id):
    rec = api_client.post(
        f"/projects/{project_id}/ops",
        json={"kind": "blur", "params": {"x": 1, "y": 1, "w": 4, "h": 4}, "evidence": {"f": 1}},
    ).json()
    api_client.post(f"/projects/{project_id}/ops/{rec['id']}/revoke", json={"note": "undo"})
    out = api_client.get(f"/projects/{project_id}/export/gate").json()
    assert out["ops_rev"] >= 2  # the tombstone advanced the cursor
    assert out["dirty"] is True  # the render must re-bake (the op is gone)


def test_gate_404_unknown_project(api_client):
    assert api_client.get("/projects/nope/export/gate").status_code == 404
