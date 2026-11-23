"""Selection API — promptable tracking endpoint + single-op detail.

- POST /assets/{id}/selection: box prompt -> tracked path -> tracked blur op
  (params.tracked) + dual-written union region (shipped render path stays
  covered — privacy never regresses)
- GET /ops/{op_id}: one op in full (the Studio reads a single tracked path)
- span cap + invalid prompts are 422; unknown backend is 409 (loud)
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/sel_api.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture()
def tracked_asset(api_client, moving_box_video):
    """Project + asset row pointing at the synthetic moving-box clip."""
    import director_os.api.main as api_main
    from director_os.core.db import MediaAssetRow

    pid = api_client.post("/projects", json={"title": "sel case"}).json()["id"]
    with api_main.session() as s:
        s.add(
            MediaAssetRow(
                id="a1",
                project_id=pid,
                source_url=str(moving_box_video),
                duration_s=4.0,
                fps=30.0,
                width=320,
                height=240,
            )
        )
        s.commit()
    return pid


def _box_body(anchor: int = 30, frame_start: int = 0, frame_end: int = 89) -> dict:
    return {
        "kind": "box",
        "x": 40 + 2 * anchor,
        "y": 60 + anchor,
        "w": 28,
        "h": 28,
        "anchor_frame": anchor,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


def test_selection_endpoint_tracks_and_creates_op(api_client, tracked_asset):
    r = api_client.post("/assets/a1/selection", json=_box_body())
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["backend"] == "ncc-cpu"
    assert out["n_hits"] >= 70
    assert out["op_id"]
    assert "-> blur op" in out["detail"]
    # the op carries the tracked path
    op = api_client.get(f"/ops/{out['op_id']}").json()
    assert op["kind"] == "blur"
    assert op["params"]["mode"] == "tracked"
    assert op["params"]["n_tracked"] == out["n_hits"]
    # and the shipped render path got its union region (privacy dual-write)
    nle = api_client.get(f"/projects/{tracked_asset}/timeline/nle").json()
    regions = [e for e in nle.get("effects", []) if e.get("kind") == "blur"]
    assert regions, "no blur effects in the NLE snapshot"
    op_effect = next(e for e in regions if e.get("op_id") == out["op_id"])
    assert op_effect["origin"] in ("human", "agent")


def test_selection_endpoint_span_cap(api_client, tracked_asset):
    body = _box_body()
    body["frame_end"] = 3599  # 120 s at 30 fps — way over the interactive cap
    r = api_client.post("/assets/a1/selection", json=body)
    assert r.status_code == 422
    assert "interactive cap" in r.json()["detail"]


def test_selection_endpoint_rejects_bad_prompts(api_client, tracked_asset):
    # box without size
    body = _box_body()
    body["w"] = 0
    r = api_client.post("/assets/a1/selection", json=body)
    assert r.status_code == 422
    # anchor beyond the tape
    body = _box_body(anchor=200)
    r = api_client.post("/assets/a1/selection", json=body)
    assert r.status_code == 422
    # unknown asset
    r = api_client.post("/assets/nope/selection", json=_box_body())
    assert r.status_code == 404


def test_selection_endpoint_no_op_mode(api_client, tracked_asset):
    body = _box_body()
    body["create_blur_op"] = False
    r = api_client.post("/assets/a1/selection", json=body)
    assert r.status_code == 201
    assert r.json()["op_id"] is None


def test_selection_endpoint_unknown_backend_is_loud(api_client, tracked_asset):
    body = _box_body()
    body["backend"] = "sam2"
    r = api_client.post("/assets/a1/selection", json=body)
    assert r.status_code in (409, 422)
    assert "sam2" in r.json()["detail"]


def test_selection_endpoint_honest_refusal(api_client, tracked_asset, tmp_path):
    """Flat media: 422 with the texture-free refusal, never a fabricated path."""
    import subprocess

    import director_os.api.main as api_main
    from director_os.core.db import MediaAssetRow

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:d=1:r=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(tmp_path / "flat.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    with api_main.session() as s:
        s.add(
            MediaAssetRow(
                id="a2",
                project_id=tracked_asset[0],
                source_url=str(tmp_path / "flat.mp4"),
                duration_s=1.0,
                fps=30.0,
                width=320,
                height=240,
            )
        )
        s.commit()
    r = api_client.post("/assets/a2/selection", json=_box_body(anchor=5, frame_end=20))
    assert r.status_code == 422
    assert "texture-free" in r.json()["detail"]


def test_get_op_404(api_client):
    r = api_client.get("/ops/does-not-exist")
    assert r.status_code == 404


def test_selection_backends_endpoint(api_client):
    r = api_client.get("/selection/backends")
    assert r.status_code == 200
    out = r.json()
    assert out["ncc-cpu"] is True
    assert "sam2" in out


def test_selection_hits_are_json_round_trippable(api_client, tracked_asset):
    """Every hit survives the JSON boundary with int coords + float t."""
    r = api_client.post("/assets/a1/selection", json=_box_body())
    hits = r.json()["hits"]
    assert hits
    for h in hits:
        assert isinstance(h["x"], int)
        assert isinstance(h["y"], int)
        assert isinstance(h["t_s"], float)
        assert isinstance(h["interpolated"], bool)
        assert json.dumps(h)  # serializable
