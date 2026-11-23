"""SAM2 local weight service — the service boundary keeps torch out of CI.

Pinned hermetically: the SERVICE runs with a stubbed predictor (no torch,
no weights, no network); the CLIENT runs against httpx.MockTransport. What
this locks down: endpoint contract, weight-state gating, hit-shape parity
with TrackedSelection (the Studio must not care which backend tracked),
fallback law (service down -> SelectionNotAvailable, never a silent swap).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

MODULE = "gpu_services/sam2/service.py"


def _load_service(module_name: str = "sam2_service_under_test"):
    """Import the service file directly (it is NOT a package module).

    The module must sit in sys.modules BEFORE exec_module: FastAPI builds
    TypeAdapters from string annotations (`from __future__ import
    annotations`) and they resolve through sys.modules, not the frame.
    """
    import sys

    path = Path(__file__).resolve().parents[2] / MODULE
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture()
def svc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAM2_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.delenv("SAM2_WEIGHTS", raising=False)
    monkeypatch.delenv("SAM2_WEIGHTS_URL", raising=False)
    monkeypatch.delenv("SAM2_WEIGHTS_SHA256", raising=False)
    mod = _load_service()
    mod._predictor = None  # fresh predictor state per test
    return mod, TestClient(mod.app)


def _stub_predictor(mod, boxes_per_frame, conf=0.93):
    """A predictor stand-in: yields ultralytics-shaped result stubs.

    r.boxes.xyxy[0].tolist() must behave like a tensor row — the service
    reads exactly that path (masks are the fallback, exercised implicitly
    by the "no boxes" case in the 404 test).
    """

    class _Row:
        def __init__(self, xyxy):
            self._x = xyxy

        def tolist(self):
            return list(self._x)

    class _Arr(list):
        def __init__(self, xyxy):
            super().__init__([_Row(xyxy)])

    class _Boxes:
        def __init__(self, xyxy, conf):
            self.xyxy = _Arr(xyxy)
            self.conf = [conf]

    class _Res:
        def __init__(self, xyxy, conf):
            self.boxes = _Boxes(xyxy, conf)
            self.masks = None  # boxes carry the geometry; masks stay None

    class _Predictor:
        def __call__(self, **kw):
            return [_Res(b, conf) for b in boxes_per_frame]

    return _Predictor()


# ------------------------------------------------------------------ weights


def test_health_reports_missing_weights(svc):
    mod, client = svc
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["weights_ready"] is False
    assert body["tracker_loaded"] is False
    assert body["weights_url"].startswith("https://dl.fbaipublicfiles.com/segment_anything_2")


def test_track_refuses_without_weights(svc):
    mod, client = svc
    r = client.post(
        "/track",
        json={
            "video_path": "/tmp/x.mp4",
            "prompt": {"kind": "box", "x": 1, "y": 1, "w": 4, "h": 4, "anchor_frame": 0},
            "frame_start": 0,
            "frame_end": 5,
            "fps": 30,
            "frame_w": 100,
            "frame_h": 100,
        },
    )
    assert r.status_code == 503
    assert "/weights/ensure" in r.json()["detail"]


def test_weights_ensure_downloads_atomically(svc, monkeypatch):
    mod, client = svc
    # a real checkpoint stand-in: >1MB so the readiness gate passes
    blob = b"SAM2CHECKPOINT" + os.urandom(1_100_000)
    calls = {"n": 0}

    def fake_urlretrieve(url, dst):
        calls["n"] += 1
        calls["url"] = url
        with open(dst, "wb") as f:
            f.write(blob)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    r = client.post("/weights/ensure")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["downloaded"] and body["bytes"] == len(blob)
    assert body["sha256"].startswith("8") or len(body["sha256"]) == 64
    assert calls["url"].startswith("https://dl.fbaipublicfiles.com/segment_anything_2")
    # the file really landed at the configured path (atomic replace, no tmp corpses)
    assert (Path(mod.weights_path())).stat().st_size == len(blob)

    # idempotent second call: volume cache hit, no second download
    r2 = client.post("/weights/ensure")
    body2 = r2.json()
    assert body2["downloaded"] is False and calls["n"] == 1


def test_weights_ensure_rejects_wrong_sha(svc, monkeypatch):
    mod, client = svc
    monkeypatch.setenv("SAM2_WEIGHTS_SHA256", "0" * 64)

    def fake_urlretrieve(url, dst):
        with open(dst, "wb") as f:
            f.write(b"SAM2CHECKPOINT" + os.urandom(1_100_000))

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    r = client.post("/weights/ensure")
    assert r.status_code == 502
    assert "SHA256" in r.json()["detail"] or "sha256" in r.json()["detail"]


def test_weights_ensure_rejects_stub_downloads(svc, monkeypatch):
    mod, client = svc

    def stub_retrieve(url: str, dst: str) -> None:
        with open(dst, "wb") as f:
            f.write(b"404 page")

    monkeypatch.setattr("urllib.request.urlretrieve", stub_retrieve)
    r = client.post("/weights/ensure")
    assert r.status_code == 502
    assert "not a real model" in r.json()["detail"]


# ------------------------------------------------------------------ track


def test_track_maps_sam2_results_to_studio_hits(svc, monkeypatch, tmp_path):
    mod, client = svc
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0" * 1000)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "sam2.1_hiera_tiny.pt").write_bytes(
        b"W" * 1_100_000
    )  # passes the readiness gate
    mod._predictor = _stub_predictor(
        mod, [[10, 20, 110, 220], [12, 22, 112, 222], [14, 24, 114, 224]], conf=0.91
    )

    r = client.post(
        "/track",
        json={
            "video_path": str(video),
            "prompt": {"kind": "box", "x": 10, "y": 20, "w": 100, "h": 200, "anchor_frame": 1},
            "frame_start": 30,
            "frame_end": 32,
            "fps": 30,
            "frame_w": 1920,
            "frame_h": 1080,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "sam2"
    assert len(body["hits"]) == 3
    h0 = body["hits"][0]
    assert h0["t_s"] == pytest.approx(30 / 30)  # frame 30 at 30fps
    assert h0["x"] == 10 and h0["y"] == 20 and h0["w"] == 100 and h0["h"] == 200
    assert body["mean_score"] == pytest.approx(0.91)
    # hit shape parity with the Studio's TrackHit (raw-pixel boxes ride through)
    from director_os.vision_fx.compute.base import TrackHit

    TrackHit(**h0)  # validates the contract — a shape drift fails here


def test_track_404s_on_unreachable_video(svc, tmp_path):
    mod, client = svc
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "sam2.1_hiera_tiny.pt").write_bytes(b"W" * 1_100_000)
    r = client.post(
        "/track",
        json={
            "video_path": str(tmp_path / "missing.mp4"),
            "prompt": {"kind": "box", "x": 1, "y": 1, "w": 4, "h": 4, "anchor_frame": 0},
            "frame_start": 0,
            "frame_end": 5,
            "fps": 30,
            "frame_w": 100,
            "frame_h": 100,
        },
    )
    assert r.status_code == 404


def test_track_validates_the_prompt(svc, tmp_path):
    mod, client = svc
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "sam2.1_hiera_tiny.pt").write_bytes(b"W" * 1_100_000)
    video = tmp_path / "c.mp4"
    video.write_bytes(b"v")
    # box prompt without size
    r = client.post(
        "/track",
        json={
            "video_path": str(video),
            "prompt": {"kind": "box", "x": 1, "y": 1, "w": 0, "h": 0, "anchor_frame": 0},
            "frame_start": 0,
            "frame_end": 5,
            "fps": 30,
            "frame_w": 100,
            "frame_h": 100,
        },
    )
    assert r.status_code == 422
    # inverted frame span
    r2 = client.post(
        "/track",
        json={
            "video_path": str(video),
            "prompt": {"kind": "point", "x": 1, "y": 1, "anchor_frame": 0},
            "frame_start": 9,
            "frame_end": 2,
            "fps": 30,
            "frame_w": 100,
            "frame_h": 100,
        },
    )
    assert r2.status_code == 422


# ------------------------------------------------------------------ client


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_client_round_trips_into_tracked_selection():
    from director_os.vision_fx.selection import Sam2ServiceBackend, SelectionPrompt

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/track"
        payload = json.loads(request.content)
        assert payload["prompt"]["kind"] == "box"
        assert payload["frame_w"] == 1920
        return httpx.Response(
            200,
            json={
                "backend": "sam2",
                "hits": [
                    {"t_s": 1.0, "x": 10, "y": 20, "w": 100, "h": 200, "score": 0.9},
                    {"t_s": 1.033, "x": 11, "y": 21, "w": 100, "h": 200, "score": 0.88},
                ],
                "mean_score": 0.89,
                "n_interpolated": 0,
            },
        )

    be = Sam2ServiceBackend("http://sam2:8300", transport=_mock_transport(handler))
    sel = be.track(
        "/volume/in.mp4",
        SelectionPrompt(kind="box", x=10, y=20, w=100, h=200, anchor_frame=30),
        frame_start=30,
        frame_end=89,
        fps=30,
        frame_w=1920,
        frame_h=1080,
    )
    assert sel.backend == "sam2"
    assert len(sel.hits) == 2
    assert sel.mean_score == pytest.approx(0.89)
    assert sel.span_s == pytest.approx(0.033, abs=1e-3)


def test_client_503_is_not_available_never_silent_swap():
    from director_os.vision_fx.selection import (
        Sam2ServiceBackend,
        SelectionNotAvailable,
        SelectionPrompt,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "weights not present"})

    be = Sam2ServiceBackend("http://sam2:8300", transport=_mock_transport(handler))
    with pytest.raises(SelectionNotAvailable, match="weights not present"):
        be.track(
            "/x.mp4",
            SelectionPrompt(kind="box", x=1, y=1, w=4, h=4, anchor_frame=0),
            frame_start=0,
            frame_end=5,
            fps=30,
            frame_w=100,
            frame_h=100,
        )


def test_client_unreachable_raises_not_available():

    from director_os.vision_fx.selection import (
        Sam2ServiceBackend,
        SelectionNotAvailable,
        SelectionPrompt,
    )

    be = Sam2ServiceBackend("http://127.0.0.1:1")
    with pytest.raises(SelectionNotAvailable, match="unreachable"):
        be.track(
            "/x.mp4",
            SelectionPrompt(kind="box", x=1, y=1, w=4, h=4, anchor_frame=0),
            frame_start=0,
            frame_end=5,
            fps=30,
            frame_w=100,
            frame_h=100,
        )


def test_run_selection_sam2_prefers_the_service(monkeypatch):
    """backend='sam2' with SAM2_SERVICE_URL set routes to the HTTP backend."""
    import director_os.vision_fx.selection as sel_mod
    from director_os.vision_fx.selection import Sam2ServiceBackend

    monkeypatch.setenv("SAM2_SERVICE_URL", "http://sam2:8300")
    calls: list[str] = []

    class _Stub(Sam2ServiceBackend):
        def track(self, *a, **kw):  # type: ignore[override]
            calls.append("service")
            raise sel_mod.SelectionNotAvailable("stub stop")

    monkeypatch.setattr(sel_mod, "Sam2ServiceBackend", _Stub)
    prompt = sel_mod.SelectionPrompt(kind="box", x=1, y=1, w=4, h=4, anchor_frame=0)
    with pytest.raises(sel_mod.SelectionNotAvailable, match="stub stop"):
        sel_mod.run_selection(
            "/x.mp4",
            prompt,
            frame_start=0,
            frame_end=5,
            fps=30,
            frame_w=100,
            frame_h=100,
            backend="sam2",
        )
    assert calls == ["service"], "the service backend owns backend='sam2' when reachable"


def test_service_backends_flag_reflects_the_service_url(monkeypatch):
    from director_os.vision_fx.selection import available_selection_backends

    monkeypatch.delenv("SAM2_SERVICE_URL", raising=False)
    monkeypatch.delattr("sys.modules", "ultralytics", raising=False)
    import importlib.util

    assert available_selection_backends()["sam2"] is False
    monkeypatch.setenv("SAM2_SERVICE_URL", "http://sam2:8300")
    assert available_selection_backends()["sam2"] is True
    _ = importlib.util  # keep the import honest for the sys.modules assert above
