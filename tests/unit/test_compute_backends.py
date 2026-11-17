"""vision_fx.compute — port selection, tracker interpolation, adapter contracts.

Hermetic: the local backend runs with an INJECTED fake detector (no model
file, no onnxruntime needed); the Kaggle adapter's wire path is covered by
MockTransport in test_kaggle_client.py; the live YOLO/Kaggle/ElevenLabs
validation runs via scripts/e2e_compute_backends.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from director_os.core.config import get_settings
from director_os.vision_fx.compute import (
    ComputeNotAvailable,
    get_compute_backend,
)
from director_os.vision_fx.compute.kaggle_batch import KaggleBatchGpuBackend
from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend
from director_os.vision_fx.compute.tracker import track_and_interpolate
from director_os.vision_fx.compute.yolo_onnx import YoloOnnxFaceDetector


# ----------------------------------------------------------------- selector
def test_selector_default_is_local(monkeypatch):
    monkeypatch.delenv("COMPUTE_BACKEND", raising=False)
    get_settings.cache_clear()
    assert get_compute_backend().name == "local-onnx-cpu"
    get_settings.cache_clear()


def test_selector_kaggle_and_bogus_mode(monkeypatch):
    monkeypatch.setenv("COMPUTE_BACKEND", "kaggle")
    monkeypatch.setenv("KAGGLE_KEY", "KGAT_x")
    get_settings.cache_clear()
    assert get_compute_backend().name == "kaggle-gpu"

    monkeypatch.setenv("COMPUTE_BACKEND", "wat")
    get_settings.cache_clear()
    with pytest.raises(ComputeNotAvailable, match="local|kaggle|modal|auto"):
        get_compute_backend()
    monkeypatch.setenv("COMPUTE_BACKEND", "local")
    get_settings.cache_clear()


def test_selector_auto_routes_heavy_to_kaggle(monkeypatch):
    monkeypatch.setenv("COMPUTE_BACKEND", "auto")
    monkeypatch.setenv("KAGGLE_KEY", "KGAT_x")
    get_settings.cache_clear()
    assert get_compute_backend("faces").name == "local-onnx-cpu"
    assert get_compute_backend("stems").name == "kaggle-gpu"
    monkeypatch.setenv("COMPUTE_BACKEND", "local")
    get_settings.cache_clear()


def test_modal_unconfigured_message_mentions_env_flip(monkeypatch):
    from director_os.vision_fx.compute.modal_gpu import ModalGpuBackend

    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ComputeNotAvailable, match="MODAL_TOKEN_ID"):
        ModalGpuBackend()
    get_settings.cache_clear()


# ------------------------------------------------------------------ tracker
def _det(frame_boxes: list[list[tuple[int, int, int, int, float]]], t: list[float]):
    return frame_boxes, t


def test_tracker_min_hits_prunes_one_shot_false_positives():
    # a single suspicious box on frame 1 must NOT become a track
    dets = [[(100, 100, 40, 40, 0.9)], [], [], []]
    t = [0.0, 0.1, 0.2, 0.3]
    tracks = track_and_interpolate(dets, t, min_hits=3)
    assert tracks == []


def test_tracker_persistent_ids_and_stride_interpolation():
    # a face gliding right across 6 sampled frames at 0.1s — stride-3 gaps
    # get filled with interpolated hits (t = 0.1 and 0.4 inside the span)
    dets = [
        [(100, 50, 40, 40, 0.91)],
        [(110, 50, 40, 40, 0.92)],
        [(120, 50, 40, 40, 0.90)],
        [(130, 50, 40, 40, 0.93)],
        [(140, 50, 40, 40, 0.89)],
        [(150, 50, 40, 40, 0.94)],
    ]
    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    tracks = track_and_interpolate(dets, t)
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.track_id == "person_1"
    assert tr.t_start_s == pytest.approx(0.0)
    assert tr.t_end_s == pytest.approx(0.5)
    assert tr.n_real == 6
    assert all(not h.interpolated for h in tr.hits)


def test_tracker_interpolates_when_detector_misses_frames():
    # real hits at 0.0, 0.1, [0.2, 0.3 missed], 0.4, 0.5 -> 4 real >= min_hits,
    # the two-frame gap gets interpolated hits gliding 120 -> 180
    dets = [
        [(100, 50, 40, 40, 0.9)],
        [(120, 50, 40, 40, 0.9)],
        [],
        [],
        [(180, 50, 40, 40, 0.9)],
        [(200, 50, 40, 40, 0.9)],
    ]
    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    tracks = track_and_interpolate(dets, t, max_age=6)
    assert len(tracks) == 1
    hits = tracks[0].hits
    assert [h.t_s for h in hits] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert [h.interpolated for h in hits] == [False, False, True, True, False, False]
    # linear glide across the gap (EMA smoothing may shave a pixel)
    assert hits[2].x == pytest.approx(140, abs=2)
    assert hits[3].x == pytest.approx(160, abs=2)


def test_tracker_splits_distant_boxes_into_distinct_ids():
    # two far-apart faces: never merged into one track
    dets = [
        [(100, 50, 40, 40, 0.9), (900, 50, 40, 40, 0.9)],
        [(105, 50, 40, 40, 0.9), (905, 50, 40, 40, 0.9)],
        [(110, 50, 40, 40, 0.9), (910, 50, 40, 40, 0.9)],
    ]
    t = [0.0, 0.1, 0.2]
    tracks = track_and_interpolate(dets, t)
    assert len(tracks) == 2
    assert {tr.track_id for tr in tracks} == {"person_1", "person_2"}


# ------------------------------------------------------------- local backend
class _FakeDetector:
    name = "fake"

    def __init__(self, boxes_per_call):
        self.boxes_per_call = boxes_per_call
        self.calls = 0

    def detect(self, frame):
        i = min(self.calls, len(self.boxes_per_call) - 1)
        self.calls += 1
        return self.boxes_per_call[i]


def _write_test_video(
    path: Path, n_frames: int = 12, fps: float = 30.0, w: int = 640, h: int = 360
):
    """Tiny synthetic mp4 via lavfi color source (no external assets)."""
    import subprocess

    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s={w}x{h}:d={n_frames / fps}:r={fps}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-300:]


def test_local_backend_stride3_and_tracking(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_test_video(video, n_frames=30, fps=30.0)  # 1.0 s -> 10 sweeps at stride 3
    boxes = [[(10, 10, 50, 50, 0.9)]]  # same box every sampled frame
    backend = LocalOnnxCpuBackend(detector=_FakeDetector(boxes), stride=3)
    tracks = backend.detect_faces(video, sample_every_s=0.1)
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.n_real == 10  # 30 frames / stride 3 = 10 sampled sweeps
    assert backend._last_detector_name == "fake"


def test_local_backend_open_failure_is_loud(tmp_path):
    backend = LocalOnnxCpuBackend(detector=_FakeDetector([]))
    with pytest.raises(ComputeNotAvailable, match="cannot open"):
        backend.detect_faces(tmp_path / "missing.mp4")


def test_local_backend_stems_not_available_without_demucs():
    backend = LocalOnnxCpuBackend(detector=_FakeDetector([]))
    from director_os.vision_fx.compute.base import Span

    with pytest.raises(ComputeNotAvailable, match="kaggle|demucs"):
        backend.separate_stems("x.wav", [Span(start_s=0, end_s=1)])


def test_yolo_detector_requires_model_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        YoloOnnxFaceDetector(model_path="/nonexistent/yolo.onnx")


# ------------------------------------------------------------- kaggle adapter
def test_kaggle_backend_requires_kernel_reachable_url():
    b = KaggleBatchGpuBackend(client=_client_stub())
    with pytest.raises(ComputeNotAvailable, match="video_url"):
        b.detect_faces("/local/file.mp4")
    with pytest.raises(ComputeNotAvailable, match="audio_url"):
        b.transcribe_align("x.wav")


def test_kaggle_backend_parses_kernel_output_and_tracks(monkeypatch, tmp_path):
    """detections.json from the kernel output dir -> same FaceTrack contract."""
    b = KaggleBatchGpuBackend(client=_client_stub(), out_dir=tmp_path)

    class FakeClient:
        def run_kernel(self, **kwargs):
            out = Path(kwargs["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "detections.json").write_text(
                json.dumps(
                    {
                        "fps": 30.0,
                        "sample_every_s": 0.1,
                        "frames_sampled": 4,
                        "detections": [
                            {"t_s": 0.0, "x": 100, "y": 50, "w": 40, "h": 40, "score": 0.9},
                            {"t_s": 0.1, "x": 120, "y": 50, "w": 40, "h": 40, "score": 0.9},
                            {"t_s": 0.2, "x": 140, "y": 50, "w": 40, "h": 40, "score": 0.9},
                            {"t_s": 0.3, "x": 160, "y": 50, "w": 40, "h": 40, "score": 0.9},
                        ],
                    }
                )
            )
            from director_os.core.kaggle import KernelRunResult

            return KernelRunResult(slug="u/s", url="u", status="complete", out_dir=out)

    b._client = FakeClient()  # property has no setter; inject the backing field
    tracks = b.detect_faces(
        "/local/file.mp4", video_url="https://r2.example/v.webm", out_dir=tmp_path
    )
    assert len(tracks) == 1
    assert tracks[0].t_end_s == pytest.approx(0.3)


def _client_stub():
    class _Stub:
        pass

    return _Stub()
