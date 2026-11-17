"""Unit tests — perception.faces (detector sweep + region consolidation)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from director_os.core.models import BBox
from director_os.perception.faces import (
    FaceDetection,
    FaceSweep,
    FaceSweepStats,
    YuNetFaceDetector,
    cluster_to_regions,
    detect_faces,
    make_detector,
)


class FakeDetector:
    name = "fake"

    def __init__(self, hits: list[tuple[int, int, int, int, float]]):
        self._hits = hits

    def detect(self, frame_bgr) -> list[tuple[int, int, int, int, float]]:
        return self._hits


def _sweep(dets: list[FaceDetection]) -> FaceSweep:
    return FaceSweep(
        stats=FaceSweepStats(
            frames_sampled=len({d.t_s for d in dets}),
            frames_decoded_failed=0,
            detections=len(dets),
            detector="fake",
            sample_every_s=1.0,
        ),
        detections=dets,
    )


def _mk(t: float, x: int, y: int, w: int = 90, h: int = 110, score: float = 0.9) -> FaceDetection:
    return FaceDetection(t_s=t, bbox=BBox(x=x, y=y, w=w, h=h), score=score)


# ----------------------------------------------------------------- clustering
def test_cluster_ignores_one_shot_false_positive() -> None:
    dets = [_mk(t, 500, 300) for t in range(12)] + [_mk(99, 100, 100, 40, 40)]
    regions = cluster_to_regions(_sweep(dets), min_hits=3)
    assert len(regions) == 1
    r = regions[0]
    # cluster spans x=500..590, y=300..410 padded 10% -> 490..600 x 289..421
    assert r.bbox.x <= 500 and r.bbox.x + r.bbox.w >= 590
    assert r.bbox.y <= 300 and r.bbox.y + r.bbox.h >= 410
    assert r.scope == "global" and r.mode == "static"
    assert r.track_id == "face_cluster_1"


def test_cluster_max_regions_keeps_top_by_hits() -> None:
    hot = [_mk(t, 400, 200) for t in range(10)]
    mid = [_mk(t, 1200, 700) for t in range(5)]
    cold = [_mk(t, 900, 400, 60, 70) for t in range(2)]
    regions = cluster_to_regions(_sweep(hot + mid + cold), min_hits=3, max_regions=2)
    assert len(regions) == 2
    ids = {r.track_id for r in regions}
    assert ids == {"face_cluster_1", "face_cluster_2"}
    # cluster_1 = the 10-hit cluster around (400,200)
    first = regions[0]
    assert first.bbox.x < 500 and first.bbox.y < 300


def test_cluster_padding_clips_to_frame_bounds() -> None:
    dets = [_mk(t, 0, 0, 80, 90) for t in range(6)]
    regions = cluster_to_regions(_sweep(dets), pad_frac=0.2, frame_w=1920, frame_h=1080)
    r = regions[0]
    assert r.bbox.x == 0 and r.bbox.y == 0  # padding clipped at origin
    assert r.bbox.w == 96 and r.bbox.h == 108  # 80/90 padded 20% (no clip on far side)


def test_cluster_no_survivors_returns_empty() -> None:
    dets = [_mk(0, 10, 10, 30, 30)]
    assert cluster_to_regions(_sweep(dets), min_hits=3) == []


# -------------------------------------------------------------------- sweep
def _tiny_video(tmp_path: Path) -> Path:
    dest = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=10:duration=3",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


def test_detect_faces_stride_threshold_and_coords(tmp_path: Path) -> None:
    vid = _tiny_video(tmp_path)
    # fixed hit on every sampled frame; threshold drops the low-score copy
    fake = FakeDetector([(100, 60, 80, 90, 0.92), (0, 0, 40, 40, 0.3)])
    sweep = detect_faces(vid, sample_every_s=1.0, detector=fake, score_threshold=0.5)
    # 10 fps, 3 s, 1 fps sampling -> 3 frames
    assert sweep.stats.frames_sampled == 3
    assert sweep.stats.detections == 3  # one per sampled frame (low-score dropped)
    assert sweep.stats.detector == "fake"
    assert all(d.t_s in (0.0, 1.0, 2.0) for d in sweep.detections)
    assert all(d.bbox.x == 100 and d.bbox.w == 80 for d in sweep.detections)


def test_detect_faces_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cannot open video"):
        detect_faces(tmp_path / "nope.mp4", detector=FakeDetector([]))


def test_make_detector_loads_packaged_yunet() -> None:
    # the model ships inside the package — a fresh checkout needs no download
    det = make_detector()
    assert isinstance(det, YuNetFaceDetector)
    hits = det.detect(_blank_frame())
    assert hits == []  # no faces in a flat frame; detector runs without error


def test_yunet_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="YuNet model missing"):
        YuNetFaceDetector(tmp_path / "absent.onnx")


def _blank_frame():
    import numpy as np

    return np.zeros((240, 320, 3), dtype=np.uint8)
