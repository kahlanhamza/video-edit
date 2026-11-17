"""License-plate redaction — the second detector family of the 3-pass law (issue #12).

Plates ride the SAME sweep/tracker/Pass-2/Pass-3 as faces (merge_redaction_tracks
is the seam). Real-model tests run when data/models/yolov8n-license-plate.onnx is
present (scripts/fetch_yolov8n_plate.py) and skip hermetically otherwise — the
synthetic plate fixture below is proven to fire the real detector at 0.65.
"""

from __future__ import annotations

import numpy as np
import pytest
from director_os.vision_fx.compute.base import FaceTrack, TrackHit
from director_os.vision_fx.compute.tracker import (
    EmptySweepError,
    merge_redaction_tracks,
    merge_tracks,
)
from director_os.vision_fx.compute.yolo_onnx import (
    PLATE_MODEL_FILENAME,
    resolve_input_size,
    resolve_plate_model_path,
)
from director_os.vision_fx.redact_track import assert_zero_leaks
from director_os.vision_fx.trajectory import repair_trajectories


def _plate_model_present() -> bool:
    return resolve_plate_model_path() is not None


def _track(tid: str, hits: list[tuple[float, int, int, int, int]]) -> FaceTrack:
    return FaceTrack(
        track_id=tid,
        hits=[TrackHit(t_s=t, x=x, y=y, w=w, h=h, score=0.9) for (t, x, y, w, h) in hits],
    )


# ------------------------------------------------------------------ resolver


def test_resolver_env_override_wins(monkeypatch, tmp_path):
    fake = tmp_path / "plate.onnx"
    fake.write_bytes(b"onnx")
    monkeypatch.setenv("DIRECTOR_YOLOV8N_PLATE_PATH", str(fake))
    assert resolve_plate_model_path() == fake


def test_resolver_explicit_missing_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_plate_model_path(tmp_path / "nope.onnx")


def test_resolver_default_filename_is_pinned():
    assert PLATE_MODEL_FILENAME == "yolov8n-license-plate.onnx"
    # the ml-debi export is a FIXED 640x640 graph — resolve_input_size must honor it
    eff, note = resolve_input_size([1, 3, 640, 640], requested=960)
    assert eff == 640
    assert "fixed" in note


# -------------------------------------------------------------- merge / refuse


def test_merge_tracks_reprefixes_plate_person_ids():
    faces = [_track("person_1", [(0.0, 10, 10, 20, 20)])]
    plates = [_track("person_1", [(0.0, 100, 100, 60, 20)])]  # same raw id — collision
    merged = merge_redaction_tracks(faces, plates)
    assert [t.track_id for t in merged] == ["person_1", "plate_1"]
    # hits preserved verbatim through the rename
    assert merged[1].hits[0].x == 100


def test_merge_tracks_two_sweeps_no_collision_at_all():
    faces = [
        _track("person_1", [(0.0, 10, 10, 20, 20)]),
        _track("person_2", [(0.1, 40, 10, 20, 20)]),
    ]
    plates = [_track("person_2", [(0.0, 100, 100, 60, 20)])]
    merged = merge_tracks(faces, plates)
    ids = [t.track_id for t in merged]
    assert len(set(ids)) == 3  # all unique
    assert ids[2] != "person_2"  # did NOT silently alias the face track


def test_empty_both_sweeps_is_a_broken_sweep():
    with pytest.raises(EmptySweepError, match="ZERO faces and ZERO plates"):
        merge_redaction_tracks([], [])


def test_plates_only_source_is_a_valid_plan():
    # a dash-cam pointed at traffic has no faces but plenty of plates
    plates = [_track("person_3", [(0.0, 100, 100, 60, 20), (0.1, 102, 100, 60, 20)])]
    merged = merge_redaction_tracks([], plates)
    assert [t.track_id for t in merged] == ["plate_3"]


def test_merged_face_plate_plan_passes_zero_leak_audit():
    fps = 30.0
    faces = [_track("person_1", [(0.0, 500, 300, 90, 110), (0.3, 520, 310, 90, 110)])]
    plates = [
        _track(
            "person_2",
            [(0.0, 800, 700, 120, 36), (0.2, 810, 700, 120, 36), (0.4, 820, 700, 120, 36)],
        )
    ]
    merged = merge_redaction_tracks(faces, plates)
    plan = repair_trajectories(merged, fps=30, frame_w=1920, frame_h=1080, n_frames=30)
    assert {t.track_id for t in plan.trajectories} == {"person_1", "plate_2"}
    # plate boxes survive Pass 2 on their own frames (envelope covers raw)
    frame12 = plan.boxes_at(12)  # 0.4 s * 30 fps
    assert any(b.x <= 820 and b.x + b.w >= 940 and b.y <= 700 for b in frame12)
    assert_zero_leaks(plan, merged, fps)  # the law: ANY leak raises


def test_plate_gap_bridging_holds_identity():
    # plate disappears (occluded by a truck) for 12 frames — <= 25 => bridge, not split
    plates = [_track("person_1", [(0.0, 400, 600, 100, 32), (0.8, 460, 600, 100, 32)])]
    plan = repair_trajectories(plates, fps=30, frame_w=1920, frame_h=1080, n_frames=30)
    tr = plan.trajectories[0]
    idxs = [b.frame_idx for b in tr.boxes]
    assert idxs == list(range(idxs[0], idxs[-1] + 1))  # dense across the gap
    bridged = [b for b in tr.boxes if b.bridged]
    assert bridged, "gap must be bridged (a stale-hole plan re-leaks the plate)"


# ------------------------------------------------- real model (optional, hermetic)


@pytest.mark.skipif(not _plate_model_present(), reason="plate model not fetched")
def test_real_plate_detector_fires_on_synthetic_plate():
    import cv2
    from director_os.vision_fx.compute.yolo_onnx import YoloOnnxPlateDetector

    det = YoloOnnxPlateDetector(conf_threshold=0.25)
    # synthetic US-style plate on a dark body — proven to fire at 0.65
    img = np.full((480, 720, 3), (60, 70, 80), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (680, 440), (45, 55, 65), -1)
    cv2.rectangle(img, (150, 300), (570, 400), (235, 235, 230), -1)
    cv2.rectangle(img, (150, 300), (570, 400), (0, 0, 0), 3)
    for i, ch in enumerate("8XTR452"):
        cv2.putText(
            img,
            ch,
            (185 + i * 55, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (20, 30, 90),
            3,
            cv2.LINE_AA,
        )
    hits = det.detect(img)
    assert hits, "synthetic plate must fire"
    x, y, w, h, score = max(hits, key=lambda t: t[4])
    assert score >= 0.3
    # box must cover the drawn plate (150,300)-(570,400) generously
    inter_x = max(0, min(x + w, 570) - max(x, 150))
    inter_y = max(0, min(y + h, 400) - max(y, 300))
    iou_area = inter_x * inter_y / (420 * 100)
    assert iou_area > 0.4, f"detector box off target: {(x, y, w, h)}"


@pytest.mark.skipif(not _plate_model_present(), reason="plate model not fetched")
def test_real_plate_detector_not_fires_on_face_only_frame():

    from director_os.vision_fx.compute.yolo_onnx import YoloOnnxPlateDetector

    det = YoloOnnxPlateDetector(conf_threshold=0.25)
    img = np.zeros((480, 720, 3), dtype=np.uint8)  # black frame — no plates
    assert det.detect(img) == []


def test_backend_capabilities_report_plates():
    from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend

    caps = LocalOnnxCpuBackend().capabilities()
    assert caps["plates"] == _plate_model_present()
    assert caps["faces"] is True


@pytest.mark.skipif(not _plate_model_present(), reason="plate model not fetched")
def test_backend_detect_plates_runs_end_to_end(tmp_path):
    """Full Pass-1 path on a real decode sweep: video -> plate tracks (raw pixels)."""
    import subprocess

    clip = tmp_path / "plate_clip.mp4"
    # same synthetic plate, baked into a 2s 30fps video via ffmpeg lavfi color + draw
    import cv2

    img = np.full((240, 320, 3), (60, 70, 80), dtype=np.uint8)
    cv2.rectangle(img, (60, 160), (260, 210), (235, 235, 230), -1)
    cv2.rectangle(img, (60, 160), (260, 210), (0, 0, 0), 2)
    for i, ch in enumerate("8XTR452"):
        cv2.putText(
            img,
            ch,
            (72 + i * 26, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 30, 90),
            2,
            cv2.LINE_AA,
        )
    frame_path = tmp_path / "plate_frame.png"
    cv2.imwrite(str(frame_path), img)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(frame_path),
            "-t",
            "2",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend

    tracks = LocalOnnxCpuBackend().detect_plates(clip, sample_every_s=0.1)
    assert tracks, "static synthetic plate must track"
    assert all(t.track_id.startswith("person_") for t in tracks)  # pre-merge ids
    x0 = tracks[0].hits[0].x
    assert 40 <= x0 <= 90  # plate drawn at x=60 (± letterbox decode tolerance)
