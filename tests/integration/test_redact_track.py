"""Integration — deterministic 3-pass tracked redaction on a FAST-PAN synthetic clip.

The gate from issue #14: build a 30fps CFR clip where a high-contrast "face"
patch whips across frame (the classic bodycam pan leak), run the full
detect -> repair -> render battery with a GROUND-TRUTH stub detector, and
prove **0 unblurred face frames inside active tracks** — pixel-verified, not
just geometry-verified. Also verifies the debug overlay draws and CFR skip.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from director_os.core.qa_probe import probe
from director_os.vision_fx.compute.tracker import track_and_interpolate
from director_os.vision_fx.redact_track import (
    normalize_cfr,
    render_tracked_redaction,
    video_n_frames,
)
from director_os.vision_fx.trajectory import repair_trajectories

FPS = 30
DUR_S = 4
W, H = 320, 240
# the "face": bright white patch sweeping 220 px over 2 s (the whip-pan) then back
FACE_W, FACE_H = 44, 52


def _face_center(frame_idx: int) -> tuple[int, int]:
    if frame_idx <= 60:  # 0-2 s: fast pan left->right
        t = frame_idx / 60.0
        x = int(40 + 220 * t)
        y = int(90 + 30 * t)
    elif frame_idx <= 90:  # 2-3 s: hold (detector "dropouts" mid-hold)
        x, y = 260, 120
    else:  # 3-4 s: slow drift back
        t = (frame_idx - 90) / 90.0
        x = int(260 - 180 * t)
        y = int(120 - 20 * t)
    return x, y


def _make_clip(tmp_path: Path) -> Path:
    """Synthesize CFR 30fps clip: noisy bg + moving white patch + 1kHz audio."""
    import cv2

    out = tmp_path / "pan_src.mp4"
    frames = np.zeros((DUR_S * FPS, H, W, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for i in range(DUR_S * FPS):
        frames[i] = rng.integers(20, 60, size=(H, W, 3), dtype=np.uint8)
        x, y = _face_center(i)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + FACE_W), min(H, y + FACE_H)
        if x1 > x0 and y1 > y0:
            frames[i][y0:y1, x0:x1] = (240, 240, 240)
    tmp = tmp_path / "raw.avi"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(tmp),
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=4",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def _stub_sweep(video: Path) -> tuple[list, list, list]:
    """Pass 1 with ground truth: detections on sampled frames + exact motion.

    The pan IS the camera motion here, so motion vectors are measured truth
    (what LK flow would find on the noisy background).
    """
    dets, ts, motion = [], [], []
    for i in range(0, DUR_S * FPS, 3):
        x, y = _face_center(i)
        # detector dropouts during the hold (7 sampled frames, <=25 real frames)
        if 66 <= i <= 84:
            dets.append([])
        else:
            dets.append([(x, y, FACE_W, FACE_H, 0.92)])
        ts.append(round(i / FPS, 3))
        if i >= 3:
            px, _ = _face_center(i - 3)
            cx, _ = _face_center(i)
            motion.append((float(cx - px), 1.0))  # patch is the only moving thing
        else:
            motion.append((0.0, 0.0))
    return dets, ts, motion


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    return _make_clip(tmp_path_factory.mktemp("redact"))


def test_cfr_source_skips_normalization(clip: Path):
    dst = clip.parent / "cfr.mp4"
    out = normalize_cfr(clip, dst, fps=FPS)
    assert out == clip  # fixture is exact CFR 30fps — frame indices trustworthy
    assert video_n_frames(clip) == DUR_S * FPS


def test_full_pipeline_zero_leak(clip: Path, tmp_path: Path):
    """THE gate: detect -> repair -> assert_zero_leaks -> render -> pixel-proof."""
    from director_os.vision_fx.redact_track import assert_zero_leaks

    dets, ts, motion = _stub_sweep(clip)
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=motion, track_buffer=60)
    assert len(tracks) == 1, "whip-pan must keep ONE identity through the dropout"

    plan = repair_trajectories(tracks, fps=FPS, frame_w=W, frame_h=H, n_frames=DUR_S * FPS)
    assert_zero_leaks(plan, tracks, FPS)  # geometry contract

    out = render_tracked_redaction(
        clip,
        plan,
        tmp_path / "redacted.mp4",
        mode="gauss",
        sigma=20,
        n_frames_expected=DUR_S * FPS,
    )
    info = probe(out)
    assert info.has_video and info.has_audio
    assert abs(info.duration_s - DUR_S) < 0.3
    _pixel_proof(clip, out, plan)


def _pixel_proof(src: Path, rendered: Path, plan) -> None:
    """Frame-by-frame luminance-contrast check inside the raw face box.

    A blurred patch has near-flat luminance; the un-blurred white-on-noise
    patch has a huge spread. Any frame inside an active track where the raw
    face box still shows the patch's contrast = a LEAK = test failure.
    """
    import cv2

    cap_in = cv2.VideoCapture(str(src))
    cap_out = cv2.VideoCapture(str(rendered))
    leaks = []
    idx = 0
    while True:
        ok_in, fin = cap_in.read()
        ok_out, fout = cap_out.read()
        if not ok_in or not ok_out:
            break
        covered_boxes = plan.boxes_at(idx)
        if covered_boxes:
            x, y = _face_center(idx)
            raw_box = (max(0, x), max(0, y), min(W, x + FACE_W), min(H, y + FACE_H))
            if raw_box[2] > raw_box[0] and raw_box[3] > raw_box[1]:
                region_in = fin[raw_box[1] : raw_box[3], raw_box[0] : raw_box[2]]
                region_out = fout[raw_box[1] : raw_box[3], raw_box[0] : raw_box[2]]
                gray_in = cv2.cvtColor(region_in, cv2.COLOR_BGR2GRAY)
                gray_out = cv2.cvtColor(region_out, cv2.COLOR_BGR2GRAY)
                contrast_in = float(gray_in.max() - gray_in.min())
                contrast_out = float(gray_out.max() - gray_out.min())
                if contrast_in > 120 and contrast_out > 0.45 * contrast_in:
                    leaks.append((idx, contrast_in, contrast_out))
        idx += 1
    cap_in.release()
    cap_out.release()
    assert leaks == [], f"UNBLURRED FACE FRAMES: {leaks[:6]}"


def test_debug_overlay_draws(clip: Path, tmp_path: Path):
    """debug_overlay=True: padded red boxes visible on rendered frames."""
    dets, ts, motion = _stub_sweep(clip)
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=motion, track_buffer=60)
    plan = repair_trajectories(tracks, fps=FPS, frame_w=W, frame_h=H, n_frames=DUR_S * FPS)
    out = render_tracked_redaction(
        clip,
        plan,
        tmp_path / "debug.mp4",
        debug_overlay=True,
        n_frames_expected=DUR_S * FPS,
    )
    import cv2

    cap = cv2.VideoCapture(str(out))
    idx = 0
    red_seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if plan.boxes_at(idx):
            b, g, r = (
                frame[:, :, 0].astype(int),
                frame[:, :, 1].astype(int),
                frame[:, :, 2].astype(int),
            )
            red_mask = (r - b > 60) & (r - g > 60)
            red_seen += int(red_mask.sum() > 20)
        idx += 1
    cap.release()
    assert red_seen >= 30, f"debug overlay drew red boxes on only {red_seen} frames"


def test_exclude_track_ids_blurs_nothing(clip: Path, tmp_path: Path):
    dets, ts, motion = _stub_sweep(clip)
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=motion, track_buffer=60)
    plan = repair_trajectories(tracks, fps=FPS, frame_w=W, frame_h=H, n_frames=DUR_S * FPS)
    out = render_tracked_redaction(
        clip,
        plan,
        tmp_path / "excluded.mp4",
        exclude_track_ids={t.track_id for t in tracks},
        n_frames_expected=DUR_S * FPS,
    )
    import cv2

    cap = cv2.VideoCapture(str(out))
    _, frame = cap.read()
    cap.release()
    x, y = _face_center(0)
    patch = frame[y : y + FACE_H, x : x + FACE_W]
    assert patch.min() > 200, "excluded track must stay un-blurred"


def test_mosaic_mode_also_covers(clip: Path, tmp_path: Path):
    dets, ts, motion = _stub_sweep(clip)
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=motion, track_buffer=60)
    plan = repair_trajectories(tracks, fps=FPS, frame_w=W, frame_h=H, n_frames=DUR_S * FPS)
    out = render_tracked_redaction(
        clip,
        plan,
        tmp_path / "mosaic.mp4",
        mode="mosaic",
        n_frames_expected=DUR_S * FPS,
    )
    _pixel_proof(clip, out, plan)
