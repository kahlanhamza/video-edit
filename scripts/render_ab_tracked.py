#!/usr/bin/env python3
"""Tracked redaction for the voice-fix A/B window — the worker seam, no DB.

Cuts the A/B window (master 22.0-38.5 s) from the ORIGINAL 1080p source,
CFR-normalizes, runs the full perception sweep (YuNet faces + plates ->
merge -> Pass-2 trajectory repair -> zero-leak audit) and renders the CLEAN
tracked blur (no debug overlay). This is the video the A/B audio muxes onto,
replacing the stale pre-3-pass master background the user flagged.

Usage: uv run python scripts/render_ab_tracked.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path("data/media/bodycam_eustis.webm")
MASTER = Path("/home/z/my-project/download/eustis_bodycam_master_720p.mp4")
SS_S = 37.0  # webm timeline = master + 15 s -> covers master 22.0-38.5 (A/B audio 22.62-37.62)
T_S = 16.5
# webm is 1080p; the window is downscaled to 720p — the 720p sweep found full
# face coverage for this scene (the 1080p sweep missed every face after 8 s)
# and the master deliverable is 720p anyway. Cutting from the webm avoids the
# stale baked blur boxes the master carries from the pre-3-pass render.
SCALE = "scale=1280:720"
OUT = Path("output/voice_fix")
WIN = OUT / "ab_window_raw.mp4"
FINAL = OUT / "ab_window_tracked.mp4"


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: {r.stderr[-400:].decode(errors='replace')}")


def _verify_cut_alignment() -> None:
    """The webm is VFR: -ss input-seek snaps to keyframes and can land 25 s
    early (measured). Prove the cut landed on the master timeline: clip t=5
    must match master t=27 (22+5, same content — the master carries baked
    boxes, so SSIM is thresholded loosely)."""
    a = OUT / "align_clip.jpg"
    b = OUT / "align_master.jpg"
    _run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "5", "-i", str(WIN),
         "-frames:v", "1", "-vf", "scale=160:90,format=gray", str(a)],
        "cut align probe",
    )
    _run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "27", "-i", str(MASTER),
         "-frames:v", "1", "-vf", "scale=160:90,format=gray", str(b)],
        "master align probe",
    )
    r = subprocess.run(
        ["ffmpeg", "-loglevel", "info", "-i", str(a), "-i", str(b),
         "-filter_complex", "ssim", "-f", "null", "-"],
        capture_output=True,
    )
    import re

    m = re.search(r"SSIM[^A]*All:([0-9.]+)", r.stderr.decode(errors="replace"), re.IGNORECASE)
    ssim = float(m.group(1)) if m else 0.0
    if ssim < 0.60:
        raise SystemExit(
            f"CUT MISALIGNED: clip(t5) vs master(t27) SSIM {ssim:.3f} < 0.60 — "
            "the webm seek did not land at master 22.0; refusing to render the wrong window"
        )
    print(f"cut alignment OK: SSIM {ssim:.3f}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not WIN.exists():
        # OUTPUT seek (-ss after -i): the webm is VFR and input-seek snapped to
        # a keyframe ~25 s early (measured 2026-11-18) — decode-and-discard is
        # slower but frame-exact.
        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(SRC),
                "-ss", f"{SS_S:.3f}", "-t", f"{T_S:.3f}",
                "-map", "0:v:0", "-an",
                "-vf", SCALE,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(WIN),
            ],
            "cut A/B window from original webm",
        )
    _verify_cut_alignment()
    print(f"window clip: {WIN} {WIN.stat().st_size / 1e6:.1f}MB")

    from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend
    from director_os.vision_fx.compute.tracker import merge_redaction_tracks
    from director_os.vision_fx.redact_track import (
        assert_zero_leaks,
        normalize_cfr,
        render_tracked_redaction,
        video_n_frames,
    )
    from director_os.vision_fx.trajectory import repair_trajectories

    cfr = normalize_cfr(WIN, OUT / "ab_window_cfr.mp4")
    n_frames = video_n_frames(cfr)
    print(f"cfr: {cfr} frames={n_frames}")

    backend = LocalOnnxCpuBackend()
    caps = backend.capabilities()
    faces = backend.detect_faces(cfr, max_samples=600)
    print(f"face tracks: {len(faces)}")
    for t in faces:
        print(f"  {t.track_id}: {t.n_real} hits, {t.t_start_s:.1f}-{t.t_end_s:.1f}s, min {t.min_score:.2f}")
    plates = []
    if caps.get("plates"):
        plates = backend.detect_plates(cfr, max_samples=600)
        print(f"plate tracks: {len(plates)}")
    merged = merge_redaction_tracks(faces, plates)
    plan = repair_trajectories(merged, fps=30, frame_w=1280, frame_h=720, n_frames=n_frames or 0)
    assert_zero_leaks(plan, merged, 30.0)
    n_boxes = sum(len(t.boxes) for t in plan.trajectories)
    print(f"plan OK: {len(plan.trajectories)} trajectories, {n_boxes} per-frame boxes, 0 leaks")

    # persistence for offline diagnosis (no re-detection needed later)
    import json as _json

    (OUT / "ab_tracks.json").write_text(
        _json.dumps(
            [
                {
                    "track_id": t.track_id,
                    "n_real": t.n_real,
                    "span_s": [round(t.t_start_s, 2), round(t.t_end_s, 2)],
                    "min_score": round(t.min_score, 3),
                }
                for t in merged
            ],
            indent=2,
        )
    )
    (OUT / "ab_plan.json").write_text(plan.model_dump_json())
    print("plan + tracks JSON written")

    render_tracked_redaction(cfr, plan, FINAL, mode="gauss", n_frames_expected=n_frames)
    print(f"tracked clean render: {FINAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
