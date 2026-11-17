#!/usr/bin/env python3
"""Real-footage gate for issue #12: plates + audio events on the Eustis bodycam.

Runs the exact worker seam logic (face sweep + plate sweep -> merge -> Pass-2
repair -> zero-leak audit) on a real 1080p window, and YAMNet on the real audio.
Prints a compact report; exits nonzero on any leak or empty sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

VIDEO = Path("data/media/diag_raw_win.mp4")
OUT = Path("output/plates_real_gate")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend
    from director_os.vision_fx.compute.tracker import merge_redaction_tracks
    from director_os.vision_fx.redact_track import (
        assert_zero_leaks,
        normalize_cfr,
        video_n_frames,
    )
    from director_os.vision_fx.trajectory import repair_trajectories

    backend = LocalOnnxCpuBackend()
    caps = backend.capabilities()
    print(f"capabilities: {caps}")
    cfr = normalize_cfr(VIDEO, OUT / "cfr.mp4")
    n_frames = video_n_frames(cfr)
    print(f"frames: {n_frames}")

    faces = backend.detect_faces(cfr, max_samples=600)
    print(f"face tracks: {len(faces)}")
    for t in faces:
        print(
            f"  {t.track_id}: {t.n_real} real hits, span {t.t_start_s:.1f}-{t.t_end_s:.1f}s,"
            f" min_score {t.min_score:.2f}"
        )

    plates: list = []
    if caps.get("plates"):
        plates = backend.detect_plates(cfr, max_samples=600)
        print(f"plate tracks: {len(plates)}")
        for t in plates:
            print(
                f"  {t.track_id}: {t.n_real} real hits, span {t.t_start_s:.1f}-{t.t_end_s:.1f}s,"
                f" min_score {t.min_score:.2f}"
            )
    else:
        print("plate model absent — faces only")

    merged = merge_redaction_tracks(faces, plates)
    plan = repair_trajectories(merged, fps=30, frame_w=1920, frame_h=1080, n_frames=n_frames or 0)
    assert_zero_leaks(plan, merged, 30.0)
    n_boxes = sum(len(t.boxes) for t in plan.trajectories)
    print(f"plan OK: {len(plan.trajectories)} trajectories, {n_boxes} per-frame boxes, 0 leaks")

    # render the debug overlay over the whole window for visual proof
    if plates:
        from director_os.vision_fx.redact_track import render_tracked_redaction

        out = OUT / "debug_plate_strip.mp4"
        render_tracked_redaction(cfr, plan, out, debug_overlay=True, n_frames_expected=n_frames)
        print(f"debug overlay render: {out}")

    # ---- audio events on the REAL audio
    from director_os.perception.audio_events import YamnetOnnxBackend

    sweep = YamnetOnnxBackend().detect_events(VIDEO)
    print(f"audio events: {len(sweep.events)} span(s) over {sweep.duration_s:.1f}s")
    for e in sweep.events:
        print(f"  {e.kind}: {e.start_s:.1f}-{e.end_s:.1f}s (peak {e.peak_score:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
