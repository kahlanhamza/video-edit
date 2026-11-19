"""Automatic watermark detection — intermittent, moving overlays (any video)."""

from pathlib import Path

import cv2
import numpy as np
from director_os.vision_fx.watermark import (
    WatermarkFinding,
    detect_watermarks,
    findings_to_trajectories,
)


def _make_video(tmp_path: Path, *, with_mark_spans, w=320, h=180, fps=10, dur_s=6):
    """Synthetic clip: smooth moving background (dHash-stable, like real
    footage) + a static white text overlay present ONLY in `with_mark_spans`.
    The background PATTERN changes after 2 s so the mark-on and mark-off
    halves land in different shot clusters."""
    path = tmp_path / "wm_test.mp4"
    n = int(dur_s * fps)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    mark = np.zeros((h, w), np.uint8)
    cv2.putText(
        mark,
        "Police Bodycam Videos",
        (60, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        255,
        2,
        cv2.LINE_AA,
    )
    for i in range(n):
        t = i / fps
        if t < 2.0:  # span A: horizontal waves
            base = 90 + 50 * np.sin((xs + i * 4) * 0.06)
        else:  # span B: vertical waves (different dHash)
            base = 90 + 50 * np.sin((ys + i * 4) * 0.06)
        frame = np.clip(base, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)
        cv2.circle(frame, ((i * 7) % w, (i * 3) % h), 25, (200, 180, 160), -1)
        if any(a <= t <= b for a, b in with_mark_spans):
            frame[mark > 0] = 255
        writer.write(frame)
    writer.release()
    return path


class _FakeVision:
    """Returns one watermark box for every asked frame (like Gemini would)."""

    def __init__(self, box):
        self.box = box
        self.calls = 0

    def analyze(self, question, frames):
        import json

        self.calls += 1
        from director_os.perception.vision import SceneInsight

        return [
            SceneInsight(
                timestamp_s=f.timestamp_s,
                kind="redaction_candidate",
                description=json.dumps(
                    {
                        "label": "Police Bodycam Videos",
                        "box_xywh": list(self.box),
                        "overlay_kind": "watermark",
                    }
                ),
                confidence=0.9,
            )
            for f in frames
        ]


def test_intermittent_mark_found_in_its_spans(tmp_path):
    video = _make_video(tmp_path, with_mark_spans=[(0.0, 2.0)])
    fake = _FakeVision((60, 150, 180, 32))
    report = detect_watermarks(video, vision_client=fake, max_samples=24)
    assert fake.calls > 0, "vision client never consulted"
    assert report.findings, "no watermark findings"
    f = report.findings[0]
    assert f.source == "vision"
    assert f.t_start_s < 2.5 and f.t_end_s < 3.0, (
        f"mark span leaked outside its clip: {f.t_start_s}..{f.t_end_s}"
    )


def test_timestamp_overlay_not_blurred(tmp_path):
    video = _make_video(tmp_path, with_mark_spans=[(0.0, 6.0)])

    class _FakeTimestamp:
        def analyze(self, question, frames):
            import json

            from director_os.perception.vision import SceneInsight

            return [
                SceneInsight(
                    timestamp_s=f.timestamp_s,
                    kind="redaction_candidate",
                    description=json.dumps(
                        {
                            "label": "AXON clock",
                            "box_xywh": [10, 10, 120, 24],
                            "overlay_kind": "timestamp",
                        }
                    ),
                    confidence=0.95,
                )
                for f in frames
            ]

    report = detect_watermarks(video, vision_client=_FakeTimestamp(), max_samples=16)
    assert report.findings == [], "timestamp burns must stay visible"


def test_local_fallback_without_vision_client(tmp_path):
    video = _make_video(tmp_path, with_mark_spans=[(0.0, 6.0)])
    report = detect_watermarks(video, vision_client=None, max_samples=24)
    assert report.findings, "local heuristic found nothing on a static overlay"
    f = report.findings[0]
    assert f.source == "local"
    # the box must actually sit on the mark row (bottom third)
    assert f.box[1] > 180 * 0.5, f"box not on the mark: {f.box}"


def test_trajectories_are_constant_and_dense(tmp_path):
    fps = 10.0
    finding = WatermarkFinding(
        label="mark",
        kind="watermark",
        box=(10, 140, 200, 30),
        t_start_s=1.2,
        t_end_s=4.8,
        confidence=0.9,
        source="vision",
    )
    trajs = findings_to_trajectories([finding], fps)
    assert len(trajs) == 1
    boxes = trajs[0].boxes
    idx = [b.frame_idx for b in boxes]
    assert idx == list(range(idx[0], idx[-1] + 1)), "not dense (flicker!)"
    assert all((b.x, b.y, b.w, b.h) == (10, 140, 200, 30) for b in boxes), "box moved"


def test_no_mark_no_findings(tmp_path):
    video = _make_video(tmp_path, with_mark_spans=[])
    detect_watermarks(video, vision_client=_FakeVision((1, 2, 3, 4)), max_samples=16)
    # the fake client would claim a mark anywhere — but only clusters that
    # exist get asked; a pure-noise video is ONE cluster, so at most that.
    # The real contract: without any overlay the LOCAL pass must find nothing.
    local = detect_watermarks(video, vision_client=None, max_samples=16)
    assert local.findings == []
