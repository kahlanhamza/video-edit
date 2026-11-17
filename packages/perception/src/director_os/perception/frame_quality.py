"""Frame quality pre-filter (PLAN §4.5, M4) — the free CPU tier of the vision stack.

Deterministic OpenCV scoring flags footage the Director should never spend
tokens (or screen time) on: pitch-black frames (vest/arm over the lens), severe
motion blur, and blackouts. Runs BEFORE any paid VLM call so the cheap
contact-sheet tier only sees frames worth describing, and the planner gets a
`unusable_spans` map it can cut around.

Two scores per sampled frame:
- brightness  = mean luma (0..255)            — dark below `dark_luma`
- sharpness   = Laplacian variance            — blurry below `blur_var`

Hermetic, deterministic on synthetic inputs, pinned by tests.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from director_os.core.qa_probe import probe

DEFAULT_DARK_LUMA = 18.0
DEFAULT_BLUR_VAR = 35.0


class FrameQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameScore:
    """One sampled frame's quality verdict."""

    t_s: float
    brightness: float  # mean luma 0..255
    sharpness: float  # Laplacian variance (higher = sharper)
    usable: bool
    reason: str = "ok"  # ok | dark | blurry | dark+blurry


@dataclass(frozen=True)
class FrameQualityReport:
    source: str
    sample_every_s: float
    frames: list[FrameScore] = field(default_factory=list)

    @property
    def usable_ratio(self) -> float:
        if not self.frames:
            return 0.0
        return sum(1 for f in self.frames if f.usable) / len(self.frames)

    @property
    def unusable_spans(self) -> list[tuple[float, float]]:
        """Consecutive unusable samples merged into (start_s, end_s) spans."""
        spans: list[tuple[float, float]] = []
        cursor: tuple[float, float] | None = None
        stride = self.sample_every_s
        for f in self.frames:
            if f.usable:
                if cursor is not None:
                    spans.append(cursor)
                    cursor = None
                continue
            cursor = (
                (f.t_s, min(f.t_s + stride, _duration(self.source)))
                if cursor is None
                else (cursor[0], f.t_s + stride)
            )
        if cursor is not None:
            spans.append(cursor)
        return spans


def _duration(path: str) -> float:
    try:
        return probe(path).duration_s
    except Exception:  # pragma: no cover — probe already validated the source upstream
        return 0.0


def score_frame_quality(
    video_path: str,
    sample_every_s: float = 1.0,
    max_samples: int = 4000,
    dark_luma: float = DEFAULT_DARK_LUMA,
    blur_var: float = DEFAULT_BLUR_VAR,
) -> FrameQualityReport:
    """Sample the video on a stride and score brightness + sharpness per frame."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FrameQualityError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * sample_every_s)))
    frames: list[FrameScore] = []
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0 and len(frames) < max_samples:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    idx += 1
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                dark = brightness < dark_luma
                blurry = sharpness < blur_var
                reason = (
                    "dark+blurry"
                    if dark and blurry
                    else "dark"
                    if dark
                    else "blurry"
                    if blurry
                    else "ok"
                )
                frames.append(
                    FrameScore(
                        t_s=round(idx / fps, 3),
                        brightness=round(brightness, 2),
                        sharpness=round(sharpness, 2),
                        usable=not (dark or blurry),
                        reason=reason,
                    )
                )
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise FrameQualityError(f"no frames sampled from {video_path}")
    return FrameQualityReport(source=str(video_path), sample_every_s=sample_every_s, frames=frames)


def black_frame_probe(video_path: str, threshold: float = 0.98) -> list[tuple[float, float]]:
    """FFmpeg `blackdetect` pass — whole-scene blackouts (deterministic, cheap).

    Complements per-frame scoring: catches complete camera-obstruction intervals
    even when a few stray frames between them score as usable.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_path,
        "-vf",
        f"blackdetect=d=0.5:pix_th={threshold:.2f}",
        "-an",
        "-f",
        "null",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise FrameQualityError(f"blackdetect failed: {r.stderr[-400:]}")
    import re

    out: list[tuple[float, float]] = []
    for line in r.stderr.splitlines():
        m = re.search(r"black_start:([0-9]+\.?[0-9]*)\s+black_end:([0-9]+\.?[0-9]*)", line)
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out
