"""Burned-caption guard — YouTube tapes sometimes arrive with captions ON.

A source that already has burned-in captions collides with our own v6
caption block (two bottom-center lanes): double text, unreadable frames.
The guard samples frames across the tape, crops the bottom caption band and
measures text-like edge energy — a deterministic, dependency-free check that
flags the source BEFORE render, so the tape can be re-sourced or the
caption lane shifted instead of shipping a double-captioned master.

Metric (per sampled frame): the bottom band is divided into columns; real
caption text produces tall high-contrast vertical strokes that survive
column averaging, while natural footage (ground, pavement, car seats) yields
low, diffuse energy. ``score`` is the fraction of band pixels whose local
gradient exceeds the strong-edge threshold; ``flagged`` fires when the
median frame score crosses ``FLAG_THRESHOLD``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

BAND_FRAC = 0.25  # bottom quarter of the frame — where captions live
SAMPLE_FRAMES = 8
EDGE_THRESHOLD = 72  # luminance gradient treated as a strong text edge
# calibrated on synthetic extremes: clean footage bands score 0.000-0.004,
# 26-28px caption text scores 0.016-0.032 — the flag line sits between
FLAG_THRESHOLD = 0.012


def _sample_timestamps(duration_s: float, n: int = SAMPLE_FRAMES) -> list[float]:
    """Spread n samples across the middle 90% (skip intro/outro black)."""
    span = max(0.0, duration_s * 0.9)
    offset = duration_s * 0.05
    if n <= 1:
        return [offset + span / 2]
    return [round(offset + span * i / (n - 1), 2) for i in range(n)]


def bottom_band_text_score(frame_png: Path) -> float:
    """Fraction of bottom-band pixels with a strong horizontal neighbor
    gradient (text strokes); deterministic PIL metric."""
    import numpy as np  # noqa: I001 — isort grouping under the function scope
    from PIL import Image

    im = Image.open(frame_png).convert("L")
    arr = np.asarray(im, dtype=np.float32)
    h, w = arr.shape
    band = arr[int(h * (1.0 - BAND_FRAC)) :, :]
    gx = np.abs(np.diff(band, axis=1))
    return float((gx > EDGE_THRESHOLD).mean())


def _extract_frame(video: Path, t_s: float, dest: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{t_s:.2f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


def scan_burned_captions(
    video: Path,
    duration_s: float,
    *,
    n_samples: int = SAMPLE_FRAMES,
) -> dict:
    """Sample the tape, score each bottom band, and flag caption collisions.

    Returns the honest report dict the production report embeds:
    ``{"samples": [...], "median_score": x, "flagged": bool, "threshold": t}``
    """
    scores: list[float] = []
    with tempfile.TemporaryDirectory(prefix="capguard_") as td:
        for i, t in enumerate(_sample_timestamps(duration_s, n_samples)):
            frame = Path(td) / f"frame_{i:02d}_{int(t)}s.png"
            _extract_frame(video, t, frame)
            scores.append(round(bottom_band_text_score(frame), 4))
    med = float(sorted(scores)[len(scores) // 2]) if scores else 0.0
    return {
        "samples": scores,
        "median_score": round(med, 4),
        "threshold": FLAG_THRESHOLD,
        "flagged": bool(scores) and med >= FLAG_THRESHOLD,
    }


def report_json(report: dict) -> str:
    return json.dumps(report, indent=2)
