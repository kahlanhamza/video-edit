"""Shared fixtures: synthetic bodycam-like fixture video (FFmpeg-generated)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_settings_overlay():
    """Isolate the runtime credential overlay (ADR-0012) between tests.

    The overlay is a process-global in core.config; without this reset a test
    that saves credentials would leak them into every later test.
    """
    from director_os.core.config import set_settings_overrides

    set_settings_overrides({})
    yield
    set_settings_overrides({})


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory) -> Path:
    """12s colorful test-source clip with a 48 kHz stereo audio track."""
    out = tmp_path_factory.mktemp("media") / "fixture_bodycam.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30:duration=12",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=12",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "30",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


@pytest.fixture(scope="session")
def moving_box_video(tmp_path_factory) -> Path:
    """4s clip with ONE high-contrast square gliding across noisy static.

    Deterministic (seeded): the box starts at (40, 60) and moves (+2, +1) px
    per frame for 120 frames at 30 fps on a 320x240 canvas — the box stays
    fully in frame for the WHOLE clip (x_max 40+2*119+28=306, y_max
    60+1*119+28=207), so the ground truth path the selection tests assert
    against is always visible. The noisy background forces the tracker to
    match the box's texture, not a flat region.
    """
    import numpy as np

    out = tmp_path_factory.mktemp("media") / "moving_box.mp4"
    w, h, fps, n = 320, 240, 30, 120
    rng = np.random.default_rng(42)
    bg = rng.integers(60, 140, size=(h, w), dtype=np.uint8)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for i in range(n):
        frame = bg.copy()
        x = 40 + 2 * i
        y = 60 + i
        frame[y : y + 28, x : x + 28] = 235  # bright square, 28x28
        frame[y + 4 : y + 24, x + 4 : x + 24] = 30  # dark core (distinct texture)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait(timeout=60)
    return out


@pytest.fixture(scope="session")
def flat_video(tmp_path_factory) -> Path:
    """1s clip of UNIFORM gray — the no-object control for refusal tests."""
    out = tmp_path_factory.mktemp("media") / "flat.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:d=1:r=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out
