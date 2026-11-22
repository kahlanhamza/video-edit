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
