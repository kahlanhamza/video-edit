"""Guardrail 2 (PLAN §9.2): exact slicing is -ss <start> -t <dur>, never -to."""

from __future__ import annotations

import pytest
from director_os.core.ffmpeg_args import exact_slice_args
from director_os.otio_engine.renderer import build_segment_cmd
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentRenderSpec,
    VideoRenderSpec,
)


def test_exact_duration_computed():
    args = exact_slice_args(10.0, 25.5)
    assert args == ["-ss", "10.000000", "-t", "15.500000"]


def test_zero_or_negative_duration_rejected():
    with pytest.raises(ValueError):
        exact_slice_args(5.0, 5.0)
    with pytest.raises(ValueError):
        exact_slice_args(6.0, 2.0)


def test_segment_cmd_never_uses_to():
    spec = SegmentRenderSpec(
        kind="raw",
        asset_id="a1",
        source_start_s=7.25,
        source_dur_s=3.75,
        video=VideoRenderSpec(w=320, h=180, fps=30),
        audio=AudioRenderSpec(),
    )
    cmd = build_segment_cmd(spec, "/tmp/src.mp4", "/tmp/out.mp4", src_has_audio=True)
    assert "-to" not in cmd, "guardrail 2: -to after -i must never appear"
    ss = cmd.index("-ss")
    t = cmd.index("-t")
    i = cmd.index("-i")
    assert ss < t < i, "input options must be `-ss <start> -t <dur>` before -i"
    assert cmd[t + 1] == "3.750000"
    # a second -t caps the output duration at the same exact value
    assert cmd.count("-t") == 2
    assert cmd[cmd.index("-t", t + 1) + 1] == "3.750000"
