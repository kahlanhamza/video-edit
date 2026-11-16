"""Guardrail 1 (PLAN §9.1): -ar/-ac audio-law flags AFTER loudnorm on every segment."""

from __future__ import annotations

import pytest
from director_os.core.ffmpeg_args import (
    AudioLawArgError,
    assert_audio_law,
    loudnorm_chain,
)
from director_os.otio_engine.renderer import build_segment_cmd
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentRenderSpec,
    VideoRenderSpec,
)


def _raw_spec() -> SegmentRenderSpec:
    return SegmentRenderSpec(
        kind="raw",
        asset_id="a1",
        source_start_s=0.0,
        source_dur_s=5.0,
        video=VideoRenderSpec(w=320, h=180, fps=30),
        audio=AudioRenderSpec(),
        audio_filters=[loudnorm_chain(-16.0)],
    )


def test_loudnorm_precedes_audio_law_flags():
    cmd = build_segment_cmd(_raw_spec(), "/tmp/src.mp4", "/tmp/out.mp4", src_has_audio=True)
    assert_audio_law(cmd)  # must not raise
    af_idx = cmd.index("-af")
    ar_idx = cmd.index("-ar")
    ac_idx = cmd.index("-ac")
    assert af_idx < ar_idx and af_idx < ac_idx, "guardrail 1 violated in generated cmd"
    assert "loudnorm" in cmd[af_idx + 1]
    ar = cmd[ar_idx + 1]
    ac = cmd[ac_idx + 1]
    assert ar == "48000" and ac == "2"


def test_reversed_flag_order_is_rejected():
    bad = [
        "ffmpeg",
        "-i",
        "x.mp4",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-af",
        loudnorm_chain(-16.0),
        "out.mp4",
    ]
    with pytest.raises(AudioLawArgError):
        assert_audio_law(bad)


def test_missing_law_flags_rejected():
    with pytest.raises(AudioLawArgError):
        assert_audio_law(["ffmpeg", "-i", "x.mp4", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
