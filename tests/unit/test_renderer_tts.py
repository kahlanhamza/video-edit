"""Narration render path — the Colab rule: narrator blocks are AUDIBLE.

Upstream wires TTS at the PIPELINE level (stage_narration synthesizes
content-addressed clips; compile_timeline attaches them as
`narration_audio_path`). This file pins that contract on the merged tree:
the TTS stem becomes the segment's audio input, the b-roll stays muted and
slightly darkened (NARRATION_DIM), and the audio law holds on the block.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from director_os.otio_engine.filters import NARRATION_DIM, plain_video_chain
from director_os.otio_engine.renderer import build_segment_cmd
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentRenderSpec,
    VideoRenderSpec,
)


def _narration_spec(
    text: str = "What started as a routine stop was about to unravel.",
) -> SegmentRenderSpec:
    return SegmentRenderSpec(
        kind="narration",
        source_start_s=120.0,
        source_dur_s=12.0,
        video=VideoRenderSpec(w=640, h=360, fps=30, preset="ultrafast", crf=34),
        audio=AudioRenderSpec(),
        narration_text=text,
    )


def test_narration_tts_stem_becomes_the_audio_input(tmp_path: Path):
    spec = _narration_spec()
    tts = tmp_path / "tts_clip.m4a"
    tts.write_bytes(b"fake stem")
    spec_with_stem = spec.model_copy(update={"narration_audio_path": str(tts)})
    cmd = build_segment_cmd(spec_with_stem, "/raw/bodycam.mp4", tmp_path / "out.mp4")
    joined = " ".join(cmd)
    assert str(tts) in joined, "persisted TTS clip rides as an input"
    assert "-map 1:a" in joined, "audio maps from the TTS stem, not the b-roll"
    assert "anullsrc" not in joined, "the silent bed is replaced by the voiceover"
    assert "-ar 48000" in joined and "-ac 2" in joined, "audio law flags hold"


def test_narration_without_stem_keeps_the_silent_bed(tmp_path: Path):
    spec = _narration_spec()
    cmd = build_segment_cmd(spec, "/raw/bodycam.mp4", tmp_path / "out.mp4")
    joined = " ".join(cmd)
    assert "anullsrc" in joined, "no persisted clip yet -> silent 48k bed"
    assert "-map 1:a" in joined


def test_narration_black_canvas_when_no_broll(tmp_path: Path):
    spec = _narration_spec()
    cmd = build_segment_cmd(spec, None, tmp_path / "out.mp4")
    assert "color=c=black:size=640x360" in " ".join(cmd)


def test_narration_dim_rides_video_chain():
    spec = _narration_spec()
    assert plain_video_chain(spec).startswith(NARRATION_DIM), "b-roll slightly darkened"
    raw = spec.model_copy(update={"kind": "raw", "narration_text": ""})
    assert not plain_video_chain(raw).startswith(NARRATION_DIM)


def test_real_tts_stem_renders_a_playable_segment(tmp_path):
    """Live edge-tts round trip through the pipeline-level contract."""
    pytest.importorskip("edge_tts")
    from director_os.audio_engine.tts import generate_tts

    try:
        stem = generate_tts(
            "What started as a routine stop was about to unravel.",
            tmp_path / "stem.m4a",
            voice="en-US-BrianMultilingualNeural",
        )
    except Exception as e:  # pragma: no cover - network dependent
        pytest.skip(f"edge-tts unreachable: {e}")
        return
    spec = _narration_spec().model_copy(update={"narration_audio_path": str(stem.path)})
    cmd = build_segment_cmd(spec, None, tmp_path / "narr.mp4")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    from director_os.core.qa_probe import assert_audio_law, probe

    info = probe(tmp_path / "narr.mp4")
    assert_audio_law(info)  # 48 kHz stereo on the narration block
    assert abs(info.duration_s - 12.0) < 0.6
