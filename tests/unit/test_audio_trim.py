"""A1 audio trim — the audio track's own trim, twin of `timeline.trim_clip`.

The VO window is metadata on the narration clip: trimming re-cuts the
voiceover itself (dead air, false starts, tails) while b-roll re-cycles
underneath. Pinned across the whole path: contract law -> tool -> segment
cache hash -> ffmpeg command -> retime block sizing.
"""

from __future__ import annotations

import json

import opentimelineio.opentime as ot_time
import opentimelineio.schema as ot_schema
import pytest
from director_os.agent_runtime.editing_tools import DirectorToolbox, ToolOperationError
from director_os.otio_engine.contract import (
    VO_MIN_WINDOW_S,
    NarratorClipMeta,
    attach,
)
from director_os.otio_engine.renderer import build_segment_cmd, narration_trim_chain
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentRenderSpec,
    VideoRenderSpec,
)
from pydantic import ValidationError

PROJECT = "proj_a1_trim"


def _narration_timeline(tts_dur: float = 8.0) -> tuple[ot_schema.Timeline, ot_schema.Clip]:
    tl = ot_schema.Timeline()
    track = ot_schema.Track(name="V1_master")
    tl.tracks.append(track)
    clip = ot_schema.Clip(name="narration_0")
    clip.source_range = ot_time.TimeRange(
        start_time=ot_time.RationalTime(12.0 * 30, 30),
        duration=ot_time.RationalTime(tts_dur * 30, 30),
    )
    attach(clip, NarratorClipMeta(script_text="Officer Brent asked for the registration twice."))
    track.append(clip)
    return tl, clip


# ------------------------------------------------------------------ contract


def test_vo_window_defaults_to_the_full_stem():
    m = NarratorClipMeta(script_text="x")
    assert m.vo_in_s == 0.0 and m.vo_out_s is None
    assert m.vo_window(8.4) == (0.0, 8.4), "untrimmed block rides the whole TTS clip"


def test_vo_window_clamps_a_stale_trim_to_a_shorter_stem():
    m = NarratorClipMeta(script_text="x", vo_in_s=1.2, vo_out_s=4.0)
    # the stem was re-synthesized shorter than the stale trim
    assert m.vo_window(2.0) == (1.2, 2.0), "out clamped to the stem — never overruns"


def test_vo_window_rejects_inverted_and_too_short_windows():
    with pytest.raises(Exception, match="must be > vo_in_s"):
        NarratorClipMeta(script_text="x", vo_in_s=3.0, vo_out_s=1.0)
    with pytest.raises(Exception, match=f"shorter than the {VO_MIN_WINDOW_S}s"):
        NarratorClipMeta(script_text="x", vo_in_s=0.0, vo_out_s=0.2)
    # exactly the minimum is legal
    m = NarratorClipMeta(script_text="x", vo_in_s=1.0, vo_out_s=1.0 + VO_MIN_WINDOW_S)
    assert m.vo_out_s == 1.5


# ------------------------------------------------------------------ tool


def test_trim_audio_sets_and_releases_the_window():
    tl, _clip = _narration_timeline()
    tb = DirectorToolbox(tl, PROJECT)
    r = tb.trim_audio(0, vo_in_s=1.25, vo_out_s=6.75)
    assert r["ok"] and "[1.25, 6.75]" in r["detail"], "window reported"
    meta = tl.tracks[0][0].metadata["director_os"]
    assert meta["vo_in_s"] == 1.25 and meta["vo_out_s"] == 6.75
    # release: out rides to the stem end again
    r2 = tb.trim_audio(0, vo_in_s=0.0, vo_out_s=None)
    assert r2["ok"]
    meta2 = tl.tracks[0][0].metadata["director_os"]
    assert meta2["vo_in_s"] == 0.0 and meta2["vo_out_s"] is None


def test_trim_audio_rejects_non_narration_and_bad_windows():
    tl, clip = _narration_timeline()
    raw = ot_schema.Clip(name="raw_0")
    raw.source_range = ot_time.TimeRange(
        start_time=ot_time.RationalTime(0, 30),
        duration=ot_time.RationalTime(90, 30),
    )
    raw.metadata.update({"director_os": {"kind": "raw", "asset_id": "a1"}})
    tl.tracks[0].append(raw)
    tb = DirectorToolbox(tl, PROJECT)
    with pytest.raises(ToolOperationError, match="not a narration clip"):
        tb.trim_audio(1, vo_in_s=1.0)
    with pytest.raises(ToolOperationError, match="vo_out_s must be > 0"):
        tb.trim_audio(0, vo_in_s=0.0, vo_out_s=0.0)
    # contract validation (pydantic ValidationError) fires before anything is written
    with pytest.raises(ValidationError, match="must be > vo_in_s"):
        tb.trim_audio(0, vo_in_s=4.0, vo_out_s=2.0)


# ------------------------------------------------------------------ cache + cmd


def _spec(vo_in: float = 0.0, vo_out: float | None = None) -> SegmentRenderSpec:
    return SegmentRenderSpec(
        kind="narration",
        source_start_s=120.0,
        source_dur_s=12.0,
        video=VideoRenderSpec(w=640, h=360, fps=30, preset="ultrafast", crf=34),
        audio=AudioRenderSpec(),
        narration_text="Officer Brent asked for the registration twice.",
        vo_in_s=vo_in,
        vo_out_s=vo_out,
    )


def test_vo_window_rides_the_segment_cache_key():
    from director_os.otio_engine.segment_cache import segment_hash

    base = segment_hash(_spec())
    trimmed = segment_hash(_spec(vo_in=1.2, vo_out=5.0))
    different_trim = segment_hash(_spec(vo_in=2.2, vo_out=5.0))
    assert base != trimmed, "a trim MUST re-render the block"
    assert trimmed != different_trim, "different windows are different segments"


def test_trimmed_vo_adds_sample_exact_atrim_chain(tmp_path):
    stem = tmp_path / "tts.m4a"
    stem.write_bytes(b"fake")
    spec = _spec(vo_in=1.2, vo_out=5.0).model_copy(update={"narration_audio_path": str(stem)})
    cmd = build_segment_cmd(spec, "/raw/bodycam.mp4", tmp_path / "out.mp4")
    joined = " ".join(cmd)
    af = cmd[cmd.index("-af") + 1]
    assert "atrim=start=1.200000:end=5.000000" in af, "sample-exact window edges"
    assert "asetpts=N/SR/TB" in af, "timestamps re-based to 0"
    assert af.index("-af") < joined.index("-ar") if False else True  # guardrail 1 shape
    cmd_i = cmd.index("-af")
    assert cmd_i < cmd.index("-ar") and cmd_i < cmd.index("-ac"), "audio law order holds"
    assert "afade=t=in:st=0:d=0.01" in af and "afade=t=out:st=3.790000:d=0.01" in af, (
        "anti-click fades at the trimmed edges (out fade lands on the AUDIO tail)"
    )


def test_untrimmed_vo_has_no_atrim(tmp_path):
    stem = tmp_path / "tts.m4a"
    stem.write_bytes(b"fake")
    spec = _spec().model_copy(update={"narration_audio_path": str(stem)})
    cmd = build_segment_cmd(spec, "/raw/bodycam.mp4", tmp_path / "out.mp4")
    assert "-af" not in cmd, "full-stem blocks keep the zero-filter path"
    assert narration_trim_chain(_spec(), 12.0), "chain builder still total"


def test_atrim_only_in_point_keeps_natural_tail(tmp_path):
    stem = tmp_path / "tts.m4a"
    stem.write_bytes(b"fake")
    spec = _spec(vo_in=2.0).model_copy(update={"narration_audio_path": str(stem)})
    cmd = build_segment_cmd(spec, "/raw/bodycam.mp4", tmp_path / "out.mp4")
    af = cmd[cmd.index("-af") + 1]
    assert "atrim=start=2.000000" in af and "end=" not in af.split("atrim")[1][:40], (
        "in-only trim skips the head, rides the stem to its natural end"
    )


# ------------------------------------------------------------------ retime


def _built_narration_timeline(vo_in: float = 0.0, vo_out: float | None = None):
    """A one-item timeline via the production builder so metadata is exact."""
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import item_duration_s

    tl, _clip = _narration_timeline(tts_dur=10.0)
    meta = dict(tl.tracks[0][0].metadata["director_os"])
    meta["vo_in_s"] = vo_in
    meta["vo_out_s"] = vo_out
    tl.tracks[0][0].metadata["director_os"] = meta
    return tl, master_track(tl), item_duration_s


def test_retime_sizes_trimmed_blocks_to_the_vo_window():
    from director_os.worker.pipeline import retime_narration_to_tts

    tl, master, item_duration_s = _built_narration_timeline(vo_in=1.0, vo_out=4.0)
    clips = [
        {
            "script_text": "Officer Brent asked for the registration twice.",
            "duration_s": 8.0,
            "clip": "/tmp/x.mp3",
        }
    ]
    assert retime_narration_to_tts(tl, clips) == 1
    # window 3.0s + 0.35 pad — NOT the full 8s stem + pad
    assert abs(item_duration_s(master[0]) - 3.35) < 0.01


def test_retime_keeps_the_untrimmed_floor():
    from director_os.worker.pipeline import retime_narration_to_tts

    tl, master, item_duration_s = _built_narration_timeline()
    clips = [
        {
            "script_text": "Officer Brent asked for the registration twice.",
            "duration_s": 1.0,
            "clip": "/tmp/x.mp3",
        }
    ]
    retime_narration_to_tts(tl, clips)
    assert abs(item_duration_s(master[0]) - 2.0) < 0.01, "2s floor rescues short untrimmed blocks"


# ------------------------------------------------------------------ snapshot


def test_nle_snapshot_exposes_the_vo_window(tmp_path):
    from director_os.api.nle import _clips_from_otio

    tl, _clip = _narration_timeline()
    tl.tracks[0][0].metadata["director_os"]["vo_in_s"] = 2.5
    tl.tracks[0][0].metadata["director_os"]["vo_out_s"] = 7.0
    import opentimelineio.adapters as ot_adapters

    otio_json = json.loads(ot_adapters.write_to_string(tl))
    clips = _clips_from_otio(otio_json, {"Officer Brent asked for the registration twice.": 8.0})
    assert len(clips) == 1
    c = clips[0]
    assert c["vo_in_s"] == 2.5 and c["vo_out_s"] == 7.0
    assert c["tts_duration_s"] == 8.0, "measured stem length rides for the trim math"


def test_nle_snapshot_defaults_for_untrimmed_blocks(tmp_path):
    from director_os.api.nle import _clips_from_otio

    tl, _clip = _narration_timeline()
    import opentimelineio.adapters as ot_adapters

    otio_json = json.loads(ot_adapters.write_to_string(tl))
    clips = _clips_from_otio(otio_json, {})
    c = clips[0]
    assert c["vo_in_s"] == 0.0 and c["vo_out_s"] is None and c["tts_duration_s"] is None
