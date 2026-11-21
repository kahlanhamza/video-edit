"""Hermetic tests for the agent READ surface (inspection_tools).

No network, no ffmpeg, no media: the evidence artifacts are tiny JSON/ASS
fixtures written into a tmp work dir, and the VAD compute path is monkeypatched
(cache-hit path is tested for real).
"""

from __future__ import annotations

import json
from pathlib import Path

import opentimelineio.opentime as ot_time
import opentimelineio.schema as ot_schema
import pytest
from director_os.agent_runtime import inspection_tools as it
from director_os.agent_runtime.registry import ToolDispatchError, ToolRegistry

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Officer,Roboto,42,&H00FFFFFF,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,58,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:12.50,Officer,,0,0,46,,{\\fad(40,30)}{\\k20}Put{\\k30} the {\\k40}phone down.
Dialogue: 0,0:00:13.00,0:00:15.00,Suspect,,0,0,46,,{\\fad(40,30)}{\\i1\\c&H9C9C9C&\\2c&H9C9C9C&\\i0}*unaudible*
Dialogue: 0,0:00:16.00,0:00:17.40,Narrator,,0,0,46,,{\\fad(40,30)}It was 6:30 AM.
"""


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    d = tmp_path / "v3"
    d.mkdir()
    (d / "caption_evidence.json").write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "role": "Officer",
                    "has_words": True,
                    "s": 10.0,
                    "e": 12.5,
                    "n_words": 4,
                    "text": "Put the phone down.",
                    "speech_ratio": 0.94,
                    "rms_db": -21.0,
                    "voice_band": 0.62,
                    "stretch": 0.35,
                    "repeat_n": 1,
                    "occupancy": 0.9,
                },
                {
                    "id": 1,
                    "role": "Suspect",
                    "has_words": True,
                    "s": 13.0,
                    "e": 15.0,
                    "n_words": 2,
                    "text": "people want to stop",
                    "speech_ratio": 0.11,
                    "rms_db": -38.0,
                    "voice_band": 0.31,
                    "stretch": 0.9,
                    "repeat_n": 4,
                    "occupancy": 0.5,
                },
            ]
        )
    )
    (d / "subtitles_v4.ass").write_text(ASS)
    (d / "qa_v4.json").write_text(
        json.dumps([{"name": "duration", "ok": True, "detail": "final 657.7s"}])
    )
    (d / "master_postfx_v4.mp4").write_bytes(b"\x00" * 1024)
    return d


@pytest.fixture()
def reg(work: Path) -> ToolRegistry:
    r = ToolRegistry()
    it.REG_TIMELINE.clear()
    it.register_inspection_tools(r, work)
    return r


# ------------------------------------------------------------ describe/qa
def test_describe_inventory(reg: ToolRegistry, work: Path) -> None:
    d = reg.dispatch("project.describe", {})
    assert d["work_dir"] == str(work)
    assert d["master_postfx_v4.mp4"]["exists"] is True
    assert d["master_postfx_v4.mp4"]["duration_s"] is None  # no real media
    assert d["subtitles_v4.ass"]["cues"] == 3
    assert d["subtitles_v4.ass"]["tag_cues"] == 1
    assert d["qa_v4.json"]["exists"] is True


def test_qa_report_green(reg: ToolRegistry) -> None:
    r = reg.dispatch("qa.read_report", {})
    assert r["qa_v4"]["green"] is True
    assert r["qa_v4"]["n"] == 1


def test_qa_missing_is_tool_error(reg: ToolRegistry, work: Path) -> None:
    (work / "qa_v4.json").unlink()
    with pytest.raises(it.ToolOperationError):
        reg.dispatch("qa.read_report", {})


# ------------------------------------------------------------- transcript
def test_transcript_window_filters_and_carries_evidence(reg: ToolRegistry) -> None:
    out = reg.dispatch("transcript.read_window", {"start_s": 12.0, "end_s": 16.0})
    assert out["n_events"] == 2
    texts = [e["text"] for e in out["events"]]
    assert texts[0] == "Put the phone down."
    ev = out["events"][1]
    assert ev["speech_ratio"] == 0.11 and ev["repeat_n"] == 4


def test_transcript_window_validation(reg: ToolRegistry) -> None:
    with pytest.raises(ToolDispatchError):  # end <= start
        reg.dispatch("transcript.read_window", {"start_s": 10, "end_s": 10})
    out = reg.dispatch("transcript.read_window", {"start_s": 0.0, "end_s": 500.0})
    assert out["window"][1] == 120.0  # clamp


def test_transcript_missing_artifact(reg: ToolRegistry, work: Path) -> None:
    (work / "caption_evidence.json").unlink()
    with pytest.raises(it.ToolOperationError):
        reg.dispatch("transcript.read_window", {"start_s": 0, "end_s": 5})


# ---------------------------------------------------------------- captions
def test_cues_parse_overrides_and_tags(reg: ToolRegistry) -> None:
    out = reg.dispatch("captions.list_cues", {"start_s": 9.0, "end_s": 18.0})
    assert out["n_cues"] == 3
    assert out["cues"][0]["text"] == "Put the phone down."  # tags stripped
    assert out["cues"][1]["text"] == "*unaudible*"
    assert out["cues"][1]["style"] == "Suspect"
    assert out["coverage_s"] == pytest.approx(5.9, abs=0.01)


def test_cues_window_excludes(reg: ToolRegistry) -> None:
    out = reg.dispatch("captions.list_cues", {"start_s": 0.0, "end_s": 9.0})
    assert out["n_cues"] == 0 and out["coverage_s"] == 0.0


# ------------------------------------------------------------------- audio
def test_speech_spans_cache_hit(reg: ToolRegistry, work: Path) -> None:
    (work / "vad_spans_cache.json").write_text(
        json.dumps(
            {"frame_s": 0.03, "sr": 16000, "spans": [[10.1, 12.4], [13.2, 15.1], [40.0, 44.0]]}
        )
    )
    out = reg.dispatch("audio.speech_spans", {"start_s": 9.0, "end_s": 20.0})
    assert out["n_spans"] == 2
    assert out["speech_s"] == pytest.approx(4.2, abs=0.01)


def test_speech_spans_compute_path(reg: ToolRegistry, work: Path, monkeypatch) -> None:
    calls = []

    def fake_load_pcm(_p):
        calls.append("pcm")
        return object()

    def fake_flags(_pcm):
        return [True] * 100, [0.1] * 100, 480

    def fake_spans(_f, _fr, min_s=0.45, join_gap=0.30):
        return [(10.0, 12.0)]

    import sys

    fake = type(sys)("vad_evidence")
    fake.SR = 16000
    fake.load_pcm = fake_load_pcm
    fake.vad_flags = fake_flags
    fake.speech_spans = fake_spans
    monkeypatch.setitem(sys.modules, "vad_evidence", fake)
    out = reg.dispatch("audio.speech_spans", {"start_s": 9.0, "end_s": 20.0})
    assert out["n_spans"] == 1 and out["spans"] == [[10.0, 12.0]]
    # second call must use the cache written by the first (resumable law)
    out2 = reg.dispatch("audio.speech_spans", {"start_s": 9.0, "end_s": 20.0})
    assert out2["spans"] == [[10.0, 12.0]]
    assert len(calls) == 1


def test_speech_spans_no_engine_no_cache(reg: ToolRegistry) -> None:
    with pytest.raises(it.ToolOperationError):
        reg.dispatch("audio.speech_spans", {"start_s": 0.0, "end_s": 5.0})


# --------------------------------------------------------------- proposals
def test_propose_tag_window_appends(reg: ToolRegistry, work: Path) -> None:
    r1 = reg.dispatch(
        "captions.propose_tag_window",
        {
            "start_s": 609.4,
            "end_s": 628.9,
            "tag": "unaudible",
            "reason": "officer murmurs; ASR loop on sr=0.0-0.43",
        },
    )
    assert "proposal #1" in r1["detail"]
    r2 = reg.dispatch(
        "captions.propose_tag_window",
        {"start_s": 340.0, "end_s": 344.0, "tag": "mumbling", "reason": "trails off"},
    )
    assert "proposal #2" in r2["detail"]
    props = json.loads((work / "proposals_v1.json").read_text())
    assert [p["tag"] for p in props] == ["unaudible", "mumbling"]
    assert props[0]["start_s"] == 609.4


def test_propose_tag_rejects_bad_tag(reg: ToolRegistry) -> None:
    with pytest.raises(ToolDispatchError):
        reg.dispatch(
            "captions.propose_tag_window",
            {"start_s": 1.0, "end_s": 2.0, "tag": "made-up", "reason": "not a real tag"},
        )


def test_propose_tag_rejects_short_reason(reg: ToolRegistry) -> None:
    with pytest.raises(ToolDispatchError):
        reg.dispatch(
            "captions.propose_tag_window",
            {"start_s": 1.0, "end_s": 2.0, "tag": "unaudible", "reason": ""},
        )


# ---------------------------------------------------------------- timeline
def _make_timeline() -> ot_schema.Timeline:
    tl = ot_schema.Timeline(name="t")
    master = ot_schema.Track(name="V1_master")
    gap = ot_schema.Gap()
    gap.source_range = ot_time.TimeRange(ot_time.RationalTime(0, 24), ot_time.RationalTime(48, 24))
    clip = ot_schema.Clip(name="c0")
    clip.source_range = ot_time.TimeRange(
        ot_time.RationalTime(30, 24), ot_time.RationalTime(120, 24)
    )
    clip.metadata["director_os"] = {"kind": "raw", "asset_id": "a0", "audio_mode": "loudnorm_48k"}
    master.append(gap)
    master.append(clip)
    tl.tracks.append(master)
    return tl


def test_timeline_list_items(reg: ToolRegistry) -> None:
    it.REG_TIMELINE.clear()
    it.REG_TIMELINE.append(_make_timeline())
    out = reg.dispatch("timeline.list_items", {})
    assert out["n_items"] == 2
    gap, clip = out["items"]
    assert gap["kind"] == "gap" and gap["dur_s"] == 2.0
    assert clip["kind"] == "raw" and clip["source_in_s"] == 1.25
    assert clip["source_out_s"] == 6.25 and clip["start_s"] == 2.0


def test_timeline_unbound(reg: ToolRegistry) -> None:
    it.REG_TIMELINE.clear()
    with pytest.raises(it.ToolOperationError):
        reg.dispatch("timeline.list_items", {})


# ----------------------------------------------------------------- finish
def test_agent_finish_marker(reg: ToolRegistry, work: Path) -> None:
    r = reg.dispatch("agent.finish", {"summary": "audit done, 2 proposals"})
    assert "session complete" in r["detail"]
    m = json.loads((work / "agent_finish.json").read_text())
    assert m["summary"] == "audit done, 2 proposals"


def test_registry_schema_roundtrip(reg: ToolRegistry) -> None:
    schemas = reg.openai_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {
        "project.describe",
        "qa.read_report",
        "transcript.read_window",
        "captions.list_cues",
        "audio.speech_spans",
        "captions.propose_tag_window",
        "timeline.list_items",
        "agent.finish",
    } <= names
    for s in schemas:
        assert s["function"]["parameters"]["type"] == "object"
