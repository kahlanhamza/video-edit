"""Agent workbench — the Director's caption/mutation surface, pinned by tests.

Ports the live-session laws into hermetic CI checks:
  ass_io         byte-stable .ass parsing, Tag style, chronological insertion
  workshop gates evidence-backed tag green; clean-speech tag red; gap fills
  critic         RED names the failing check (no_master_gaps)
  memory         session state survives a fresh registry (atomic, resumable)
"""

from __future__ import annotations

import json

import opentimelineio.opentime as ot_time
import opentimelineio.schema as ot_schema
import pytest
from director_os.agent_runtime.agents import PlannerExecutor, ScriptedBackend, ToolStep
from director_os.agent_runtime.ass_io import (
    ensure_tag_style,
    parse_cues,
    rewrite_cue,
    s_to_ts,
)
from director_os.agent_runtime.critic import register_critic_tools
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
from director_os.agent_runtime.inspection_tools import REG_TIMELINE, register_inspection_tools
from director_os.agent_runtime.session_memory import register_session_memory_tools
from director_os.agent_runtime.workshop_tools import register_workshop_tools
from director_os.core.db import init_models, make_engine, make_session_factory
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    track_by_name,
)

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Officer,Roboto,42,&H00FFFFFF,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,58,1
Style: Suspect,Roboto,42,&H00EDD7C8,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,58,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:12.50,Officer,,0,0,46,,Put the phone down.
Dialogue: 0,0:00:13.00,0:00:15.00,Suspect,,0,0,46,,I am not resisting.
"""

EVIDENCE = [
    {
        "id": 0,
        "role": "Officer",
        "has_words": True,
        "s": 10.0,
        "e": 12.5,
        "n_words": 4,
        "text": "Put the phone down.",
        "speech_ratio": 0.9,
        "rms_db": -18.0,
        "repeat_n": 1,
        "stretch": 0.6,
    },
    {
        "id": 1,
        "role": "Suspect",
        "has_words": True,
        "s": 13.0,
        "e": 15.0,
        "n_words": 4,
        "text": "I am not resisting.",
        "speech_ratio": 0.85,
        "rms_db": -20.0,
        "repeat_n": 1,
        "stretch": 0.5,
    },
    {
        "id": 2,
        "role": "Officer",
        "has_words": False,
        "s": 20.0,
        "e": 24.0,
        "n_words": 0,
        "text": "",
        "speech_ratio": 0.22,
        "rms_db": -42.0,
        "repeat_n": 4,
        "stretch": 4.0,
    },
]

VAD = {"frame_s": 0.03, "sr": 16000, "spans": [[10.0, 12.5], [13.0, 15.0], [20.0, 24.0]]}


@pytest.fixture()
def workbench(tmp_path):
    d = tmp_path / "wb"
    d.mkdir()
    (d / "subtitles_v4.ass").write_text(ASS)
    (d / "caption_evidence.json").write_text(json.dumps(EVIDENCE))
    (d / "vad_spans_cache.json").write_text(json.dumps(VAD))
    (d / "qa_v4.json").write_text(json.dumps([{"name": "duration", "ok": True}]))
    return d


@pytest.fixture()
def timeline():
    return build_timeline(
        TimelineSpec(
            name="wb_case",
            fps=30,
            width=1280,
            height=720,
            tracks=[
                TrackSpec(
                    name=MASTER_TRACK,
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=12.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                        ClipSpec(
                            name="raw_2",
                            source_start_s=12.0,
                            dur_s=14.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                    ],
                )
            ],
        )
    )


@pytest.fixture()
def registry(workbench, timeline, tmp_path):
    REG_TIMELINE.clear()
    REG_TIMELINE.append(timeline)
    reg = build_director_registry(DirectorToolbox(timeline, "wb_case", resolution=(1280, 720)))
    register_inspection_tools(reg, workbench)
    register_critic_tools(reg, workbench)
    register_session_memory_tools(reg, workbench / "session")
    engine = make_engine(f"sqlite:///{tmp_path / 'wb_agent.db'}")
    init_models(engine)
    from director_os.agent_runtime.checkpoint import CheckpointStore

    store = CheckpointStore(make_session_factory(engine))
    register_workshop_tools(
        reg,
        work_dir=workbench,
        source_ass=workbench / "subtitles_v4.ass",
        evidence_path=workbench / "caption_evidence.json",
        vad_cache_path=workbench / "vad_spans_cache.json",
        store=store,
        project_id="wb_case",
    )
    return reg


# ------------------------------------------------------------------- ass_io
def test_ass_round_trip_stable():
    cues = parse_cues(ASS)
    assert [c.text for c in cues] == ["Put the phone down.", "I am not resisting."]
    assert cues[0].style == "Officer"
    assert cues[0].start_s == 10.0 and cues[0].end_s == 12.5


def test_tag_style_inserted_once():
    once = ensure_tag_style(ASS)
    assert "Style: Tag," in once
    twice = ensure_tag_style(once)
    assert once.count("Style: Tag,") == 1 and twice.count("Style: Tag,") == 1


def test_rewrite_cue_preserves_fields():
    cues = parse_cues(ASS)
    line = rewrite_cue(cues[0].raw, "Tag", 10.0, 12.5, "[unaudible]")
    # ASS Dialogue order: Layer,Start,End,Style,Name,ML,MR,MV,Effect,Text
    assert line.startswith("Dialogue: 0,0:00:10.00,0:00:12.50,Tag,")
    assert line.endswith("[unaudible]")
    assert parse_cues(line)[0].style == "Tag"


def test_s_to_ts_format():
    assert s_to_ts(10.0) == "0:00:10.00"
    assert s_to_ts(3723.25) == "1:02:03.25"


# ------------------------------------------------------------------ gates
def test_evidence_backed_tag_passes_all_gates(registry):
    """The canonical five gates: color-vs-speaker, owner tag-only, no loop
    tiling, gap coverage, no-tags-on-clean-speech (remote contract)."""
    registry.dispatch("captions.init_work_copy", {})
    registry.dispatch(
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 20.0,
                    "end_s": 24.0,
                    "tag": "unaudible",
                    "reason": "speech_ratio 0.22, rms -42dB, repeat 4 = ASR loop",
                }
            ]
        },
    )
    report = registry.dispatch("qa.run_caption_checks", {"owner_windows": [[20.0, 24.0]]})
    assert report["green"], report
    names = {c["name"] for c in report["checks"]}
    assert {
        "caption_color_vs_speaker",
        "owner_windows_tag_only",
        "no_loop_text",
        "speech_gap_coverage",
        "no_tags_on_clean_speech",
    } <= names


def test_tag_on_clean_speech_is_red(registry):
    """A tag that REPLACED clean caption text trips the gate; a gap fill over
    cueless speech is legal (the owner's gap-fill law)."""
    registry.dispatch("captions.init_work_copy", {})
    registry.dispatch(
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 10.0,
                    "end_s": 12.5,
                    "tag": "unaudible",
                    "reason": "wrong: clean speech",
                }
            ]
        },
    )
    report = registry.dispatch("qa.run_caption_checks", {"owner_windows": [[20.0, 24.0]]})
    clean = next(c for c in report["checks"] if c["name"] == "no_tags_on_clean_speech")
    assert not clean["ok"]
    assert not report["green"]


def test_gap_fill_inserts_cue_for_uncaptioned_speech(registry):
    registry.dispatch("captions.init_work_copy", {})
    # 20-24 s has NO cue: the edit inserts one tag cue covering the window
    registry.dispatch(
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 20.0,
                    "end_s": 24.0,
                    "tag": "unaudible",
                    "reason": "speech with no caption",
                }
            ]
        },
    )
    work = registry.dispatch("captions.list_work_cues", {"start_s": 19.0, "end_s": 25.0})
    assert work["n_tag_cues"] == 1
    tag_cue = next(c for c in work["cues"] if c["text"].startswith("*"))
    assert tag_cue["s"] <= 20.0 + 1e-6 and tag_cue["e"] >= 24.0 - 1e-6


def test_reinit_restores_shipped_captions(registry):
    """Re-initializing re-copies the shipped source — the work copy is the only
    truth, so a discarded pass cannot haunt the gates."""
    registry.dispatch("captions.init_work_copy", {})
    registry.dispatch(
        "captions.apply_tag_edits",
        {"edits": [{"start_s": 20.0, "end_s": 24.0, "tag": "unaudible", "reason": "first pass"}]},
    )
    registry.dispatch("captions.init_work_copy", {})
    work = registry.dispatch("captions.list_work_cues", {"start_s": 19.0, "end_s": 25.0})
    assert work["n_tag_cues"] == 0


# ------------------------------------------------------------------- critic
def test_critic_red_names_failing_check(registry, timeline):
    before = registry.dispatch("qa.self_audit", {})
    assert before["green"], before["verdict"]
    track = track_by_name(timeline, MASTER_TRACK)
    track.insert(
        1,
        ot_schema.Gap(
            source_range=ot_time.TimeRange(
                start_time=ot_time.RationalTime(0, 30), duration=ot_time.RationalTime(3 * 30, 30)
            )
        ),
    )
    after = registry.dispatch("qa.self_audit", {})
    assert not after["green"]
    assert "no_master_gaps" in after["verdict"]


# ------------------------------------------------------------------- memory
def test_session_memory_survives_fresh_registry(registry, workbench, timeline, tmp_path):
    registry.dispatch("session.plan", {"action": "set", "tasks": ["inspect", "tag"]})
    registry.dispatch("session.plan", {"action": "mark_done", "task_index": 0})
    registry.dispatch(
        "session.notes", {"action": "append", "text": "VAD confirms 20-24s unreliable (ratio 0.22)"}
    )
    reg2 = build_director_registry(DirectorToolbox(timeline, "wb_case", resolution=(1280, 720)))
    register_inspection_tools(reg2, workbench)
    register_session_memory_tools(reg2, workbench / "session")
    st = reg2.dispatch("session.status", {})
    assert st["n_tasks_done"] == 1 and st["n_tasks_total"] == 2
    assert any("ratio 0.22" in n["text"] for n in st["last_notes"])


# ---------------------------------------------------------------- executor
def test_fail_closed_unknown_tool_rolls_back(registry, tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'fc.db'}")
    init_models(engine)
    from director_os.agent_runtime.checkpoint import CheckpointStore

    store = CheckpointStore(make_session_factory(engine))
    run = PlannerExecutor(registry, store, "wb_fc").run(
        ScriptedBackend([ToolStep(tool="timeline.explode_clip", args={})]),
        REG_TIMELINE[0],
    )
    assert not run.ok and run.rolled_back_to is not None
    assert "unknown tool" in run.steps[0].detail and "known:" in run.steps[0].detail
