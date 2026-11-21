"""Workshop tools: caption workbench + self-QA + undo + save (ACTION surface).

Hermetic: tmp-dir ASS/evidence/VAD fixtures, a real SQLite CheckpointStore,
a real OTIO timeline bound through REG_TIMELINE. No media, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import opentimelineio.adapters as ot_adapters
import opentimelineio.schema as ot_schema
import pytest
from director_os.agent_runtime import (
    CheckpointStore,
    ToolOperationError,
    register_workshop_tools,
)
from director_os.agent_runtime.inspection_tools import REG_TIMELINE
from director_os.agent_runtime.workshop_tools import _ass_ts
from director_os.core.db import init_models, make_engine, make_session_factory
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    RawClipMeta,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)

PROJECT = "proj_workshop_test"

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Officer,Roboto,42,&H00FFFFFF,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,58,1
Style: Suspect,Roboto,42,&H0000D7FF,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,58,1
Style: Narrator,Roboto,42,&H00EBCE87,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,46,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:12.50,Officer,,0,0,46,,Put the phone down.
Dialogue: 0,0:00:13.00,0:00:15.00,Suspect,,0,0,46,,{\\i1\\c&H9C9C9C&\\2c&H9C9C9C&}*unaudible*
Dialogue: 0,0:00:16.00,0:00:17.40,Narrator,,0,0,46,,It was 6:30 AM.
"""

EVIDENCE = [
    {
        "id": 0,
        "role": "Officer",
        "has_words": True,
        "s": 10.0,
        "e": 12.5,
        "text": "Put the phone down.",
        "speech_ratio": 0.94,
        "rms_db": -21.0,
        "repeat_n": 1,
    },
    {
        "id": 1,
        "role": "Suspect",
        "has_words": False,
        "s": 13.0,
        "e": 15.0,
        "text": "",
        "speech_ratio": 0.11,
        "rms_db": -38.0,
        "repeat_n": 4,
    },
    {
        "id": 2,
        "role": "Narrator",
        "has_words": True,
        "s": 16.0,
        "e": 17.4,
        "text": "It was 6:30 AM.",
        "speech_ratio": 0.9,
        "rms_db": -19.0,
        "repeat_n": 1,
    },
]

VAD = {"frame_s": 0.03, "sr": 16000, "spans": [[10.0, 12.5], [16.0, 17.4]]}


def _timeline() -> ot_schema.Timeline:
    return build_timeline(
        TimelineSpec(
            name="workshop_case",
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
                            dur_s=30.0,
                            meta=RawClipMeta(asset_id="bwc_1"),
                        ),
                    ],
                )
            ],
        )
    )


@pytest.fixture()
def env(tmp_path: Path):
    v3 = tmp_path / "v3"
    v3.mkdir()
    (v3 / "subtitles_v4.ass").write_text(ASS, encoding="utf-8")
    (v3 / "caption_evidence.json").write_text(json.dumps(EVIDENCE), encoding="utf-8")
    (v3 / "vad_spans_cache.json").write_text(json.dumps(VAD), encoding="utf-8")
    work = tmp_path / "agent_run"
    work.mkdir()

    engine = make_engine(f"sqlite:///{tmp_path / 'agent.db'}")
    init_models(engine)
    store = CheckpointStore(make_session_factory(engine))

    tl = _timeline()
    REG_TIMELINE.clear()
    REG_TIMELINE.append(tl)

    from director_os.agent_runtime.editing_tools import (
        DirectorToolbox,
        build_director_registry,
    )

    registry = build_director_registry(DirectorToolbox(tl, PROJECT, resolution=(1280, 720)))
    register_workshop_tools(
        registry,
        work_dir=work,
        source_ass=v3 / "subtitles_v4.ass",
        evidence_path=v3 / "caption_evidence.json",
        vad_cache_path=v3 / "vad_spans_cache.json",
        store=store,
        project_id=PROJECT,
    )
    yield {
        "work": work,
        "v3": v3,
        "registry": registry,
        "store": store,
        "timeline": tl,
    }
    REG_TIMELINE.clear()


def _dispatch(env: dict, tool: str, args: dict | None = None):
    return env["registry"].dispatch(tool, args or {})


# ---------------------------------------------------------------- workbench
def test_init_and_apply_replaces_overlapping_cues(env):
    _dispatch(env, "captions.init_work_copy")
    r = _dispatch(
        env,
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 10.0,
                    "end_s": 12.5,
                    "tag": "unaudible",
                    "reason": "officer murmur, speech_ratio 0.11, rms -38 dB",
                }
            ]
        },
    )
    assert "1 cue(s) replaced" in r["detail"]
    cues = _dispatch(env, "captions.list_work_cues", {"start_s": 9.0, "end_s": 18.0})
    texts = [c["text"] for c in cues["cues"]]
    assert "*unaudible*" in texts and "Put the phone down." not in texts
    tag_cue = next(c for c in cues["cues"] if c["text"] == "*unaudible*")
    assert tag_cue["style"] == "Officer"  # lane kept from the replaced cue
    audit = json.loads((env["work"] / "caption_edits_applied.json").read_text())
    assert audit[0]["cues_replaced"] == 1
    # the SHIPPED source is untouched (workbench law)
    assert "Put the phone down." in (env["v3"] / "subtitles_v4.ass").read_text()


def test_apply_merges_multiple_cues_and_gapfills(env):
    _dispatch(env, "captions.init_work_copy")
    _dispatch(
        env,
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 10.0,
                    "end_s": 17.4,
                    "tag": "cross-talk",
                    "reason": "simultaneous speech, no clean words",
                },
            ]
        },
    )
    cues = _dispatch(env, "captions.list_work_cues", {"start_s": 0.0, "end_s": 30.0})
    assert cues["n_cues"] == 1 and cues["n_tag_cues"] == 1
    _dispatch(
        env,
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 20.0,
                    "end_s": 22.0,
                    "tag": "mumbling",
                    "reason": "speech without cue, VAD span 20-22s",
                },
            ]
        },
    )
    cues = _dispatch(env, "captions.list_work_cues", {"start_s": 19.0, "end_s": 23.0})
    assert cues["n_tag_cues"] == 1  # gap-fill created a cue


def test_apply_requires_init_and_rejects_overlaps(env):
    with pytest.raises(ToolOperationError, match="init_work_copy"):
        _dispatch(
            env,
            "captions.apply_tag_edits",
            {
                "edits": [
                    {"start_s": 1.0, "end_s": 2.0, "tag": "unaudible", "reason": "no copy yet"}
                ]
            },
        )
    _dispatch(env, "captions.init_work_copy")
    with pytest.raises(Exception, match="overlap"):
        _dispatch(
            env,
            "captions.apply_tag_edits",
            {
                "edits": [
                    {"start_s": 1.0, "end_s": 5.0, "tag": "unaudible", "reason": "first"},
                    {"start_s": 4.0, "end_s": 6.0, "tag": "unaudible", "reason": "second"},
                ]
            },
        )


# ---------------------------------------------------------------- self-QA
def test_qa_green_after_owner_window_edit(env):
    _dispatch(env, "captions.init_work_copy")
    # owner law: the unclear-suspect window must be tag-only (it already is)
    report = _dispatch(env, "qa.run_caption_checks", {"owner_windows": [[13.0, 15.0]]})
    assert report["green"] is True, report
    assert (env["work"] / "qa_caption_work.json").exists()


def test_qa_catches_wrong_color_and_gap(env):
    _dispatch(env, "captions.init_work_copy")
    # corrupt: give the narrator cue the Suspect style (color-vs-role violation)
    work_ass = env["work"] / "captions_work.ass"
    work_ass.write_text(ASS.replace("Narrator,,0,0,46,,It was", "Suspect,,0,0,46,,It was"))
    report = _dispatch(env, "qa.run_caption_checks", {"owner_windows": [[13.0, 15.0]]})
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["caption_color_vs_speaker"]["ok"] is False
    assert report["green"] is False

    # extend the VAD cache with an uncovered span -> gap check fails
    vad = json.loads((env["v3"] / "vad_spans_cache.json").read_text())
    vad["spans"].append([25.0, 27.5])
    (env["v3"] / "vad_spans_cache.json").write_text(json.dumps(vad))
    report = _dispatch(env, "qa.run_caption_checks", {"owner_windows": [[13.0, 15.0]]})
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["speech_gap_coverage"]["ok"] is False
    # the agent FIXES the gap with a tag edit -> green again
    _dispatch(
        env,
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 25.0,
                    "end_s": 27.5,
                    "tag": "unaudible",
                    "reason": "VAD speech span without caption",
                }
            ]
        },
    )
    report = _dispatch(env, "qa.run_caption_checks", {"owner_windows": [[13.0, 15.0]]})
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["speech_gap_coverage"]["ok"] is True


def test_qa_catches_owner_window_plain_text(env):
    _dispatch(env, "captions.init_work_copy")
    report = _dispatch(env, "qa.run_caption_checks", {"owner_windows": [[10.0, 12.5]]})
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["owner_windows_tag_only"]["ok"] is False


def test_qa_detects_loop_text(env):
    _dispatch(env, "captions.init_work_copy")
    work_ass = env["work"] / "captions_work.ass"
    loop = "\n".join(
        f"Dialogue: 0,0:01:{i:02d}.00,0:01:{i:02d}.30,Suspect,,0,0,46,,people want to stop"
        for i in range(3)
    )
    work_ass.write_text(ASS + "\n" + loop + "\n")
    report = _dispatch(env, "qa.run_caption_checks")
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["no_loop_text"]["ok"] is False


def test_qa_requires_vad_cache(env):
    _dispatch(env, "captions.init_work_copy")
    (env["v3"] / "vad_spans_cache.json").unlink()
    with pytest.raises(ToolOperationError, match="speech_spans"):
        _dispatch(env, "qa.run_caption_checks")


# ---------------------------------------------------------------- undo + save
def test_undo_list_and_rollback_rebind(env):
    store, tl = env["store"], env["timeline"]
    v1 = store.checkpoint(tl, PROJECT, note="clean two-clip cut")
    env["registry"].dispatch("timeline.remove_clip", {"item_index": 0})
    store.checkpoint(REG_TIMELINE[0], PROJECT, note="over-trimmed")

    listing = _dispatch(env, "undo.list_checkpoints")
    assert listing["n_versions"] >= 2

    r = _dispatch(env, "undo.rollback", {"to_version_num": v1})
    assert str(v1) in r["detail"] and "plan-final" in r["detail"]
    restored = REG_TIMELINE[0]
    assert restored is not tl  # rebound to the loaded object
    assert len(list(restored.tracks[0])) == 1  # the removed clip is back
    assert store.list_versions(PROJECT)[-1]["checkpoint_note"].startswith("rollback")


def test_save_timeline_export(env):
    _dispatch(env, "captions.init_work_copy")  # unrelated: keep session flow real
    r = _dispatch(env, "timeline.save")
    out = Path(r["path"])
    assert out.exists() and out.name == "timeline_saved.otio"
    tl = ot_adapters.read_from_file(str(out))
    assert len(list(tl.tracks[0])) == 1


def test_save_requires_bound_timeline(env):
    REG_TIMELINE.clear()
    with pytest.raises(ToolOperationError, match="no working timeline bound"):
        _dispatch(env, "timeline.save")


def test_ass_ts_formatting():
    assert _ass_ts(609.4) == "0:10:09.40"
    assert _ass_ts(0) == "0:00:00.00"
    assert _ass_ts(3723.126) == "1:02:03.13"  # centisecond rounding, ASS-style
