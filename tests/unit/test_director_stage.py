"""M2 close-out: the LLM Director autodrive stage (the 'LLM controls the edit' seam).

Covers:
- narration.add toolbox conversion: raw -> NarratorClipMeta, source window preserved
- stage_director with a scripted backend: edits persist as versions, the
  active pointer moves, the report accounts every step
- soft-fail paths: disabled flag / no provider / zero proposal / failing step
  (rollback keeps the durable cut) / round cap
- the brief: perception products (VLM insights, YAMNet events incl. Music,
  speech beats) appear in the state summary the model actually sees
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from director_os.agent_runtime.agents import ScriptedBackend, ToolStep
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.editing_tools import (
    DirectorToolbox,
    ToolOperationError,
    build_director_registry,
)
from director_os.agent_runtime.llm import LLMBackendError
from director_os.core.config import get_settings
from director_os.core.db import (
    PerceptionIndexRow,
    ProjectRow,
    init_models,
    make_engine,
    make_session_factory,
)
from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta, parse_clip_meta
from director_os.otio_engine.segment_cache import VideoRenderSpec
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    OVERLAY_TRACK,
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    track_by_name,
)
from director_os.worker import director as director_mod
from director_os.worker.director import (
    _BriefedDirector,
    build_director_brief,
    master_state_lines,
    stage_director,
)
from director_os.worker.pipeline import PipelineDeps

PROJECT = "proj_director_stage"


def _timeline():
    return build_timeline(
        TimelineSpec(
            name="director_case",
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
                            dur_s=8.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                        ClipSpec(
                            name="raw_3",
                            source_start_s=20.0,
                            dur_s=5.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                    ],
                ),
                TrackSpec(name=OVERLAY_TRACK, items=[GapSpec(dur_s=25.0)]),
            ],
        )
    )


@pytest.fixture()
def env(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path}/director.db")
    init_models(engine)
    sessions = make_session_factory(engine)
    settings = dataclasses.replace(
        get_settings(),
        output_dir=tmp_path / "output",
        ingest_dir=tmp_path / "output" / "ingest",
        segments_dir=tmp_path / "output" / "segments",
    )
    deps = PipelineDeps(
        sessions=sessions,
        settings=settings,
        video=VideoRenderSpec(w=1280, h=720, fps=30),
        llm_direct=True,
        director_max_rounds=1,
    )
    with sessions() as s:
        s.add(ProjectRow(id=PROJECT, title="director stage test"))
        s.commit()
    timeline = _timeline()
    store = CheckpointStore(sessions)
    base_version = store.checkpoint(
        timeline, PROJECT, note="v1 auto-cut", created_by="assembler_agent"
    )
    with sessions() as s:
        project = s.get(ProjectRow, PROJECT)
        project.active_otio_version_id = base_version
        s.commit()
    return deps, sessions, store, timeline, base_version


def _seed_perception(sessions) -> None:
    with sessions() as s:
        row = PerceptionIndexRow(
            asset_id="a1",
            shots_json={
                "visual_timeline": {
                    "model": "fake",
                    "sheets": 1,
                    "insights": [
                        {
                            "timestamp_s": 1.0,
                            "kind": "observation",
                            "description": "Officer approaches the sedan window",
                            "confidence": 0.92,
                        }
                    ],
                },
                "audio_events": {
                    "backend": "yamnet-cpu",
                    "duration_s": 25.0,
                    "events": [{"kind": "Music", "start_s": 5.0, "end_s": 9.0, "peak_score": 0.71}],
                },
            },
        )
        s.merge(row)
        s.commit()


# ------------------------------------------------------------------ toolbox tool
def test_add_narration_converts_raw_clip():
    timeline = _timeline()

    box = DirectorToolbox(timeline, PROJECT, resolution=(1280, 720))
    out = box.add_narration(0, "The night starts like any other shift.")
    assert out["ok"]
    clip = list(timeline.tracks[0])[0]
    meta = parse_clip_meta(clip)
    assert isinstance(meta, NarratorClipMeta)
    assert meta.script_text == "The night starts like any other shift."
    assert meta.broll_asset_id == "a1"
    assert meta.broll_range == (0.0, 12.0)
    # the trimmed source range is untouched — the same window plays as b-roll
    assert clip.source_range.duration.to_seconds() == pytest.approx(12.0)
    # narration.set_text now addresses the converted clip
    assert box.set_narration_text(0, "Refined line.")["ok"]
    with pytest.raises(ToolOperationError, match="not a raw clip"):
        box.add_narration(0, "double convert")
    with pytest.raises(ToolOperationError, match="must not be empty"):
        box.add_narration(1, "   ")


def test_registry_exposes_narration_add():
    box = DirectorToolbox(_timeline(), PROJECT)
    reg = build_director_registry(box)
    assert "narration.add" in reg.names()


# ------------------------------------------------------------------ stage: happy path
def test_stage_director_scripted_edits_persist_and_move_pointer(env):
    deps, sessions, store, timeline, base_version = env
    _seed_perception(sessions)
    steps = [
        ToolStep(tool="timeline.remove_clip", args={"item_index": 1}),
        ToolStep(
            tool="narration.add",
            args={"item_index": 0, "script_text": "The night starts like any other shift."},
        ),
        ToolStep(
            tool="cards.add",
            args={"template": "evidence_fact", "title": "BAC 0.21", "start_s": 2.0, "dur_s": 3.0},
        ),
    ]
    scripted = ScriptedBackend(steps)
    original_build = director_mod.build_backend
    director_mod.build_backend = lambda registry: scripted  # type: ignore[assignment]
    try:
        timeline, version, report = stage_director(
            timeline, "a1", {"duration_s": 25.0}, PROJECT, deps
        )
    finally:
        director_mod.build_backend = original_build

    assert report["skipped"] is None
    assert report["rounds"] == 1
    assert report["steps_ok"] == 3
    assert report["steps_failed"] == 0
    assert version > base_version
    assert report["final_version"] == version

    with sessions() as s:
        project = s.get(ProjectRow, PROJECT)
        assert project.active_otio_version_id == version

    master = track_by_name(timeline, MASTER_TRACK)
    clips = [i for i in master]
    assert len(clips) == 2  # raw_2 dropped
    meta0 = parse_clip_meta(clips[0])
    assert isinstance(meta0, NarratorClipMeta)
    assert meta0.script_text.startswith("The night starts")
    overlay = track_by_name(timeline, OVERLAY_TRACK)
    card_kinds = []
    for item in overlay:
        try:
            card_kinds.append(parse_clip_meta(item).kind)
        except Exception:  # noqa: BLE001 — gaps carry no contract block
            continue
    assert "card" in card_kinds


# ------------------------------------------------------------------ soft-fail paths
def test_stage_director_disabled_passthrough(env):
    deps, sessions, _store, timeline, base_version = env
    deps.llm_direct = False
    timeline, version, report = stage_director(timeline, "a1", {}, PROJECT, deps)
    assert report["skipped"] == "llm_direct disabled"
    assert version == base_version


def test_stage_director_no_provider_passthrough(env, monkeypatch):
    deps, sessions, _store, timeline, base_version = env

    def _raise(registry):
        raise LLMBackendError("LLM not configured: set LLM_API_KEY")

    monkeypatch.setattr(director_mod, "build_backend", _raise)
    timeline, version, report = stage_director(timeline, "a1", {}, PROJECT, deps)
    assert "LLM not configured" in report["skipped"]
    assert version == base_version
    assert len(list(timeline.tracks[0])) == 3  # untouched


def test_stage_director_zero_proposal_locks_cut(env, monkeypatch):
    deps, sessions, _store, timeline, base_version = env

    class DoneBackend(ScriptedBackend):
        def propose(self, state_summary: str, role: str) -> list[ToolStep]:
            self.propose_calls += 1
            return []

    monkeypatch.setattr(director_mod, "build_backend", lambda registry: DoneBackend([]))
    timeline, version, report = stage_director(timeline, "a1", {}, PROJECT, deps)
    assert report["rounds"] == 1
    assert report["steps_ok"] == 0
    assert version == base_version  # no edits -> pointer never moves
    assert len(list(timeline.tracks[0])) == 3


def test_stage_director_bad_step_rolls_back(env, monkeypatch):
    deps, sessions, _store, timeline, base_version = env
    steps = [ToolStep(tool="timeline.remove_clip", args={"item_index": 999})]
    monkeypatch.setattr(director_mod, "build_backend", lambda registry: ScriptedBackend(steps))
    timeline, version, report = stage_director(timeline, "a1", {}, PROJECT, deps)
    assert report["steps_failed"] == 1
    assert report["steps_ok"] == 0
    assert version == base_version  # rollback -> pointer stays on the good cut
    assert len(list(timeline.tracks[0])) == 3  # durable state restored


def test_stage_director_round_cap(env, monkeypatch):
    deps, sessions, _store, timeline, base_version = env
    deps.director_max_rounds = 3
    calls = {"n": 0}

    class LoopBackend(ScriptedBackend):
        def propose(self, state_summary: str, role: str) -> list[ToolStep]:
            calls["n"] += 1
            return [
                ToolStep(
                    tool="timeline.trim_clip",
                    args={"item_index": 0, "new_start_s": 0.5, "new_dur_s": 10.0},
                )
            ]

    monkeypatch.setattr(director_mod, "build_backend", lambda registry: LoopBackend([]))
    timeline, version, report = stage_director(timeline, "a1", {}, PROJECT, deps)
    assert report["rounds"] == 3
    assert calls["n"] == 3
    assert report["steps_ok"] == 3
    assert version > base_version


# ------------------------------------------------------------------ brief + summary
def test_brief_carries_perception_products(env):
    deps, sessions, _store, _timeline, _base = env
    _seed_perception(sessions)
    transcript = {
        "duration_s": 25.0,
        "words": [
            {"text": "Step", "start_s": 1.0, "end_s": 1.3, "speaker": "spk0"},
            {"text": "out", "start_s": 1.4, "end_s": 1.7, "speaker": "spk0"},
            {"text": "of", "start_s": 3.2, "end_s": 3.4, "speaker": "spk0"},
            {"text": "the", "start_s": 3.5, "end_s": 3.7, "speaker": "spk0"},
            {"text": "car", "start_s": 3.8, "end_s": 4.1, "speaker": "spk0"},
        ],
    }
    brief = build_director_brief(deps.sessions, "a1", transcript)
    assert "Officer approaches the sedan window" in brief
    assert "COPYRIGHT GUARDRAIL" in brief
    assert "yamnet-cpu" in brief
    assert '1.0-1.7s spk0: "Step out"' in brief
    assert "raw window: 25.0s" in brief


def test_state_summary_lists_master_and_mission(env):
    deps, sessions, _store, timeline, _base = env
    _seed_perception(sessions)
    transcript: dict[str, Any] = {"duration_s": 25.0, "words": []}
    brief = build_director_brief(deps.sessions, "a1", transcript)
    executor = _BriefedDirector(None, CheckpointStore(sessions), PROJECT, brief)  # type: ignore[arg-type]
    summary = executor._state_summary(timeline)
    assert "[0] RAW src=0.00s dur=12.00s" in summary
    assert "[1] RAW src=12.00s dur=8.00s" in summary
    assert "EDIT MISSION" in summary
    assert "PERCEPTION BRIEF" in summary
    assert "NO SPEECH DETECTED" in summary
    assert "master track (25.0s total)" in summary


def test_master_state_lines_narration_and_cards():
    timeline = _timeline()

    box = DirectorToolbox(timeline, PROJECT, resolution=(1280, 720))
    box.add_narration(1, "He sways before he speaks.")
    box.add_card("location_intro", "EUSTIS, FL", start_s=0.0, dur_s=3.0, subtitle="BWC 1")
    text = master_state_lines(timeline)
    assert "[1] NARRATION src=12.00s dur=8.00s" in text
    assert "He sways before he speaks." in text
    assert "t=0.0s CARD location_intro" in text
