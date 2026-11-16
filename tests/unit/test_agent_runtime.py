"""M2 agent runtime: director tools, checkpoint/rollback, planner-executor loop.

Covers PLAN §8 M2 build items:
- typed editing tools mutate a real OTIO timeline (master cut + overlay cards)
- registry validation rejects hallucinated args before the timeline is touched
- CheckpointStore persists timeline_versions + agent_tool_logs to SQLite
- PlannerExecutor: checkpoint-before-execute, auto-rollback on bad step
"""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.agent_runtime import (
    CheckpointStore,
    DirectorToolbox,
    PlannerExecutor,
    ScriptedBackend,
    ToolOperationError,
    ToolStep,
    build_director_registry,
)
from director_os.core.db import (
    init_models,
    make_engine,
    make_session_factory,
)
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    OVERLAY_TRACK,
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)

PROJECT = "proj_agent_test"


@pytest.fixture()
def timeline():
    return build_timeline(
        TimelineSpec(
            name="agent_case",
            fps=30,
            width=1920,
            height=1080,
            tracks=[
                TrackSpec(
                    name=MASTER_TRACK,
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=10.0,
                            meta={"kind": "raw", "asset_id": "a1"},
                        ),
                        ClipSpec(
                            name="raw_2",
                            source_start_s=10.0,
                            dur_s=8.0,
                            meta={"kind": "raw", "asset_id": "a1"},
                        ),
                        ClipSpec(
                            name="narr",
                            source_start_s=0.0,
                            dur_s=5.0,
                            meta={"kind": "narration", "script_text": "Original narration line."},
                        ),
                    ],
                ),
                TrackSpec(
                    name=OVERLAY_TRACK,
                    items=[GapSpec(dur_s=23.0)],
                ),
            ],
        )
    )


@pytest.fixture()
def store(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path}/agent.db")
    init_models(engine)
    return CheckpointStore(make_session_factory(engine))


# ------------------------------------------------------------------ tools
def test_trim_and_split_mutate_master(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    assert box.trim_clip(0, 2.0, 6.0)["ok"]
    items = list(timeline.tracks[0])
    sr = items[0].source_range
    assert sr.start_time.to_seconds() == pytest.approx(2.0)
    assert sr.duration.to_seconds() == pytest.approx(6.0)

    assert box.split_clip(0, 3.0)["ok"]
    items = list(timeline.tracks[0])
    assert len(items) == 4  # 10s clip split -> a/b, plus the other two
    a, b = items[0], items[1]
    assert a.name.endswith(".a") and b.name.endswith(".b")
    assert a.source_range.duration.to_seconds() == pytest.approx(3.0)
    assert b.source_range.start_time.to_seconds() == pytest.approx(2.0 + 3.0)
    # metadata contract survives the split on both halves
    for c in (a, b):
        assert c.metadata["director_os"]["kind"] == "raw"


def test_move_and_remove_clip(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    names_before = [c.name for c in timeline.tracks[0]]
    assert box.move_clip(2, 0)["ok"]
    assert [c.name for c in timeline.tracks[0]][0] == names_before[2]
    assert box.remove_clip(0)["ok"]
    assert len(list(timeline.tracks[0])) == 2


def test_narration_text_rewrite(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    out = box.set_narration_text(2, "Rewritten by the director agent.")
    assert out["ok"]
    clip = list(timeline.tracks[0])[2]
    assert clip.metadata["director_os"]["script_text"].startswith("Rewritten")


def test_narration_tool_rejects_raw_clip(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    with pytest.raises(ToolOperationError, match="not a narration clip"):
        box.set_narration_text(0, "nope")


def test_blur_region_validation(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    assert box.add_blur_region(0, [1500, 30, 300, 90], sigma=18)["ok"]
    assert len(timeline.tracks[0][0].metadata["director_os"]["blur_regions"]) == 1
    with pytest.raises(ToolOperationError, match="exceeds canvas"):
        box.add_blur_region(0, [1800, 1000, 300, 120])
    with pytest.raises(ToolOperationError, match="bbox must be"):
        box.add_blur_region(0, [10, 10, 50])
    with pytest.raises(ToolOperationError, match="not a raw clip"):
        box.add_blur_region(2, [10, 10, 50, 50])


def test_remove_blur_region(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    box.add_blur_region(0, [100, 100, 200, 80], sigma=24)
    box.add_blur_region(0, [400, 100, 200, 80], sigma=24)
    assert box.remove_blur_region(0, 1)["ok"]
    regions = timeline.tracks[0][0].metadata["director_os"]["blur_regions"]
    assert len(regions) == 1
    with pytest.raises(ToolOperationError, match="out of range"):
        box.remove_blur_region(0, 5)


def test_card_inserts_into_gap_and_splits_it(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    out = box.add_card("location_intro", "5TH & MARKET", start_s=4.0, dur_s=5.0, subtitle="BWC 1")
    assert out["ok"]
    overlay = list(timeline.tracks[1])
    assert len(overlay) == 3  # gap(4s) + card(5s) + gap(14s)
    card = overlay[1]
    assert card.metadata["director_os"]["template"] == "location_intro"
    assert card.source_range.duration.to_seconds() == pytest.approx(5.0)
    assert overlay[0].source_range.duration.to_seconds() == pytest.approx(4.0)
    assert overlay[2].source_range.duration.to_seconds() == pytest.approx(14.0)


def test_card_outside_gap_rejected(timeline):
    box = DirectorToolbox(timeline, PROJECT)
    box.add_card("location_intro", "A", start_s=2.0, dur_s=10.0)
    # total overlay = 23s; this would end at 24s -> no covering gap
    with pytest.raises(ToolOperationError, match="no free gap"):
        box.add_card("location_intro", "B", start_s=20.0, dur_s=5.0)


# ------------------------------------------------------------------ registry wiring
def test_registry_dispatches_and_rejects(timeline):
    reg = build_director_registry(DirectorToolbox(timeline, PROJECT))
    names = reg.names()
    assert "timeline.trim_clip" in names and "cards.add" in names

    out = reg.dispatch(
        "timeline.trim_clip", {"item_index": 1, "new_start_s": 0.5, "new_dur_s": 4.0}
    )
    assert out["ok"]

    from director_os.agent_runtime.registry import ToolDispatchError

    with pytest.raises(ToolDispatchError, match="rejected args"):
        reg.dispatch("timeline.trim_clip", {"item_index": 0, "new_start_s": -5, "new_dur_s": 4.0})
    with pytest.raises(ToolDispatchError, match="unknown tool"):
        reg.dispatch("timeline.teleport_clip", {})


def test_openai_schemas_available(timeline):
    reg = build_director_registry(DirectorToolbox(timeline, PROJECT))
    schemas = reg.openai_schemas()
    assert len(schemas) == 8
    fn = next(s for s in schemas if s["function"]["name"] == "redaction.add_blur")
    assert fn["function"]["parameters"]["properties"]["bbox"]["type"] == "array"


# ------------------------------------------------------------------ checkpoint store
def test_checkpoint_and_rollback_roundtrip(timeline, store):
    v1 = store.checkpoint(timeline, PROJECT, note="initial cut")
    assert v1 == 1
    box = DirectorToolbox(timeline, PROJECT)
    box.trim_clip(0, 5.0, 3.0)
    v2 = store.checkpoint(timeline, PROJECT, note="trimmed")
    assert v2 == 2

    restored, v3 = store.rollback(PROJECT, v1)
    assert v3 == 3  # rollback writes a NEW version, history intact
    assert [t.name for t in restored.tracks] == [t.name for t in timeline.tracks]
    assert list(restored.tracks[0])[0].source_range.duration.to_seconds() == pytest.approx(10.0)

    versions = store.list_versions(PROJECT)
    assert [v["version_num"] for v in versions] == [1, 2, 3]
    assert versions[2]["checkpoint_note"] == "rollback to v1"

    with pytest.raises(Exception, match="no version 99"):
        store.load_version(PROJECT, 99)


def test_tool_logs_persist(store):
    store.log_tool(
        "director_agent",
        "timeline.trim_clip",
        {"item_index": 0},
        result_summary="ok",
        duration_ms=42,
        project_id=PROJECT,
    )
    store.log_tool(
        "director_agent",
        "cards.add",
        {},
        duration_ms=10,
        project_id=PROJECT,
        ok=False,
        error="no free gap",
    )
    logs = store.recent_logs(PROJECT)
    assert len(logs) == 2  # ok row + failure row (errors live in the summary)
    assert any("ERROR: no free gap" in log_row["summary"] for log_row in logs)


# ------------------------------------------------------------------ planner-executor
def test_plan_executes_with_checkpoints(timeline, store):
    reg = build_director_registry(DirectorToolbox(timeline, PROJECT))
    runner = PlannerExecutor(reg, store, PROJECT, agent="director_v1")
    plan = [
        ToolStep(
            tool="narration.set_text",
            args={"item_index": 2, "script_text": "Plan-approved narration."},
            checkpoint_note="proofread line 3",
        ),
        ToolStep(
            tool="redaction.add_blur",
            args={"item_index": 0, "bbox": [1400, 20, 400, 110], "sigma": 16},
            checkpoint_note="badge watermark",
        ),
    ]
    run = runner.execute_plan(plan, timeline)
    assert run.ok and run.steps_ok == 2
    assert run.timeline is timeline
    narr = list(run.timeline.tracks[0])[2]
    assert narr.metadata["director_os"]["script_text"] == "Plan-approved narration."
    versions = store.list_versions(PROJECT)
    # plan start + 2 pre-step + terminal = 4
    assert len(versions) == 4
    assert versions[-1]["checkpoint_note"] == "director_v1: plan complete"


def test_bad_step_rolls_back_durable_state(timeline, store):
    reg = build_director_registry(DirectorToolbox(timeline, PROJECT))
    runner = PlannerExecutor(reg, store, PROJECT, agent="director_v2")
    plan = [
        ToolStep(
            tool="narration.set_text",
            args={"item_index": 2, "script_text": "Good edit."},
            checkpoint_note="step 1",
        ),
        # semantically impossible: narration clip is not a raw clip
        ToolStep(
            tool="redaction.add_blur",
            args={"item_index": 2, "bbox": [10, 10, 50, 50]},
            checkpoint_note="step 2 (doomed)",
        ),
        ToolStep(
            tool="timeline.trim_clip",
            args={"item_index": 0, "new_start_s": 1.0, "new_dur_s": 3.0},
            checkpoint_note="never reached",
        ),
    ]
    run = runner.execute_plan(plan, timeline)
    assert not run.ok
    assert run.steps_ok == 1
    assert run.rolled_back_to is not None
    # rollback = state as of before the FAILED step: step 1's edit survives,
    # the doomed step 2 (blur on a narration clip) was never applied
    narr = list(run.timeline.tracks[0])[2]
    assert narr.metadata["director_os"]["script_text"] == "Good edit."
    assert "blur_regions" not in narr.metadata["director_os"]
    # history intact: the pre-failure state is still loadable
    versions = store.list_versions(PROJECT)
    assert len(versions) >= 4
    pre_fail = store.load_version(PROJECT, versions[-2]["version_num"])
    assert list(pre_fail.tracks[0])[2].metadata["director_os"]["script_text"] == "Good edit."
    # failure written to the audit log
    logs = store.recent_logs(PROJECT)
    assert any("not a raw clip" in log_row["summary"] for log_row in logs)


def test_validation_failure_rolls_back_before_mutation(timeline, store):
    reg = build_director_registry(DirectorToolbox(timeline, PROJECT))
    runner = PlannerExecutor(reg, store, PROJECT)
    doomed = ToolStep(
        tool="timeline.trim_clip",
        args={"item_index": 0, "new_start_s": -99, "new_dur_s": 1.0},
    )
    run = runner.execute_plan([doomed], timeline)
    assert not run.ok
    # timeline untouched: schema rejection happens before the tool function runs
    assert list(run.timeline.tracks[0])[0].source_range.duration.to_seconds() == pytest.approx(10.0)


def test_scripted_backend_loop(timeline, store):
    reg = build_director_registry(DirectorToolbox(timeline, PROJECT))
    runner = PlannerExecutor(reg, store, PROJECT, agent="director_v3")
    backend = ScriptedBackend(
        [ToolStep(tool="timeline.remove_clip", args={"item_index": 1})]
    )
    run = runner.run(backend, timeline)
    assert run.ok
    assert backend.propose_calls == 1
    assert len(list(run.timeline.tracks[0])) == 2
