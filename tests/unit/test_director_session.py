"""DirectorSession — the 2-phase Story Map -> Full-Arc Execution flow (M5).

Phase A (macro plan JSON): lenient parsing, pre-flight problem detection.
Phase B: plan expansion into ordered tool steps, then the refinement loop
that may NOT terminate while duration invariants are unmet. End-to-end runs
against the REAL tool registry + checkpoint store on sqlite with a scripted
backend, so the expansion/expansion-order/invariant gates are exercised
exactly as production sees them.
"""

from __future__ import annotations

import json

import pytest
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.director_session import (
    DirectorSession,
    MacroPlan,
    expand_macro_plan,
    parse_macro_plan,
)
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
from director_os.agent_runtime.plan_invariants import invariants_ok
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)

SOURCE_DUR = 2627.0


def _good_plan_json() -> str:
    acts = [
        {
            "name": "Cold Open",
            "blocks": [
                {"type": "raw", "source_start_s": 1650.0, "dur_s": 30.0},
                {"type": "raw", "source_start_s": 1900.0, "dur_s": 26.0},
                {
                    "type": "narration",
                    "script": "July 4th, 2022. A bodycam is rolling.",
                    "dur_s": 16.0,
                },
            ],
        }
    ]
    # acts 1-6: chronological coverage from ~60s to ~2560s
    windows = [(60, 380), (390, 760), (770, 1140), (1150, 1520), (1530, 1990), (2000, 2560)]
    for i, (lo, hi) in enumerate(windows, start=1):
        blocks = [
            {
                "type": "narration",
                "script": f"Act {i} bridge setting up the tension.",
                "dur_s": 12.0,
            }
        ]
        for j in range(2):
            start = lo + j * ((hi - lo) / 3)
            blocks.append({"type": "raw", "source_start_s": round(start, 1), "dur_s": 45.0})
        acts.append({"name": f"Act {i}", "blocks": blocks})
    return json.dumps(
        {
            "title": "Eustis bodycam case",
            "acts": acts,
            "cards": [
                {
                    "template": "location_intro",
                    "title": "CASE FILE",
                    "subtitle": "Eustis, FL",
                    "at_act": 1,
                    "at_block": 0,
                },
                {
                    "template": "case_outcome",
                    "title": "OUTCOME",
                    "subtitle": "verified",
                    "at_act": 6,
                    "at_block": 0,
                },
            ],
        }
    )


def _scaffold():
    return build_timeline(
        TimelineSpec(
            name="scaffold",
            fps=30,
            width=1920,
            height=1080,
            tracks=[
                TrackSpec(
                    name="V1_master",
                    items=[
                        ClipSpec(
                            name="raw_scaffold",
                            source_start_s=0.0,
                            dur_s=SOURCE_DUR,
                            meta=RawClipMeta(asset_id="asset_x"),
                        )
                    ],
                ),
                TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=SOURCE_DUR)]),
            ],
        )
    )


class ScriptedChain:
    """FallbackChain stand-in: one canned macro plan, no refinement proposals."""

    provider = "scripted_chain"
    active_label = "scripted:plan"
    model = "scripted:plan"

    def __init__(self, plan_json: str, proposals: list[list] | None = None):
        self._plan_json = plan_json
        self._proposals = proposals or [[]]
        self.complete_json_calls = 0
        self.propose_calls = 0

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        self.complete_json_calls += 1
        return self._plan_json

    def propose(self, state_summary: str, role: str):
        from director_os.agent_runtime.agents import ToolStep

        self.propose_calls += 1
        idx = min(self.propose_calls - 1, len(self._proposals) - 1)
        batch = self._proposals[idx]
        return [
            s if isinstance(s, ToolStep) else ToolStep(tool=s["tool"], args=s["args"])
            for s in batch
        ]

    def record_results(self, steps, results) -> None:
        pass

    def reset_history(self) -> None:
        pass


# ------------------------------------------------------------------ phase A
def test_parse_macro_plan_accepts_fenced_json():
    text = "```json\n" + _good_plan_json() + "\n```"
    plan = parse_macro_plan(text)
    assert len(plan.acts) == 7
    assert plan.raw_count >= 10
    assert plan.narration_count >= 7


def test_parse_macro_plan_rejects_garbage():
    with pytest.raises(ValueError):
        parse_macro_plan("the documentary should open with a bang")
    with pytest.raises(ValueError):
        parse_macro_plan('{"title": "x"}')  # no acts -> validation error


def test_plan_problems_detect_truncated_plan():
    plan = parse_macro_plan(_good_plan_json())
    session = DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p1")),
        None,  # type: ignore[arg-type]
        "p1",
        ScriptedChain(_good_plan_json()),
        source_duration_s=SOURCE_DUR,
    )
    assert session._plan_problems(plan) == []


def test_plan_problems_flag_undercoverage():
    plan = parse_macro_plan(_good_plan_json())
    # drop the last act -> the ending disappears
    plan.acts = plan.acts[:-1]
    session = DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p1")),
        None,  # type: ignore[arg-type]
        "p1",
        ScriptedChain(_good_plan_json()),
        source_duration_s=SOURCE_DUR,
    )
    problems = session._plan_problems(plan)
    assert any("latest clip ends" in p for p in problems)


# ------------------------------------------------------------------ phase B0
def test_expand_macro_plan_orders_steps_and_computes_card_positions():
    plan = parse_macro_plan(_good_plan_json())
    steps = expand_macro_plan(plan, SOURCE_DUR)
    tools = [s.tool for s in steps]
    # first block is raw -> scaffold consumed by trim
    assert steps[0].tool == "timeline.trim_clip"
    assert steps[0].args["new_start_s"] == 1650.0
    assert "timeline.add_narration" in tools and "timeline.insert_raw_clip" in tools
    assert tools[-1] == "cards.add" or tools[-2] == "cards.add"
    card_steps = [s for s in steps if s.tool == "cards.add"]
    assert card_steps[0].args["start_s"] > 60.0  # after cold open + act1 narration
    assert all(s.args["start_s"] >= 0 for s in card_steps)


def test_expand_macro_plan_narration_first_keeps_and_drops_scaffold():
    plan = parse_macro_plan(_good_plan_json())
    plan.acts[0].blocks.insert(
        0,
        {
            "type": "narration",
            "script": "Before the arrest, everything looked routine.",
            "dur_s": 10.0,
        },
    )
    steps = expand_macro_plan(plan, SOURCE_DUR)
    tools = [s.tool for s in steps]
    assert steps[0].tool == "timeline.add_narration"
    assert "timeline.remove_clip" in tools  # unconsumed scaffold dropped at the end
    remove_idx = tools.index("timeline.remove_clip")
    assert remove_idx == len(tools) - 1 - len(plan.cards)  # before the cards


# ------------------------------------------------------------------ end to end
@pytest.fixture()
def store(tmp_path):
    from director_os.core.db import init_models, make_engine, make_session_factory

    engine = make_engine(f"sqlite:///{tmp_path}/session.db")
    init_models(engine)
    return CheckpointStore(make_session_factory(engine))


def test_director_session_end_to_end_passes_invariants(store):
    scaffold = _scaffold()
    toolbox = DirectorToolbox(scaffold, "proj_e2e")
    registry = build_director_registry(toolbox)
    backend = ScriptedChain(_good_plan_json())
    session = DirectorSession(registry, store, "proj_e2e", backend, source_duration_s=SOURCE_DUR)
    from director_os.agent_runtime.story_map import build_story_map

    transcript = {
        "words": [
            {"text": "w", "start_s": t, "end_s": t + 0.4, "speaker": "Officer"}
            for t in range(0, 2600, 60)
        ],
        "segments": [
            {"speaker": "Officer", "start_s": t, "end_s": t + 18, "text": f"minute {t}"}
            for t in range(0, 2600, 60)
        ],
    }
    buckets = build_story_map(transcript, SOURCE_DUR)
    plan, err = session.plan_macro(buckets, {"subject_name": "Eustis officer"})
    assert plan is not None, err
    assert backend.complete_json_calls == 1

    timeline, report = session.execute(plan, scaffold)
    assert report.ok, report.failures
    ok, failures = invariants_ok(timeline, SOURCE_DUR)
    assert ok, [f.detail for f in failures]
    assert report.composition is not None
    assert report.composition.raw_segment_count >= 8
    assert report.composition.narrator_segment_count >= 6
    assert "Cold Open" in report.composition.acts


def test_director_session_keeps_prompting_while_under_target(store):
    """The loop may NOT die at 1-2 tool calls while invariants are unmet."""
    scaffold = _scaffold()
    toolbox = DirectorToolbox(scaffold, "proj_loop")
    registry = build_director_registry(toolbox)

    # Phase A returns a TRUNCATED plan (2 clips, 1 narration, 75s)
    truncated = json.dumps(
        {
            "title": "truncated",
            "acts": [
                {
                    "name": "Cold Open",
                    "blocks": [
                        {"type": "raw", "source_start_s": 60.0, "dur_s": 45.0},
                        {
                            "type": "narration",
                            "script": "A short look at what happened.",
                            "dur_s": 15.0,
                        },
                        {"type": "raw", "source_start_s": 2000.0, "dur_s": 30.0},
                    ],
                }
            ],
            "cards": [],
        }
    )
    # refinements: two rounds of "add more material", then stop
    add_more = [
        {
            "tool": "timeline.insert_raw_clip",
            "args": {"source_start_s": 900.0 + i * 180.0, "dur_s": 60.0, "act_name": "Act 3"},
        }
        for i in range(8)
    ]
    narrations = [
        {
            "tool": "timeline.add_narration",
            "args": {
                "script_text": f"Bridge {i}: the story turns on what the camera keeps showing.",
                "dur_s": 12.0,
                "act_name": f"Act {i + 1}",
            },
        }
        for i in range(6)
    ]
    backend = ScriptedChain(truncated, proposals=[add_more + narrations, [], []])
    session = DirectorSession(
        registry, store, "proj_loop", backend, source_duration_s=SOURCE_DUR, max_turns=30
    )
    plan, _err = session.plan_macro([], {"subject_name": None})
    assert plan is not None
    timeline, report = session.execute(plan, scaffold)
    # the truncated plan was repaired in-loop to full-arc coverage
    assert backend.propose_calls >= 1, "loop must keep proposing while under target"
    ok, failures = invariants_ok(timeline, SOURCE_DUR)
    assert ok, [f.detail for f in failures]


def test_macro_plan_model_bounds():
    plan = parse_macro_plan(_good_plan_json())
    assert isinstance(plan, MacroPlan)
    assert plan.total_dur_s > 600.0
    first, last = plan.clip_span(SOURCE_DUR)
    assert first is not None and first < SOURCE_DUR * 0.15
    assert last is not None and last > SOURCE_DUR * 0.75
