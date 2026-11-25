"""DirectorSession — the 2-phase Story Map -> Full-Arc Execution flow (M5).

Phase A (macro plan JSON): lenient parsing, v0.3-law pre-flight (10-14
narration blocks, 30-52% narration ratio, full-arc coverage). Phase B: plan
expansion into ordered tool steps, then the refinement loop that may NOT
terminate while duration invariants are unmet. End-to-end runs against the
REAL tool registry + checkpoint store on sqlite with a scripted backend.
"""

from __future__ import annotations

import json

import pytest
from director_os.agent_runtime.agents import ToolStep
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


def _script_n(n: int, seed: str) -> str:
    """A narration script with EXACTLY n words (the word-count law fixture)."""
    words = seed.split()
    filler = [
        "the",
        "record",
        "shows",
        "the",
        "camera",
        "keeps",
        "rolling",
        "and",
        "every",
        "detail",
        "lands",
        "on",
        "the",
        "tape",
    ]
    i = 0
    while len(words) < n:
        words.append(filler[i % len(filler)])
        i += 1
    return " ".join(words[:n])


# transcript the fixture windows are built against: 40s utterances each minute
FIXTURE_TRANSCRIPT = {
    "words": [
        {"text": "w", "start_s": float(t), "end_s": float(t) + 0.4, "speaker": "Officer"}
        for t in range(0, 2600, 60)
    ],
    "segments": [
        {
            "speaker": "Officer",
            "start_s": float(t),
            "end_s": float(t) + 40.0,
            "text": f"minute {t} dialogue about the case",
        }
        for t in range(0, 2600, 60)
    ],
}


def _good_plan_json() -> str:
    """A word-count-law plan (hook-first): cold open = 4s silent-gap sting +
    50-word hook narration (<=8s in, context <=32s), 12 mid blocks of 60 words
    (~23.9s each at 2.6 w/s), an 85-word outro, and 14 raw windows aligned to
    real utterances (41.1s snapped pieces) — ~885s total, ratio ~0.40
    (inside the 736-900s keep band)."""
    acts = [
        {
            "name": "Cold Open",
            "blocks": [
                # a silent gap between utterances (1660..1680): snaps to itself
                {"type": "raw", "source_start_s": 1662.0, "dur_s": 4.0},
                {
                    "type": "narration",
                    "script": _script_n(
                        50,
                        "July 4th 2022 Eustis Florida. A bodycam is rolling on a "
                        "police officer and what it catches over the next forty "
                        "minutes turns a routine holiday call into a criminal case "
                        "the report never saw coming. Every second of it is on tape.",
                    ),
                    "dur_s": 20.0,
                },
                {"type": "raw", "source_start_s": 1925.0, "dur_s": 45.0},
            ],
        }
    ]
    for i in range(1, 7):
        u = 60 + (i - 1) * 420
        blocks = [
            {
                "type": "narration",
                "script": _script_n(
                    60,
                    f"Act {i} commentary on what the camera just showed and what the "
                    "verified record says about it, plainly",
                ),
                "dur_s": 22.0,
            },
            {"type": "raw", "source_start_s": float(u), "dur_s": 42.0},
            {"type": "raw", "source_start_s": float(u + 180), "dur_s": 42.0},
            {
                "type": "narration",
                "script": _script_n(
                    60,
                    f"Act {i} closing analysis of the contradiction the tape just "
                    "proved, said calmly and on the record",
                ),
                "dur_s": 22.0,
            },
        ]
        if i == 6:
            blocks.append(
                {
                    "type": "narration",
                    "script": _script_n(
                        85,
                        "The record closes the loop: booking finishes, the tests come "
                        "back, and the case files state the verified outcome in plain "
                        "terms for everyone involved in this stop",
                    ),
                    "dur_s": 32.0,
                }
            )
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
    assert 10 <= plan.narration_count <= 14
    assert plan.raw_count >= 8
    assert plan.total_dur_s > 700.0


def test_parse_macro_plan_rejects_garbage():
    with pytest.raises(ValueError):
        parse_macro_plan("the documentary should open with a bang")
    with pytest.raises(ValueError):
        parse_macro_plan('{"title": "x"}')  # no acts -> validation error


def test_plan_problems_pass_on_v03_law_plan():
    plan = parse_macro_plan(_good_plan_json())
    session = DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p1")),
        None,  # type: ignore[arg-type]
        "p1",
        ScriptedChain(_good_plan_json()),
        source_duration_s=SOURCE_DUR,
    )
    assert session._plan_problems(plan) == []


def test_plan_problems_flag_undercoverage_and_thin_narration():
    plan = parse_macro_plan(_good_plan_json())
    plan.acts = plan.acts[:-1]  # drop the last act -> the ending disappears
    session = DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p1")),
        None,  # type: ignore[arg-type]
        "p1",
        ScriptedChain(_good_plan_json()),
        source_duration_s=SOURCE_DUR,
    )
    problems = session._plan_problems(plan)
    assert any("latest clip ends" in p for p in problems)


def test_plan_problems_flag_ratio_violation():
    plan = parse_macro_plan(_good_plan_json())
    # starve the narration WORDS: dur_s is ignored (computed from word count),
    # so only a thin script can drag the narration ratio under the floor
    for act in plan.acts:
        act.blocks = [
            (
                b.model_copy(update={"script": "A short look at what happened.", "dur_s": 22.0})
                if b.__class__.__name__ == "PlanNarration"
                else b
            )
            for b in act.blocks
        ]
    session = DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p1")),
        None,  # type: ignore[arg-type]
        "p1",
        ScriptedChain(_good_plan_json()),
        source_duration_s=SOURCE_DUR,
    )
    problems = session._plan_problems(plan)
    assert any("narration ratio" in p for p in problems)


# ------------------------------------------------------------------ phase B0
def test_expand_macro_plan_orders_steps_and_computes_card_positions():
    plan = parse_macro_plan(_good_plan_json())
    steps = expand_macro_plan(plan, SOURCE_DUR)
    tools = [s.tool for s in steps]
    # first block is raw -> scaffold consumed by trim
    assert steps[0].tool == "timeline.trim_clip"
    assert steps[0].args["new_start_s"] == 1662.0  # the silent-gap climax sting
    assert "timeline.add_narration" in tools and "timeline.insert_raw_clip" in tools
    card_steps = [s for s in steps if s.tool == "cards.add"]
    assert card_steps[0].args["start_s"] > 60.0  # after cold open + act1 narration
    assert all(s.args["start_s"] >= 0 for s in card_steps)
    # card_anchors FIX: location_intro lands at the TRUE start of act 1
    # (sting + hook + cold raw), not a full act late
    cold_total = sum(
        b.est_s if b.__class__.__name__ == "PlanNarration" else b.dur_s for b in plan.acts[0].blocks
    )
    assert abs(card_steps[0].args["start_s"] - cold_total) < 0.5


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


# ------------------------------------------------------------------ end to end
@pytest.fixture()
def store(tmp_path):
    from director_os.core.db import init_models, make_engine, make_session_factory

    engine = make_engine(f"sqlite:///{tmp_path}/session.db")
    init_models(engine)
    return CheckpointStore(make_session_factory(engine))


def test_director_session_end_to_end_passes_invariants(store):
    from director_os.agent_runtime.story_map import build_story_map

    scaffold = _scaffold()
    toolbox = DirectorToolbox(scaffold, "proj_e2e")
    registry = build_director_registry(toolbox)
    backend = ScriptedChain(_good_plan_json())
    session = DirectorSession(registry, store, "proj_e2e", backend, source_duration_s=SOURCE_DUR)

    transcript = FIXTURE_TRANSCRIPT
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
    assert report.composition.narrator_segment_count >= 9
    assert report.composition.narration_duration_s / report.composition.final_duration_s >= 0.30
    assert "Cold Open" in report.composition.acts


def test_director_session_keeps_prompting_while_under_target(store):
    """The loop may NOT die at 1-2 tool calls while invariants are unmet."""
    scaffold = _scaffold()
    toolbox = DirectorToolbox(scaffold, "proj_loop")
    registry = build_director_registry(toolbox)

    # Phase A returns a TRUNCATED, narration-starved plan (2 raw, 1 narr, 75s)
    # — hook-first ordering (sting + narration first) so the loop's failure is
    # the floors, not the hook law
    truncated = json.dumps(
        {
            "title": "truncated",
            "acts": [
                {
                    "name": "Cold Open",
                    "blocks": [
                        {"type": "raw", "source_start_s": 60.0, "dur_s": 4.0},
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
    # one big repair round: 9 raws + 9 narration blocks -> ratio lands ~0.33
    add_more = [
        {
            "tool": "timeline.insert_raw_clip",
            "args": {"source_start_s": 900.0 + i * 170.0, "dur_s": 52.0, "act_name": "Act 3"},
        }
        for i in range(9)
    ]
    narrations = [
        {
            "tool": "timeline.add_narration",
            "args": {
                "script_text": _script_n(
                    62,
                    f"Bridge {i}: the story turns on what the camera keeps showing and "
                    "the record is about to say so in plain terms before this act closes",
                ),
                "dur_s": 25.0,
                "act_name": f"Act {i + 1}",
            },
        }
        for i in range(9)
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
    assert 10 <= plan.narration_count <= 14
    first, last = plan.clip_span(SOURCE_DUR)
    assert first is not None and first < SOURCE_DUR * 0.15
    assert last is not None and last > SOURCE_DUR * 0.75


def test_execute_refine_false_never_proposes(store):
    """An authored plan (preflight-clean) must not make a single LLM call.

    Measured live: the refine loop proposed on turn 0 even with every
    invariant green, and the model call (no transport timeout) pinned an
    authored-plan tape in epoll_wait across three render windows.
    """
    scaffold = _scaffold()
    toolbox = DirectorToolbox(scaffold, "proj_authored")  # noqa: F841
    registry = build_director_registry(toolbox)
    backend = ScriptedChain(_good_plan_json())
    backend._proposals = []
    session = DirectorSession(
        registry, store, "proj_authored", backend, source_duration_s=SOURCE_DUR
    )
    plan = parse_macro_plan(_good_plan_json())
    assert plan is not None

    timeline, report = session.execute(plan, scaffold, refine=False)
    assert backend.propose_calls == 0
    assert report.turns == 0
    ok, failures = invariants_ok(timeline, SOURCE_DUR)
    assert ok, [f.detail for f in failures]
    assert report.ok, report.failures
    comp = report.composition
    assert comp.raw_segment_count >= 8
    assert comp.narrator_segment_count >= 9


def test_execute_refine_default_still_proposes(store):
    """The default path keeps the refinement loop (regression guard)."""
    scaffold = _scaffold()
    registry = build_director_registry(DirectorToolbox(scaffold, "proj_default"))
    backend = ScriptedChain(_good_plan_json())
    backend._proposals = []
    session = DirectorSession(
        registry, store, "proj_default", backend, source_duration_s=SOURCE_DUR, max_turns=3
    )
    plan = parse_macro_plan(_good_plan_json())
    assert plan is not None
    _timeline, report = session.execute(plan, scaffold)
    assert backend.propose_calls >= 1, "default refine must run the loop"
    assert report.phase in {"refine", "macro"}
