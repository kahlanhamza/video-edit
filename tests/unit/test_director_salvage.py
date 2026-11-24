"""Salvage-before-scrap law for the Director's timeline.

The owner watched the stock fallback narration ship on a real 21-tool-call
director run because the audit nit-picked MECHANICAL violations (undersize
narration durations, raw-before-narration hook). These tests pin the three
repairs:

1. `_plan_problems` keep-band CAP: a plan may not exceed the doc-scale keep
   band (28-35% of the tape) — the live v23 plan shipped 35s+ raws that died
   in the audit AFTER a full 2-minute execute.
2. `plan_macro` re-flies the pre-flight AFTER critic/quote refinement (the
   refiners can mutate an accepted plan into an auditable violation) and
   retries with the concrete feedback.
3. `salvage_director_timeline` fixes the mechanical violations ON the
   timeline (word-law retime with b-roll window widening, hook ordering)
   so the director's program survives instead of the stock fallback.
"""

from __future__ import annotations

import director_os.agent_runtime.director_session as ds
import pytest
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.director_session import (
    DirectorSession,
    narr_est_s,
    parse_macro_plan,
    salvage_director_timeline,
)
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
from director_os.otio_engine.compiler import master_track
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from test_director_session import SOURCE_DUR, _good_plan_json, _script_n

# snap-decisions-free transcript: utterance snapping is not what this test
# pins (the critic-refine recheck is) — silence keeps the plan body fixed
SILENT_TRANSCRIPT = {"words": [], "segments": []}


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


def _kinds(track) -> list[str]:
    return [
        (it.metadata.get("director_os") or {}).get("kind", "raw")
        for it in track
        if it.source_range is not None
    ]


def test_salvage_retimes_narration_to_word_law_and_widens_broll():
    """60 words (~24s at the word law) labeled 10s: salvage widens both the
    clip and its muted b-roll window — the story never shrinks to a stale
    window that was itself built from the undersize dur_s."""
    toolbox = DirectorToolbox(_scaffold(), "p_salvage")
    toolbox.insert_narration_block(
        _script_n(
            60,
            "The tape shows the officer explaining himself while the recorder "
            "keeps catching details the report later contradicted in full and "
            "on the record for everyone watching this case unfold today.",
        ),
        dur_s=10.0,
    )
    track = master_track(toolbox.timeline)
    narr = [it for it in track if (it.metadata.get("director_os") or {}).get("kind") == "narration"]
    assert len(narr) == 1
    assert narr[0].source_range.duration.to_seconds() == 10.0

    timeline, notes = salvage_director_timeline(toolbox.timeline, SOURCE_DUR)
    track = master_track(timeline)
    narr = [it for it in track if (it.metadata.get("director_os") or {}).get("kind") == "narration"]
    est = narr_est_s(str(narr[0].metadata["director_os"]["script_text"]))
    got = narr[0].source_range.duration.to_seconds()
    assert abs(got - est) < 1.0 / 30.0 + 1e-6, (got, est)
    b0, b1 = narr[0].metadata["director_os"]["broll_range"]
    assert abs((b1 - b0) - got) < 1e-6
    assert any("word-law" in n for n in notes), notes


def test_salvage_moves_hook_narration_to_open_the_master():
    """raw(4s) then narration (the live v23 shape, hook start 35s): the first
    narration block is moved to index 0 so the 8s hook law holds."""
    toolbox = DirectorToolbox(_scaffold(), "p_hook")
    toolbox.insert_raw_clip(source_start_s=1660.0, dur_s=4.0)
    toolbox.insert_narration_block(
        _script_n(
            50,
            "July fourth twenty twenty two. A bodycam rolls on a quiet lake "
            "town and catches a stop that turns into a case on tape tonight.",
        ),
        dur_s=20.0,
    )
    assert _kinds(master_track(toolbox.timeline))[-1] == "narration"

    timeline, notes = salvage_director_timeline(toolbox.timeline, SOURCE_DUR)
    kinds = _kinds(master_track(timeline))
    assert kinds[0] == "narration", kinds
    assert any("hook law" in n for n in notes), notes


def test_salvage_leaves_a_compliant_timeline_untouched():
    """A narration-first timeline at word-law durations gets ZERO notes —
    salvage is a surgical pass, not a rewrite."""
    toolbox = DirectorToolbox(_scaffold(), "p_clean")
    script = _script_n(
        50,
        "A calm hook that already rides the exact word law so the salvage "
        "pass must find nothing to fix in this timeline at all anywhere.",
    )
    toolbox.insert_narration_block(script, dur_s=narr_est_s(script), at_index=0)
    before = [it.source_range.duration.value for it in master_track(toolbox.timeline)]

    timeline, notes = salvage_director_timeline(toolbox.timeline, SOURCE_DUR)
    assert notes == []
    after = [it.source_range.duration.value for it in master_track(timeline)]
    assert after == before


# ------------------------------------------------- keep-band cap + post-refine
class RetryChain:
    """complete_json always returns the same plan; counts calls + keeps users."""

    provider = "retry_chain"
    active_label = "scripted:retry"
    model = "scripted:retry"

    def __init__(self, plan_json: str):
        self._plan_json = plan_json
        self.calls = 0
        self.users: list[str] = []

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        self.calls += 1
        self.users.append(user)
        return self._plan_json

    def propose(self, state_summary: str, role: str):
        return []

    def record_results(self, steps, results) -> None:
        pass

    def reset_history(self) -> None:
        pass


def _session(backend, tmp_path, **kwargs) -> DirectorSession:
    from director_os.core.db import init_models, make_engine, make_session_factory

    engine = make_engine(f"sqlite:///{tmp_path}/s.db")
    init_models(engine)
    store = CheckpointStore(make_session_factory(engine))
    return DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p_band")),
        store,
        "p_band",
        backend,
        source_duration_s=SOURCE_DUR,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)


def test_over_band_plan_yields_keep_band_problem():
    """A plan inflating raw durations past target_hi (35% of the tape) trips
    the keep-band CAP — with a refusal hint, not just a number. The good plan
    itself stays clean."""
    session = DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p_band2")),
        None,  # type: ignore[arg-type]
        "p_band2",
        RetryChain(_good_plan_json()),
        source_duration_s=SOURCE_DUR,
    )
    good = parse_macro_plan(_good_plan_json())
    assert session._plan_problems(good) == []
    target_hi = session.scale.target_hi
    first_raw = next(b for a in good.acts for b in a.blocks if type(b).__name__ == "PlanRawClip")
    first_raw.dur_s += max(1.0, target_hi - session._honest_total(good) + 60.0)
    problems = session._plan_problems(good)
    assert problems, "an over-band plan must trip the cap"
    assert any("keep band" in p and f"{target_hi:.0f}s" in p for p in problems), problems


def test_plan_macro_reflies_preflight_after_refine_and_retries_with_feedback(tmp_path, monkeypatch):
    """The refiners mutate an accepted plan past the keep band -> plan_macro
    catches it AFTER refinement, retries, and the second plan ships clean —
    the live v23 failure path (audit after a 2-minute execute)."""
    good_json = _good_plan_json()
    backend = RetryChain(good_json)

    def inflating_critic_then_clean(
        plan, transcript, backend_, source_duration_s, problems_fn=None, niche=""
    ):
        if inflating_critic_then_clean.done:  # attempts 2+ behave — the retry converges
            return plan, ""
        inflating_critic_then_clean.done = True
        first_raw = next(
            b for a in plan.acts for b in a.blocks if type(b).__name__ == "PlanRawClip"
        )
        first_raw.dur_s += 300.0  # push the plan past the keep band
        return plan, "critic extended the climax window (in-fixture mutation)"

    inflating_critic_then_clean.done = False

    monkeypatch.setattr(ds, "critic_refine_plan", inflating_critic_then_clean)
    monkeypatch.setattr(ds, "refine_plan_quotes", lambda plan, *a, **k: (plan, ""))
    session = _session(backend, tmp_path, transcript=SILENT_TRANSCRIPT, quote_refine=True)
    plan, err = session.plan_macro([], {"subject_name": "Eustis officer"})

    assert plan is not None, err
    assert session._plan_problems(plan) == [], err
    assert backend.calls == 2, backend.calls
    assert "## Previous attempt failed" in backend.users[1]
    assert "keep band" in backend.users[1]
