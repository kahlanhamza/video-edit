"""Content safety law — the benign-footage gate (ADR-0013, owner directive 2026-11-24).

The channel ships only benign incident categories (shoplifting, trespassing,
first-amendment audits); extreme violence and gunfire are vetoed at plan
time, before any clip is cut. Pinned here: the case gate refuses off-menu
cases (the old OWI/DUI launch picks would be refused today), transcript
windows carrying violent markers are excluded from raw-clip candidates, and
seed ranges from a previous LLM plan are trimmed off veto ranges too.
"""

from __future__ import annotations

import pytest
from director_os.agent_runtime.content_policy import (
    ContentPolicyError,
    clip_safe_span,
    drop_vetoed,
    evaluate_case,
    first_safe_window,
    vetoed_ranges,
)
from director_os.agent_runtime.fallback_cut import plan_fallback_blocks


# ------------------------------------------------------------------ case gate
@pytest.mark.parametrize(
    "facts",
    [
        "arrested for shoplifting merchandise from a retail store",
        "retail theft investigation, items stolen from the store",
        "trespassing: the man refused to leave the business after being asked",
        "she was banned from the property and would not leave",
        "first amendment audit of the police station lobby",
        "auditor recording the police in public",
    ],
)
def test_on_menu_categories_are_allowed(facts: str):
    verdict = evaluate_case(facts)
    assert verdict.allowed, verdict.reason
    assert verdict.veto_hits == ()
    assert verdict.categories


@pytest.mark.parametrize(
    "facts",
    [
        "OWI stop: impaired driving on the freeway, field sobriety test",
        "DUI arrest after the crash investigation",
        "high-speed pursuit of a stolen vehicle ended in a PIT maneuver",
        "welfare check on an intoxicated man",
    ],
)
def test_off_menu_cases_are_refused_in_strict_mode(facts: str):
    verdict = evaluate_case(facts)
    assert not verdict.allowed
    assert "no on-menu category" in verdict.reason


@pytest.mark.parametrize(
    "marker",
    [
        "the suspect fired shots during the arrest",
        "a gunshot was heard in the parking lot",
        "officer-involved shooting investigation",
        "the victim was killed at the scene",
        "police found a dead body behind the store",
        "the man was stabbed during the argument",
        "a three-hour standoff with the suspect",
        "shots fired inside the mall",
    ],
)
def test_violent_markers_veto_even_on_menu_cases(marker: str):
    verdict = evaluate_case(f"shoplifting arrest: {marker}")
    assert not verdict.allowed
    assert verdict.veto_hits
    assert "veto" in verdict.reason


def test_empty_case_facts_pass_veto_only_law():
    verdict = evaluate_case("")
    assert verdict.allowed
    assert verdict.categories == frozenset()


def test_veto_only_mode_allows_off_menu_cases(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTENT_POLICY_MODE", "veto_only")
    verdict = evaluate_case("OWI stop: impaired driving, field sobriety test")
    assert verdict.allowed


def test_unknown_mode_env_falls_back_to_strict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTENT_POLICY_MODE", "banana")
    verdict = evaluate_case("OWI stop, impaired driving")
    assert not verdict.allowed


# ------------------------------------------------------ transcript veto ranges
def _words(*utterances: tuple[float, str]) -> dict:
    words = []
    for t, text in utterances:
        words.append({"text": text, "start_s": t, "end_s": t + len(text) * 0.05})
    return {"words": words}


def test_vetoed_ranges_flag_only_the_violent_utterance():
    transcript = _words(
        (10.0, "he grabbed merchandise and walked out without paying"),
        (300.0, "shots fired in the back room"),
        (600.0, "he was banned from the store for trespassing"),
    )
    ranges = vetoed_ranges(transcript)
    assert len(ranges) == 1
    a, b, label = ranges[0]
    assert label == "gunfire"
    assert a < 300.0 < b
    # benign utterances stay untouched
    assert b < 600.0 or a > 600.0


def test_vetoed_ranges_empty_for_clean_transcript():
    transcript = _words(
        (5.0, "shoplifting at the target store"), (60.0, "trespassing warning issued")
    )
    assert vetoed_ranges(transcript) == []


def test_drop_vetoed_removes_overlapping_candidates():
    ranges = [(100.0, 110.0, "gunfire")]
    out = drop_vetoed([(50.0, 60.0), (98.0, 120.0), (115.0, 130.0), (200.0, 220.0)], ranges)
    # 50-60 clean; 98-120 straddles the range from the left (trimmed to 2s);
    # 115-130 starts after the range (untouched); 200-220 clean
    assert (50.0, 60.0) in out
    assert (98.0, 120.0) not in out
    assert (200.0, 220.0) in out
    for a, b in out:
        assert not (a < 110.0 and b > 100.0), (a, b)


def test_clip_safe_span_trims_before_and_after():
    ranges = [(100.0, 110.0, "gunfire")]
    # a window straddling from the left keeps its start, ends before the range
    assert clip_safe_span(90.0, 30.0, ranges) == (90.0, 10.0)
    # a window starting inside the range jumps past it and keeps its end
    assert clip_safe_span(105.0, 20.0, ranges) == (110.0, 15.0)
    assert clip_safe_span(100.0, 10.0, ranges) is None
    assert clip_safe_span(200.0, 30.0, ranges) == (200.0, 30.0)


def test_first_safe_window_skips_veto_ranges():
    ranges = [(0.0, 60.0, "gunfire"), (100.0, 150.0, "death")]
    start, dur = first_safe_window(600.0, 24.0, ranges)
    assert start == pytest.approx(60.0)
    assert dur == 24.0


# ------------------------------------------------------------ planner wiring
def _violent_transcript() -> dict:
    """A mostly-benign transcript with ONE violent utterance (out of order!)."""
    words = []
    for m in range(45):
        t = m * 60.0
        words.append(
            {
                "text": f"minute {m} shoplifting admission about the stop",
                "start_s": t,
                "end_s": t + 4.0,
            }
        )
    # violence inside the hook region at t=200s — appended LAST on purpose:
    # out-of-order word lists must not hide the marker (regression: the
    # utterance merger once welded it to the minute-44 utterance)
    words.append({"text": "shots fired near the register", "start_s": 200.0, "end_s": 204.0})
    return {"words": words}


def _clean_transcript() -> dict:
    words = []
    for m in range(45):
        t = m * 60.0
        words.append(
            {"text": f"minute {m} the audit continues in the lobby", "start_s": t, "end_s": t + 4.0}
        )
    return {"words": words}


def test_planner_refuses_violent_case_facts():
    with pytest.raises(ContentPolicyError):
        plan_fallback_blocks(2700.0, {"words": []}, case_facts="OWI impaired driving stop")


def test_planner_refuses_off_menu_case_facts():
    with pytest.raises(ContentPolicyError):
        plan_fallback_blocks(
            2700.0, {"words": []}, case_facts="welfare check on an intoxicated man"
        )


def test_planner_cuts_no_raw_clip_from_a_vetoed_window():
    blocks = plan_fallback_blocks(
        2700.0, _violent_transcript(), case_facts="shoplifting arrest at the retail store"
    )
    raws = [(b.source_start_s, b.source_start_s + b.dur_s) for b in blocks if b.kind == "raw"]
    assert raws, "the plan must still exist for an on-menu case"
    for a, b in raws:
        assert not (a < 205.0 and b > 199.0), (
            f"raw clip {a:.1f}-{b:.1f}s overlaps the vetoed window"
        )


def test_planner_veto_trims_seed_ranges_too():
    # a previous LLM plan seeded a clip straight across the violent window
    blocks = plan_fallback_blocks(
        2700.0,
        _violent_transcript(),
        seed_raw_ranges=[(190.0, 230.0)],
        case_facts="trespassing: he refused to leave the business",
    )
    for blk in blocks:
        if blk.kind != "raw":
            continue
        a, b = blk.source_start_s, blk.source_start_s + blk.dur_s
        assert not (a < 205.0 and b > 199.0), f"seeded raw clip {a:.1f}-{b:.1f}s overlaps veto"


def test_planner_clean_case_still_produces_full_plan():
    blocks = plan_fallback_blocks(
        2700.0, _clean_transcript(), case_facts="first amendment audit of the police station"
    )
    kinds = [b.kind for b in blocks]
    assert kinds.count("narration") >= 6 and kinds.count("raw") >= 8
