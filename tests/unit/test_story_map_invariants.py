"""Story Map + plan invariants + deterministic 6-act fallback (M5 postmortem gates).

The Eustis failure (59s master from a 43-min source) is pinned here as
regression tests: the story map must condense the FULL timeline, the
invariants must reject truncated plans, and the deterministic fallback must
produce a full-arc 600s+ documentary with no LLM at all.
"""

from __future__ import annotations

import pytest
from director_os.agent_runtime.fallback_cut import (
    build_fallback_documentary,
    plan_fallback_blocks,
)
from director_os.agent_runtime.plan_invariants import (
    audit_retention,
    coverage_span,
    invariants_ok,
    timeline_composition,
)
from director_os.agent_runtime.story_map import (
    build_story_map,
    render_story_map,
    story_map_prompt,
)
from director_os.otio_engine.timeline import build_timeline


def _transcript(minutes: int = 45) -> dict:
    """Synthetic diarized transcript spread across the whole runtime."""
    words, segments = [], []
    for m in range(minutes):
        t = m * 60.0
        speaker = "Officer" if m % 2 == 0 else "Suspect"
        text = f"minute {m} admission about the stop and the breath test"
        words.append({"text": "word", "start_s": t, "end_s": t + 0.5, "speaker": speaker})
        segments.append({"speaker": speaker, "start_s": t, "end_s": t + 20.0, "text": text})
    return {"words": words, "segments": segments}


# ------------------------------------------------------------------ story map
def test_story_map_covers_every_minute_with_speakers():
    buckets = build_story_map(_transcript(45), source_duration_s=2700.0)
    assert len(buckets) == 45
    assert all(b.has_signal for b in buckets)
    rendered = render_story_map(buckets)
    assert "[00:00" in rendered and "[44:00" in rendered
    assert "Officer:" in rendered and "Suspect:" in rendered


def test_story_map_stays_compact_for_45_minutes():
    rendered = render_story_map(build_story_map(_transcript(45), 2700.0))
    # ~2.5k token budget => ~10k chars ceiling for the full index
    assert len(rendered) < 12_000
    prompt = story_map_prompt(build_story_map(_transcript(45), 2700.0), 2700.0)
    assert "Full-timeline Story Index" in prompt and "45 min raw" in prompt


def test_story_map_empty_transcript_degrades_cleanly():
    buckets = build_story_map({"words": [], "segments": []}, 600.0)
    assert render_story_map(buckets).startswith("(no perception signal")


# ------------------------------------------------------------------ invariants
def test_invariants_reject_the_eustis_failure_shape():
    """2 raw clips + 1 narration + 59s from a 43-min source = 4 failures."""
    from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
    from director_os.otio_engine.timeline import ClipSpec, TimelineSpec, TrackSpec

    spec = TimelineSpec(
        name="bad",
        fps=30,
        width=1920,
        height=1080,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="r1", source_start_s=60.0, dur_s=29.5, meta=RawClipMeta(asset_id="a")
                    ),
                    ClipSpec(
                        name="n1",
                        source_start_s=60.0,
                        dur_s=29.5,
                        meta=NarratorClipMeta(script_text="short hook"),
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[]),
        ],
    )
    timeline = build_timeline(spec)
    ok, failures = invariants_ok(timeline, 2627.0)
    assert not ok
    codes = {f.code for f in failures}
    assert codes == {
        "final_duration_floor",
        "narrator_floor",
        "raw_segment_floor",
        "coverage_span",
        "hook_late",  # hook-first law: narration at 29.5s is already too late
    }


def test_audit_retention_report_shape():
    from director_os.otio_engine.contract import RawClipMeta
    from director_os.otio_engine.timeline import ClipSpec, TimelineSpec, TrackSpec

    spec = TimelineSpec(
        name="bad",
        fps=30,
        width=1920,
        height=1080,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="r1", source_start_s=60.0, dur_s=59.0, meta=RawClipMeta(asset_id="a")
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[]),
        ],
    )
    report = audit_retention(build_timeline(spec), 2627.0)
    assert not report.ok
    names = {c.name for c in report.checks}
    assert "final_duration_floor" in names and "coverage_span" in names


# ------------------------------------------------------------------ fallback
@pytest.fixture(scope="module")
def fallback_timeline():
    return build_timeline(
        build_fallback_documentary(
            asset_id="asset_eustis",
            source_duration_s=2627.0,
            transcript=_transcript(44),
            fps=30,
            width=1920,
            height=1080,
        )
    )


def test_fallback_blocks_full_arc_structure():
    blocks = plan_fallback_blocks(2627.0, _transcript(44))
    acts = [b.act for b in blocks if b.kind == "narration"]
    assert acts[0] == "Cold Open"
    assert acts[-1] == "Act 6: Verified Outcome"
    assert len(acts) >= 10  # v0.3 narration law: 10-14 commentary blocks
    raw_starts = [b.source_start_s for b in blocks if b.kind == "raw"]
    assert min(raw_starts) < 2627.0 * 0.15
    assert max(raw_starts) > 2627.0 * 0.60


def test_fallback_passes_every_retention_invariant(fallback_timeline):
    comp = timeline_composition(fallback_timeline)
    assert comp.final_duration_s >= 600.0
    assert comp.narrator_segment_count >= 9  # v0.3: rich commentary floor
    assert comp.narration_duration_s / comp.final_duration_s >= 0.30
    assert comp.narration_duration_s / comp.final_duration_s <= 0.52
    assert comp.raw_segment_count >= 8
    assert coverage_span(comp, 2627.0) >= 0.70
    ok, failures = invariants_ok(fallback_timeline, 2627.0)
    assert ok, [f.detail for f in failures]
    report = audit_retention(fallback_timeline, 2627.0)
    assert report.ok, [c.detail for c in report.failures]


def _short_transcript() -> dict:
    words = []
    for start, spk, text in [
        (6, "Officer", "Unit one to dispatch we have a code four"),
        (60, "Officer", "dispatch copies show unit one en route"),
        (140, "Officer", "step out of the vehicle keep your hands"),
        (225, "Suspect", "I got off work there is nothing in the car"),
        (300, "Officer", "you are being detained for a field sobriety test"),
        (355, "Suspect", "I want my lawyer on the phone right now"),
        (410, "Officer", "stay right there the K nine unit is on the way"),
        (445, "Suspect", "I did nothing wrong call my lawyer"),
    ]:
        for i, w in enumerate(text.split()):
            s = start + i * 0.35
            words.append({"text": w, "start_s": s, "end_s": s + 0.3, "speaker": spk})
    return {"words": words, "segments": []}


def test_fallback_short_documentary_respects_narration_ratio_law():
    """The any-length law at 8 minutes: the fixed long bridges used to carry
    ~108s of narration onto a ~160s master (ratio 0.67 > 0.52) and the
    deterministic fallback failed its own retention pre-flight."""
    from director_os.otio_engine.timeline import build_timeline

    tl = build_timeline(
        build_fallback_documentary(
            asset_id="asset_short",
            source_duration_s=480.0,
            transcript=_short_transcript(),
            fps=30,
            width=640,
            height=360,
        )
    )
    comp = timeline_composition(tl)
    ratio = comp.narration_duration_s / comp.final_duration_s
    assert 0.30 <= ratio <= 0.52, (
        f"ratio {ratio:.2f} ({comp.narration_duration_s:.0f}s of {comp.final_duration_s:.0f}s)"
    )
    report = audit_retention(tl, 480.0)
    assert report.ok, [c.detail for c in report.failures]


def test_fallback_cards_present(fallback_timeline):
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import track_by_name

    overlay = track_by_name(fallback_timeline, "V2_overlay_cards")
    templates = [
        str(i.metadata.get("director_os", {}).get("template"))
        for i in overlay
        if not hasattr(i, "source_range") or i.source_range is not None
    ]
    joined = " ".join(templates)
    assert "location_intro" in joined and "case_outcome" in joined
    _ = master_track  # compiler accepts the timeline


def test_fallback_act_markers_ride_metadata(fallback_timeline):
    comp = timeline_composition(fallback_timeline)
    assert comp.acts and comp.acts[0] == "Cold Open"
    assert any("Act 6" in a for a in comp.acts)
