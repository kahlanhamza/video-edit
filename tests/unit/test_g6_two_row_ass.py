"""Guardrail 6 (PLAN §9.6): simultaneous speakers ride two lanes (MarginV 46/96),
never clamped onto one row; lanes are internally overlap-free."""

from __future__ import annotations

from director_os.subtitles.ass_builder import (
    LANE1_MARGIN_V,
    LANE2_MARGIN_V,
    CaptionBeat,
    WordTiming,
    _assign_rows,
    build_ass,
)


def beats():
    return [
        CaptionBeat(text="We pulled him over", start_ms=0, end_ms=2000, speaker="Officer Reed"),
        CaptionBeat(text="I was going home officer", start_ms=1000, end_ms=3000, speaker="Suspect"),
        CaptionBeat(
            text="Step out of the vehicle", start_ms=3100, end_ms=5200, speaker="Officer Reed"
        ),
    ]


def test_overlapping_speakers_land_on_different_lanes():
    rows = _assign_rows(beats())
    assert rows[0] != rows[1], "overlapping different speakers must split lanes"
    assert {LANE1_MARGIN_V, LANE2_MARGIN_V} == {46, 96}


def test_ass_contains_both_lane_margins():
    ass = build_ass(beats())
    assert "MarginV" in ass
    assert str(LANE1_MARGIN_V) in ass and str(LANE2_MARGIN_V) in ass


def test_lanes_are_overlap_free():
    # build_ass raises on any intra-lane overlap — call must succeed cleanly
    ass = build_ass(beats())
    assert ass.startswith("[Script Info]")


def test_karaoke_tags_from_word_timings():
    b = CaptionBeat(
        text="hands where I can see them",
        start_ms=0,
        end_ms=2000,
        speaker="Officer Reed",
        words=[
            WordTiming(word="hands", start_ms=0, end_ms=500),
            WordTiming(word="where", start_ms=500, end_ms=900),
            WordTiming(word="I", start_ms=900, end_ms=1100),
            WordTiming(word="can", start_ms=1100, end_ms=1400),
            WordTiming(word="see", start_ms=1400, end_ms=1700),
            WordTiming(word="them", start_ms=1700, end_ms=2000),
        ],
    )
    ass = build_ass([b])
    assert r"\k" in ass


def test_profanity_masked_by_default():
    b = CaptionBeat(text="what the shit is this", start_ms=0, end_ms=1000, speaker="Suspect")
    ass = build_ass([b])
    assert "shit" not in ass
    assert "*" in ass
