"""Guardrail 4 (PLAN §9.4): dead-air cuts use speech + audio-event + motion
intervals, never raw volume (highway noise would defeat volume thresholds)."""

from __future__ import annotations

from director_os.audio_engine.speech_trim import anchors_from_inputs, plan_cuts


def test_volume_noise_is_ignored_by_construction():
    """No 'loudness' input exists in the API — only semantic interval sources."""
    anchors = anchors_from_inputs(
        speech=[(10.0, 20.0)],
        audio_events=[(21.5, 23.0)],
        motion=[(0.0, 5.0)],
    )
    assert (0.0, 5.0) in anchors
    assert (10.0, 20.0) in anchors
    assert (21.5, 23.0) in anchors


def test_small_gaps_merge_long_gaps_cut():
    cuts = plan_cuts(
        keep_anchors=[(10.0, 20.0), (21.5, 23.0), (0.0, 5.0)],
        duration_s=30.0,
        min_gap_s=1.8,
        pad_before_s=0.6,
        pad_after_s=0.8,
    )
    # motion keep (0-5.8) and the merged speech+shout keep (9.4-23.8) -> two cuts
    assert cuts == [(5.8, 9.4), (23.8, 30.0)]


def test_breathing_pads_are_kept():
    cuts = plan_cuts(
        [(10.0, 20.0)],
        duration_s=30.0,
        min_gap_s=1.8,
        pad_before_s=0.6,
        pad_after_s=0.8,
    )
    assert cuts == [(0.0, 9.4), (20.8, 30.0)]


def test_no_anchors_means_full_cut():
    assert plan_cuts([], duration_s=30.0) == [(0.0, 30.0)]
