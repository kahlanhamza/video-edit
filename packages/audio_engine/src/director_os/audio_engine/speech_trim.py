"""Speech-aware dead-air trimming — PLAN guardrail 4.

Volume-based silence cutting (auto-editor style) fails on bodycams: highway traffic
and cruiser A/C keep the meter alive. Cuts are therefore planned from the union of
speech, audio-event and motion intervals — never from loudness alone.
"""

from __future__ import annotations


def anchors_from_inputs(
    speech: list[tuple[float, float]],
    audio_events: list[tuple[float, float]] | None = None,
    motion: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Union of the interval sources that justify keeping footage."""
    merged: list[tuple[float, float]] = []
    for interval in list(speech) + list(audio_events or []) + list(motion or []):
        s, e = interval
        if e <= s:
            continue
        merged.append((s, e))
    return _merge(merged)


def plan_cuts(
    keep_anchors: list[tuple[float, float]],
    duration_s: float,
    min_gap_s: float = 1.8,
    pad_before_s: float = 0.6,
    pad_after_s: float = 0.8,
) -> list[tuple[float, float]]:
    """Return the dead-air cut list [(start, end), ...] to remove.

    - anchors whose padded gap < min_gap_s are merged (a 1.2s pause is breath, not dead air)
    - each kept range is expanded by pad_before/after and clamped to the media
    - the complement is the cut list
    """
    if duration_s <= 0:
        return []

    padded = [
        (max(0.0, s - pad_before_s), min(duration_s, e + pad_after_s))
        for s, e in keep_anchors
        if e > s
    ]
    merged = _merge_gap_aware(padded, min_gap_s)

    cuts: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            cuts.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration_s:
        cuts.append((cursor, duration_s))
    return [(s, e) for s, e in cuts if e - s > 0.01]


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def _merge_gap_aware(
    intervals: list[tuple[float, float]], min_gap_s: float
) -> list[tuple[float, float]]:
    merged = _merge(intervals)
    if min_gap_s <= 0:
        return merged
    out: list[tuple[float, float]] = []
    for s, e in merged:
        if out and s - out[-1][1] < min_gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
