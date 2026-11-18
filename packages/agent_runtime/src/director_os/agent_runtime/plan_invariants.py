"""Hard duration & full-arc coverage invariants — the Eustis postmortem gates.

The 59-second master shipped because NOTHING downstream of the Director had
authority to reject it. These invariants are checked in three places:

1. inside the Director session loop (keep proposing while unmet),
2. after the Director plan (reject + retry with failure feedback),
3. Stage 8 QA (`qa.audit_retention`) — failure parks the job in `qa_failed`
   and the master is NEVER delivered.

Rules (raw source >= 1200s / 20 min):
- final_duration_s >= 600.0  (10-minute floor for 40-min sources)
- narrator_segment_count >= 6
- raw_segment_count >= 8
- source_coverage_span >= 0.70 — kept clips must start within the first 15%
  of the raw file and end inside the final 25%, so the arrest/booking ending
  can never be dropped.

Shorter sources get proportional floors (50-65% of raw), never above raw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LONG_SOURCE_S = 1200.0
DOCUMENTARY_MIN_SOURCE_S = 900.0  # >= this gets the 6-act documentary treatment
MIN_FINAL_DURATION_S = 600.0
MIN_NARRATOR_SEGMENTS = 9  # v0.3 narration law: rich commentary, not 15% sprinkles
MIN_RAW_SEGMENTS = 8
MIN_COVERAGE_SPAN = 0.70
HEAD_FRACTION = 0.15  # earliest allowed clip start (as fraction of raw)
TAIL_FRACTION = 0.25  # latest allowed clip end must reach into the final 25%
# narration ratio law (v0.3): 30-52% of the final runtime is narrator commentary
# — monetization-safe rich analysis, never a podcast, never a sprinkle
NARRATION_RATIO_MIN = 0.30
NARRATION_RATIO_MAX = 0.52


@dataclass
class TimelineComposition:
    """What the master track is made of (source-clock view)."""

    final_duration_s: float = 0.0
    narrator_segment_count: int = 0
    narration_duration_s: float = 0.0
    raw_segment_count: int = 0
    first_source_s: float | None = None
    last_source_end_s: float | None = None
    acts: list[str] = field(default_factory=list)


def timeline_composition(timeline) -> TimelineComposition:
    """Read the master track into the invariant inputs (OTIO timeline)."""
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import is_gap, item_duration_s

    comp = TimelineComposition()
    try:
        track = master_track(timeline)
    except Exception:
        return comp
    for item in track:
        if is_gap(item):
            continue
        dur = item_duration_s(item)
        comp.final_duration_s += dur
        meta = item.metadata.get("director_os") or {}
        kind = str(meta.get("kind", "?"))
        if kind == "narration":
            comp.narrator_segment_count += 1
            comp.narration_duration_s += dur
        elif kind == "raw":
            comp.raw_segment_count += 1
            sr = getattr(item, "source_range", None)
            if sr is not None:
                src0 = sr.start_time.to_seconds()
                src1 = src0 + sr.duration.to_seconds()
                comp.first_source_s = (
                    src0 if comp.first_source_s is None else min(comp.first_source_s, src0)
                )
                comp.last_source_end_s = (
                    src1 if comp.last_source_end_s is None else max(comp.last_source_end_s, src1)
                )
        act = str(meta.get("act_name", "") or "")
        if act and act not in comp.acts:
            comp.acts.append(act)
    return comp


def coverage_span(comp: TimelineComposition, source_duration_s: float) -> float:
    """0..1 — how much of the raw clock the kept clips span, head and tail.

    Returns 0.0 when no raw clip exists (a narration-only timeline covers
    nothing); a span of 0.70 means clips reach from the first 15% of the file
    through at least the final 25%+ of it.
    """
    if source_duration_s <= 0 or comp.first_source_s is None or comp.last_source_end_s is None:
        return 0.0
    if comp.last_source_end_s <= comp.first_source_s:
        return 0.0
    return (comp.last_source_end_s - comp.first_source_s) / source_duration_s


@dataclass
class InvariantFailure:
    code: str
    detail: str


def audit_invariants(comp: TimelineComposition, source_duration_s: float) -> list[InvariantFailure]:
    """The mandatory-change gate list; empty list = plan is acceptable."""
    failures: list[InvariantFailure] = []

    if source_duration_s >= LONG_SOURCE_S:
        min_dur = MIN_FINAL_DURATION_S
    else:
        # proportional floor for shorter sources (50-65% of raw), capped
        min_dur = min(source_duration_s * 0.5, MIN_FINAL_DURATION_S)
    if comp.final_duration_s < min_dur:
        failures.append(
            InvariantFailure(
                code="final_duration_floor",
                detail=f"final_duration_s={comp.final_duration_s:.1f} < required {min_dur:.1f}"
                f" (raw={source_duration_s:.0f}s)",
            )
        )

    if comp.narrator_segment_count < MIN_NARRATOR_SEGMENTS:
        failures.append(
            InvariantFailure(
                code="narrator_floor",
                detail=(
                    f"narrator_segment_count={comp.narrator_segment_count} "
                    f"< {MIN_NARRATOR_SEGMENTS}"
                ),
            )
        )
    if source_duration_s >= DOCUMENTARY_MIN_SOURCE_S and comp.final_duration_s > 0:
        ratio = comp.narration_duration_s / comp.final_duration_s
        if not (NARRATION_RATIO_MIN <= ratio <= NARRATION_RATIO_MAX):
            failures.append(
                InvariantFailure(
                    code="narration_ratio",
                    detail=(
                        f"narration_ratio={ratio:.2f} outside "
                        f"{NARRATION_RATIO_MIN:.2f}..{NARRATION_RATIO_MAX:.2f} "
                        f"({comp.narration_duration_s:.0f}s narration of "
                        f"{comp.final_duration_s:.0f}s master)"
                    ),
                )
            )
    if comp.raw_segment_count < MIN_RAW_SEGMENTS:
        failures.append(
            InvariantFailure(
                code="raw_segment_floor",
                detail=f"raw_segment_count={comp.raw_segment_count} < {MIN_RAW_SEGMENTS}",
            )
        )

    if source_duration_s >= LONG_SOURCE_S:
        span = coverage_span(comp, source_duration_s)
        if span < MIN_COVERAGE_SPAN:
            failures.append(
                InvariantFailure(
                    code="coverage_span",
                    detail=f"source_coverage_span={span:.2f} < {MIN_COVERAGE_SPAN:.2f} "
                    f"(clips {comp.first_source_s:.0f}s..{comp.last_source_end_s:.0f}s "
                    f"of {source_duration_s:.0f}s raw — the ending is being dropped)",
                )
            )
    return failures


def invariants_ok(timeline, source_duration_s: float) -> tuple[bool, list[InvariantFailure]]:
    comp = timeline_composition(timeline)
    failures = audit_invariants(comp, source_duration_s)
    return not failures, failures


def audit_retention(
    timeline,
    source_duration_s: float,
    *,
    min_final_duration_s: float = MIN_FINAL_DURATION_S,
    min_narrator_segments: int = MIN_NARRATOR_SEGMENTS,
    min_raw_segments: int = MIN_RAW_SEGMENTS,
    min_coverage_span: float = MIN_COVERAGE_SPAN,
):
    """Full-arc retention audit (PLAN §3.3G `qa.audit_retention`, M5).

    The Eustis postmortem gate: a rendered documentary from a 20+ minute raw
    source that is under 10 minutes, under-narrated, clip-starved, or that
    drops the arrest/booking ending NEVER ships — Stage 8 parks the job in
    `qa_failed` and the master is quarantined. Same floors the Director
    session retried against in Phase B, so planner and QA cannot disagree.
    """
    from director_os.core.qa_probe import AuditCheck, AuditReport

    checks: list[AuditCheck] = []
    try:
        comp = timeline_composition(timeline)
    except Exception as e:  # unreadable timeline: fail loudly, never pass
        return AuditReport(
            path="<timeline>",
            ok=False,
            checks=[AuditCheck(name="timeline_readable", ok=False, detail=str(e))],
        )

    if source_duration_s >= LONG_SOURCE_S:
        floor = min_final_duration_s
    else:
        floor = min(source_duration_s * 0.5, min_final_duration_s)
    checks.append(
        AuditCheck(
            name="final_duration_floor",
            ok=comp.final_duration_s >= floor,
            detail=(
                f"final={comp.final_duration_s:.1f}s (floor {floor:.1f}s, "
                f"raw={source_duration_s:.0f}s)"
            ),
        )
    )
    checks.append(
        AuditCheck(
            name="narrator_floor",
            ok=comp.narrator_segment_count >= min_narrator_segments,
            detail=f"narration blocks={comp.narrator_segment_count} (min {min_narrator_segments})",
        )
    )
    if source_duration_s >= DOCUMENTARY_MIN_SOURCE_S and comp.final_duration_s > 0:
        ratio = comp.narration_duration_s / comp.final_duration_s
        checks.append(
            AuditCheck(
                name="narration_ratio",
                ok=NARRATION_RATIO_MIN <= ratio <= NARRATION_RATIO_MAX,
                detail=(
                    f"narration ratio={ratio:.2f} "
                    f"({comp.narration_duration_s:.0f}s of {comp.final_duration_s:.0f}s; "
                    f"law {NARRATION_RATIO_MIN:.2f}..{NARRATION_RATIO_MAX:.2f})"
                ),
            )
        )
    checks.append(
        AuditCheck(
            name="raw_floor",
            ok=comp.raw_segment_count >= min_raw_segments,
            detail=f"raw clips={comp.raw_segment_count} (min {min_raw_segments})",
        )
    )
    if source_duration_s >= LONG_SOURCE_S:
        span = coverage_span(comp, source_duration_s)
        checks.append(
            AuditCheck(
                name="coverage_span",
                ok=span >= min_coverage_span,
                detail=(
                    f"source span={span:.2f} (min {min_coverage_span:.2f}; clips "
                    f"{comp.first_source_s:.0f}s..{comp.last_source_end_s:.0f}s of "
                    f"{source_duration_s:.0f}s)"
                ),
            )
        )
    return AuditReport(path="<timeline>", ok=all(c.ok for c in checks), checks=checks)
