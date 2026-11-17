"""Retention scoring (issue #19) — the Critic Agent's story-first lens.

A timeline that is technically clean can still be a retention disaster: no
hook, one 40% dead middle, no open loops, no ending momentum. The Critic
scores every section layout on:

- HOOK: something intentional inside the first 8 seconds
- ACTS: clear Beginning -> Middle -> End coverage
- MOMENTUM: no single section swallows the runtime; sections trend tighter
  into the climax, not looser
- OPEN LOOPS: tease/next-up markers mid-video (the reason viewers stay)
- DEAD SPANS: unexplained gaps between sections

Pure heuristics on section metadata — the VLM Critic adds judgment on top;
this floor is deterministic and testable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

HOOK_WINDOW_S = 8.0
DEAD_SPAN_S = 10.0
HOG_SECTION_FRAC = 0.40  # one section owning >40% of runtime = momentum killer
HOOK_KINDS = {"hook", "cold_open", "tease", "cold-open"}
OPEN_LOOP_KINDS = {"tease", "open_loop", "next_up", "cliffhanger"}
END_KINDS = {"outcome", "case_outcome", "wrap", "outro", "resolution"}
BEGIN_KINDS = {"hook", "cold_open", "tease", "location_intro", "setup", "intro"}


class SectionSpan(BaseModel):
    """One timeline section as the critic sees it."""

    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    kind: str = ""
    title: str = ""
    tease_next: bool = False  # section ends by opening a loop

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class RetentionReport(BaseModel):
    score: int = Field(ge=0, le=100)
    verdict: str
    reasons: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)


def score_retention(sections: list[SectionSpan]) -> RetentionReport:
    """Deterministic story-first score of a section layout (0-100)."""
    if not sections:
        return RetentionReport(score=0, verdict="empty timeline — nothing to retain")
    sections = sorted(sections, key=lambda s: s.start_s)
    total = max(s.end_s for s in sections) - sections[0].start_s
    if total <= 0:
        return RetentionReport(score=0, verdict="zero-length timeline")
    reasons: list[str] = []
    penalties: list[str] = []
    score = 50

    # 1. HOOK — something intentional inside the first 8 seconds
    first = sections[0]
    hook = first.start_s <= HOOK_WINDOW_S and (
        first.kind.lower() in HOOK_KINDS or "hook" in first.title.lower()
    )
    if hook:
        score += 20
        reasons.append(
            f"hook in first {HOOK_WINDOW_S:g}s ({first.kind or first.title or 'section'})"
        )
    else:
        penalties.append(f"no hook inside the first {HOOK_WINDOW_S:g}s — cold open missing")

    # 2. ACTS — beginning / middle / end coverage
    kinds = {s.kind.lower() for s in sections if s.kind}
    has_begin = bool(kinds & BEGIN_KINDS)
    has_end = bool(kinds & END_KINDS)
    has_middle = len(sections) >= 3
    if has_begin and has_middle and has_end:
        score += 15
        reasons.append("beginning -> middle -> end all present")
    else:
        missing = [
            n
            for n, ok in (("beginning", has_begin), ("middle", has_middle), ("end", has_end))
            if not ok
        ]
        penalties.append(f"acts incomplete: missing {', '.join(missing)}")

    # 3. MOMENTUM — no section hogs the runtime
    hog = max(sections, key=lambda s: s.duration_s)
    hog_frac = hog.duration_s / total
    if hog_frac > 0.60:
        score -= 25
        label = hog.title or hog.kind or str(hog.start_s)
        penalties.append(f"section '{label}' owns {hog_frac:.0%} of runtime — momentum dies")
    elif hog_frac > HOG_SECTION_FRAC:
        score -= 15
        label = hog.title or hog.kind or str(hog.start_s)
        penalties.append(f"section '{label}' owns {hog_frac:.0%} of runtime — pacing sags")
    else:
        score += 5
        reasons.append(f"section rhythm holds (longest is {hog_frac:.0%} of runtime)")

    # 4. OPEN LOOPS — tease/next-up mid-video
    loops = sum(1 for s in sections[:-1] if s.tease_next or s.kind.lower() in OPEN_LOOP_KINDS)
    if loops:
        score += 10
        reasons.append(f"{loops} open loop(s) mid-timeline")
    else:
        penalties.append("no open loops — nothing pulls the viewer past the middle")

    # 5. DEAD SPANS — unexplained gaps
    gaps = [
        (b.start_s - a.end_s, a, b)
        for a, b in zip(sections, sections[1:], strict=False)
        if b.start_s - a.end_s > DEAD_SPAN_S
    ]
    if gaps:
        score -= 10
        worst = max(gaps, key=lambda g: g[0])
        nxt = worst[2].title or worst[2].kind or "next section"
        penalties.append(f"dead span {worst[0]:.1f}s before '{nxt}'")
    else:
        score += 5
        reasons.append("no dead spans between sections")

    score = max(0, min(100, score))
    verdict = (
        "shippable" if score >= 70 else ("needs a re-cut" if score >= 45 else "flat — restructure")
    )
    return RetentionReport(score=score, verdict=verdict, reasons=reasons, penalties=penalties)
