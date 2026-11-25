"""Content safety law — the benign-footage gate (owner directive, 2026-11-24).

The channel ships ONLY benign incident categories: shoplifting / retail
theft, trespassing, first-amendment audits, and DUI/OWI stops (owner
directive 2026-11-25: the niche is escalated-situation incident docs, not
one vein). Crashes, pursuits, standoffs and gunfire remain off-menu
entirely, and extreme violence and gunfire are never shipped.

Two layers, both driven by one truth: the words in the case facts and the
transcript are the only honest signal available at plan time.

1. VETO layer (always on) — hard refusal on violent markers. Weapon
   discharges, death, and serious violence. The veto list is deliberately
   conservative: taser deployments, cuffs, and physical holds are standard
   arrest content inside the benign categories, so they are NOT vetoed — a
   veto that fires on every arrest vetoes the niche itself.

2. ALLOWLIST layer (strict mode) — case facts must match at least one
   on-menu category. Unclassifiable NON-EMPTY case facts are rejected
   (fail-closed for the production path: an OWI tape can never be planned).
   EMPTY case facts pass the veto only — the veto is what keeps a run safe;
   the allowlist simply needs text to classify (tests and transcript-only
   flows rely on this and it is documented, not silent).

The planner consumes ``vetoed_ranges`` so no raw clip is ever cut from a
window whose transcript contains a violent marker, and ``clip_safe_span``
trims any planned window off a veto range. Muted b-roll riding narration
blocks is not re-anchored in v1 — a violence marker is spoken/audible, so
its footage lives inside speech windows that raw clips draw from; the
residual risk is documented in ADR-0013, not hidden.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

STRICT_MODE = "strict"
VETO_ONLY_MODE = "veto_only"


class ContentPolicyError(RuntimeError):
    """The case fails the content safety law — fail-closed, never planned."""


def _mode() -> str:
    mode = (os.environ.get("CONTENT_POLICY_MODE") or STRICT_MODE).strip().lower()
    if mode not in (STRICT_MODE, VETO_ONLY_MODE):
        return STRICT_MODE
    return mode


# --- VETO layer: extreme violence and gunfire (word-boundary regexes) ------
# Conservative by design: gun discharges, death, and serious violence only.
# "shot" alone is NOT a marker ("shot down my idea"); "beating" alone is NOT
# a marker ("beating on the door"); tasers/cuffs/holds are NOT markers.
VETO_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "gunfire",
        r"\bgun ?shots?\b|\bshots? fired\b|\b(fired|discharged) (shots|rounds)\b"
        r"|\bgunfire\b|\bgun ?battle\b",
    ),
    (
        "weapon discharge",
        r"\b(fired|discharged) (his|her|their|the) (gun|weapon|firearm)\b"
        r"|\bweapon (was )?discharged\b",
    ),
    (
        "shooting",
        r"\bshooting\b|\bofficer[- ]involved shooting\b|\bactive shooter\b|\bshoot ?out\b",
    ),
    (
        "shot at",
        r"\bshot at\b"
        r"|\bshot (him|her|them|the (man|woman|officer|deputy|trooper|suspect|victim))\b",
    ),
    (
        "death",
        r"\bhomicide\b|\bmurder(ed)?\b|\bmanslaughter\b|\bkilled\b|\bfatal(ly)?\b|\bdeceased\b",
    ),
    ("dead body", r"\bdead (body|man|woman|person|guy)\b|\bbody (was )?found\b"),
    ("stabbing", r"\bstab(bed|bing)?\b"),
    ("standoff", r"\bstandoff\b|\bhostage\b|\bhome ?invasion\b"),
    ("swat raid", r"\bswat (team|raid|tactical)\b|\bserve(d)? a( no-)?knock\b"),
    (
        "serious assault",
        r"\bsevere(ly)? (beaten|injur(ed|es))\b|\bbeaten unconscious\b|\bbroken (bones|ribs)\b",
    ),
    ("siege", r"\bbarricad(ed)? (subject|himself|herself|in)\b"),
)

# --- ALLOWLIST layer: the on-menu benign categories ------------------------
SAFE_CATEGORIES: dict[str, str] = {
    "shoplifting": (
        r"shoplift|retail theft|stealing from (the )?store|stole(n)? (merchandise|items|goods)"
        r"|petty theft|larceny|theft from (a|the) (store|retailer)"
    ),
    "trespassing": (
        r"trespass|refus(e|ed|es|ing) to (leave|exit)|refuses? to leave|would not leave"
        r"|banned from|order(ed)? to leave|stay(ed)? away"
    ),
    "first_amendment_audit": (
        r"first amendment|auditor"
        r"|audit(ing|ed)? (the )?(police|station|lobby|city hall|officers|department)"
        r"|filming (in|at) public|recording (the )?police|filming (the )?police"
    ),
    # owner directive 2026-11-25: the menu is the ESCALATED-SITUATION niche,
    # not just custody-avoidance incidents — a DUI/OWI stop (field sobriety,
    # breath testing, the arrest itself) is ordinary arrest content: no
    # violence, no pursuit, no crash. The veto layer above stays the hard
    # wall for anything that turns violent.
    "dui_owi_stop": (
        r"\bdui\b|\bowi\b|\bdwi\b|driving under the influence"
        r"|drunk (driving|cop)|impaired driving|breath(alizer)? tests?"
        r"|field sobriety|\bsobriety\b|\bbac\b|blood alcohol"
    ),
}

_COMPILED_VETO = tuple((label, re.compile(pat, re.IGNORECASE)) for label, pat in VETO_PATTERNS)
_COMPILED_SAFE = {name: re.compile(pat, re.IGNORECASE) for name, pat in SAFE_CATEGORIES.items()}


@dataclass(frozen=True)
class VetoHit:
    label: str
    sample: str


@dataclass(frozen=True)
class PolicyVerdict:
    allowed: bool
    mode: str
    categories: frozenset[str] = frozenset()
    veto_hits: tuple[VetoHit, ...] = field(default_factory=tuple)
    reason: str = ""


def _samples(text: str, regex: re.Pattern[str], limit: int = 3) -> list[str]:
    found: list[str] = []
    for m in regex.finditer(text):
        a, b = max(0, m.start() - 30), min(len(text), m.end() + 30)
        found.append(" ".join(text[a:b].split()))
        if len(found) >= limit:
            break
    return found


def evaluate_case(case_facts: str) -> PolicyVerdict:
    """Classify the case brief: veto first (always), then the allowlist.

    Fail-closed order matters — a case that matches "shoplifting" AND
    "shots fired" is a violence case, not a shoplifting case.
    """
    text = " ".join((case_facts or "").split())
    mode = _mode()
    hits: list[VetoHit] = []
    for label, regex in _COMPILED_VETO:
        for sample in _samples(text, regex, limit=1):
            hits.append(VetoHit(label=label, sample=sample))
    if hits:
        labels = ", ".join(sorted({h.label for h in hits}))
        return PolicyVerdict(
            allowed=False,
            mode=mode,
            veto_hits=tuple(hits),
            reason=f"content policy veto ({labels}): {hits[0].sample!r}",
        )
    if mode == VETO_ONLY_MODE:
        return PolicyVerdict(allowed=True, mode=mode, reason="veto-only mode: no violent markers")
    if not text:
        return PolicyVerdict(
            allowed=True,
            mode=mode,
            reason="no case facts to classify — veto layer only (documented law)",
        )
    categories = frozenset(name for name, regex in _COMPILED_SAFE.items() if regex.search(text))
    if not categories:
        return PolicyVerdict(
            allowed=False,
            mode=mode,
            reason=(
                "case facts match no on-menu category "
                f"({', '.join(sorted(_COMPILED_SAFE))}) — off-menu footage is not planned"
            ),
        )
    return PolicyVerdict(
        allowed=True,
        mode=mode,
        categories=categories,
        reason=f"on-menu: {', '.join(sorted(categories))}",
    )


# --- transcript veto ranges -------------------------------------------------
def _utterance_spans(transcript: dict) -> list[tuple[float, float, str]]:
    # sort defensively — _speech_intervals sorts too; an out-of-order word
    # list must never merge a marker into the wrong utterance
    words = sorted(
        (w for w in transcript.get("words", []) if w.get("text")),
        key=lambda w: float(w["start_s"]),
    )
    if not words:
        return []
    spans: list[list] = []
    for w in words:
        start, end = float(w["start_s"]), float(w["end_s"])
        text = str(w["text"])
        if spans and start <= spans[-1][1] + 2.0:
            spans[-1][1] = max(spans[-1][1], end)
            spans[-1][2] += f" {text}"
        else:
            spans.append([start, end, text])
    return [(a, b, " ".join(t.split())) for a, b, t in spans]


def vetoed_ranges(transcript: dict, pad_s: float = 1.0) -> list[tuple[float, float, str]]:
    """Source-clock windows whose transcript carries a violent marker.

    A hit on one utterance vetoes that utterance's window (padded): the
    planner will not cut raw footage from inside it.
    """
    ranges: list[tuple[float, float, str]] = []
    for a, b, text in _utterance_spans(transcript):
        for label, regex in _COMPILED_VETO:
            if regex.search(text):
                ranges.append((max(0.0, a - pad_s), b + pad_s, label))
                break
    return _merge_ranges(ranges)


def _merge_ranges(ranges: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    if not ranges:
        return []
    out = sorted(ranges)
    merged = [out[0]]
    for a, b, label in out[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2])
        else:
            merged.append((a, b, label))
    return merged


def clip_safe_span(
    start_s: float, dur_s: float, ranges: list[tuple[float, float, str]]
) -> tuple[float, float] | None:
    """Trim a planned window off veto ranges; None when no safe span survives.

    A window straddling a veto range is cut back to the safe part before the
    range; a window swallowed by ranges is refused entirely. Returns a
    (start, duration) pair.
    """
    dur = max(0.0, dur_s)
    for _a, b, _label in _merge_ranges(list(ranges)):
        if b <= start_s or start_s + dur <= _a:
            continue
        if _a > start_s:
            dur = _a - start_s  # safe part BEFORE the veto range
            break
        # range covers the window start: jump the start to the range end
        start_s, dur = b, start_s + dur - b
    if dur < 0.5:
        return None
    return (round(start_s, 3), round(dur, 3))


def drop_vetoed(
    intervals: list[tuple[float, float]], ranges: list[tuple[float, float, str]]
) -> list[tuple[float, float]]:
    """Filter candidate clip windows: any overlap with a veto range is out.

    Input and output are (start, end) interval pairs — a straddling window
    survives trimmed to its safe part, a swallowed window is dropped.
    """
    if not ranges:
        return list(intervals)
    out = []
    for a, b in intervals:
        safe = clip_safe_span(a, b - a, ranges)
        if safe is None or safe[1] < min(2.0, b - a):
            continue
        out.append((safe[0], safe[0] + safe[1]))
    return out


def first_safe_window(
    source_duration_s: float, dur_s: float, ranges: list[tuple[float, float, str]]
) -> tuple[float, float]:
    """First veto-free window scanning from the source start (fallback law)."""
    dur = min(dur_s, max(1.0, source_duration_s))
    merged = _merge_ranges(list(ranges))
    cursor = 0.0
    for a, b, _label in merged:
        if cursor + dur <= a:
            return (cursor, dur)
        cursor = max(cursor, b)
        if cursor >= source_duration_s:
            break
    return (min(cursor, max(0.0, source_duration_s - dur)), dur)
