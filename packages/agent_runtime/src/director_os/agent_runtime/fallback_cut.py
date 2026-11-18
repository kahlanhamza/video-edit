"""Deterministic multi-act documentary fallback (no-LLM path).

Mandatory change #2: when no LLM rung is reachable, planning must NOT emit a
1-minute slice or an 8s-grained full dump — it samples the whole story arc:

  Cold Open (2 climax hooks from the 60-80% region + narrator hook)
  -> Act 1 Initial Contact -> Act 2 Evidence & Excuses -> Act 3 Escalation
  -> Act 4 Miranda / Policy -> Act 5 Arrest -> Act 6 Booking + Outro

Clips are speech-anchored (top speech intervals per act window, so dead air
falls out by construction) and sized to land the 600-900s target window for
long sources. Narrator bridges are generic, fact-free scripting — the
deterministic path never invents case facts (that is the dossier's job).
"""

from __future__ import annotations

from dataclasses import dataclass

from director_os.otio_engine.contract import (
    CardMeta,
    NarratorClipMeta,
    RawClipMeta,
)
from director_os.otio_engine.timeline import ClipSpec, GapSpec, TimelineSpec, TrackSpec

FALLBACK_MIN_SOURCE_S = 900.0  # below this the v1 auto-cut stays (short clips)
TARGET_FLOOR_S = 600.0
TARGET_CEIL_S = 900.0
HOOK_REGION = (0.60, 0.80)  # climax hooks come from later in the video

ACT_WINDOWS = (
    ("Act 1: Initial Contact", 0.02, 0.15),
    ("Act 2: Evidence & Excuses", 0.15, 0.38),
    ("Act 3: Escalation & Admissions", 0.38, 0.58),
    ("Act 4: Miranda / Policy", 0.58, 0.72),
    ("Act 5: The Arrest", 0.72, 0.88),
    ("Act 6: Booking & Outro", 0.88, 1.00),
)

NARRATOR_BRIDGES = {
    "Cold Open": (
        "The bodycam was already rolling. What it caught turned a routine "
        "night shift into a criminal case."
    ),
    "Act 1: Initial Contact": (
        "It starts like any other stop. Small talk, routine questions — and "
        "the first details that will not add up."
    ),
    "Act 2: Evidence & Excuses": (
        "Officers start comparing what they are seeing with what they are "
        "being told. The excuses are not holding."
    ),
    "Act 3: Escalation & Admissions": ("The questions get sharper. So do the answers."),
    "Act 4: Miranda / Policy": (
        "Miranda rights, department policy, and the conversation that decides how this case ends."
    ),
    "Act 5: The Arrest": ("The arrest itself — caught in full, from both sides of the handcuffs."),
    "Act 6: Booking & Outro": ("Booking, processing, and the tests that lock the facts in place."),
}


@dataclass
class _Block:
    kind: str  # raw | narration
    source_start_s: float
    dur_s: float
    script: str = ""
    act: str = ""


def _speech_intervals(transcript: dict) -> list[tuple[float, float]]:
    words = [w for w in transcript.get("words", []) if w.get("text")]
    if not words:
        return []
    spans = sorted((float(w["start_s"]), float(w["end_s"])) for w in words)
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1] + 0.2:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged if b - a >= 2.0]


def _top_intervals(
    intervals: list[tuple[float, float]], window: tuple[float, float], source_dur: float, count: int
) -> list[tuple[float, float]]:
    lo, hi = window[0] * source_dur, window[1] * source_dur
    in_win = [(max(a, lo), min(b, hi)) for a, b in intervals if a < hi and b > lo]
    in_win = [(a, b) for a, b in in_win if b - a >= 4.0]
    in_win.sort(key=lambda p: -(p[1] - p[0]))
    return sorted(in_win[:count])


def plan_fallback_blocks(source_duration_s: float, transcript: dict) -> list[_Block]:
    """Pure deterministic planner: the full-arc block list."""
    target = max(TARGET_FLOOR_S, min(TARGET_CEIL_S, source_duration_s * 0.28))
    intervals = _speech_intervals(transcript)

    blocks: list[_Block] = []
    # --- cold open: 2 climax hooks + 15s narrator hook
    hook_a, hook_b = HOOK_REGION
    hooks = _top_intervals(intervals, (hook_a, hook_b), source_duration_s, 4)[:2]
    if hooks:
        for a, b in hooks:
            d = max(12.0, min(28.0, b - a))
            blocks.append(_Block("raw", max(0.0, a - 0.5), d, act="Cold Open"))
    else:  # no speech signal: fixed climax positions
        blocks.append(_Block("raw", source_duration_s * hook_a, 28.0, act="Cold Open"))
        blocks.append(_Block("raw", source_duration_s * (hook_a + 0.08), 28.0, act="Cold Open"))
    blocks.append(_Block("narration", 0.0, 15.0, NARRATOR_BRIDGES["Cold Open"], "Cold Open"))

    # --- acts: narrator bridge + 3 speech-anchored clips each
    n_acts = len(ACT_WINDOWS)
    narr_per_act = 12.0
    narration_total = 15.0 + n_acts * narr_per_act
    per_act = max(40.0, (target - narration_total) / n_acts)
    per_clip = per_act / 3.0

    for name, lo, hi in ACT_WINDOWS:
        blocks.append(
            _Block("narration", source_duration_s * lo, narr_per_act, NARRATOR_BRIDGES[name], name)
        )
        picks = _top_intervals(intervals, (lo, hi), source_duration_s, 3)
        if picks:
            for a, b in picks:
                d = min(max(12.0, min(per_clip + 6.0, b - a + 1.2)), 75.0)
                blocks.append(_Block("raw", max(0.0, a - 0.5), d, act=name))
        else:  # fixed grid fallback inside the window
            w = (hi - lo) * source_duration_s
            for k in range(3):
                start = lo * source_duration_s + k * (w / 3.2)
                blocks.append(_Block("raw", start, min(per_clip, w / 3.0), act=name))
    return blocks


def build_fallback_documentary(
    asset_id: str,
    source_duration_s: float,
    transcript: dict,
    fps: int,
    width: int,
    height: int,
    project_id: str = "fallback",
) -> TimelineSpec:
    """The deterministic 6-act documentary timeline spec (QA-invariant-safe)."""
    blocks = plan_fallback_blocks(source_duration_s, transcript)

    master: list[ClipSpec | GapSpec] = []
    for i, b in enumerate(blocks):
        if b.kind == "raw":
            end = min(b.source_start_s + b.dur_s, source_duration_s - 0.5)
            dur = max(8.0, end - b.source_start_s)
            master.append(
                ClipSpec(
                    name=f"raw_{i}",
                    source_start_s=b.source_start_s,
                    dur_s=dur,
                    meta=RawClipMeta(asset_id=asset_id, act_name=b.act),
                )
            )
        else:
            master.append(
                ClipSpec(
                    name=f"narration_{i}",
                    source_start_s=b.source_start_s,
                    dur_s=b.dur_s,
                    meta=NarratorClipMeta(
                        script_text=b.script,
                        broll_asset_id=asset_id,
                        broll_range=(b.source_start_s, b.source_start_s + b.dur_s),
                        broll_audio_volume=0.0,
                        act_name=b.act,
                    ),
                )
            )

    # cards: location_intro on Act 1's narration, evidence_fact mid-doc,
    # case_outcome on the Outro — placed at the narration block's start + 0.5s
    card_specs: list[tuple[float, ClipSpec]] = []

    def _card_at(act_name: str, template: str, title: str, subtitle: str, dur_s: float) -> None:
        t = 0.0
        for b in blocks:
            if b.kind == "narration" and b.act == act_name:
                card_specs.append(
                    (
                        t + 0.5,
                        ClipSpec(
                            name=f"card_{template}",
                            source_start_s=0.0,
                            dur_s=dur_s,
                            meta=CardMeta(
                                template=template, fields={"title": title, "subtitle": subtitle}
                            ),
                        ),
                    )
                )
                return
            t += b.dur_s

    _card_at(
        "Act 1: Initial Contact",
        "location_intro",
        "CASE FILE",
        "Bodycam footage — public record",
        4.0,
    )
    _card_at(
        "Act 3: Escalation & Admissions",
        "evidence_fact",
        "THE EVIDENCE",
        "From the raw transcript",
        4.0,
    )
    _card_at("Act 6: Booking & Outro", "case_outcome", "CASE OUTCOME", "Per official records", 5.0)

    overlay: list[ClipSpec | GapSpec] = []
    cursor = 0.0
    total = sum(b.dur_s for b in blocks)
    for pos, card in sorted(card_specs, key=lambda p: p[0]):
        if pos > cursor + 0.01:
            overlay.append(GapSpec(dur_s=pos - cursor))
            cursor = pos
        overlay.append(card)
        cursor += card.dur_s
    if total - cursor > 0.01:
        overlay.append(GapSpec(dur_s=total - cursor))

    return TimelineSpec(
        name=f"doc_fallback_{project_id[:8]}",
        fps=fps,
        width=width,
        height=height,
        tracks=[
            TrackSpec(name="V1_master", items=master),
            TrackSpec(name="V2_overlay_cards", items=overlay),
        ],
    )
