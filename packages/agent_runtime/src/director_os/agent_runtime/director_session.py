"""Director session — the 2-phase "Story Map -> Full-Arc Execution" flow.

Postmortem fixed here: the Director used to get ONE blind propose() call against
an 8-second-grained auto-cut scaffold; it made 2 trims and the loop ended with
a 59-second master. The new flow:

Phase A (Macro Plan) — the ENTIRE raw footage, condensed into a ~2.5k-token
chronological Story Index (story_map.py) plus the CaseDossier, is sent in ONE
JSON completion. The Director returns a complete Cold Open + multi-act
documentary outline scaled to the SOURCE LENGTH (doc_scale.py — any tape
length, not just the 43-min acceptance file) with explicit source ranges,
narration scripts, and cards.

Phase A.5 (deterministic truth passes, no LLM trust):
- ASR-boundary snapping  — every raw clip's start/end snaps to real utterance
  edges (pad_before=0.5s, pad_after=0.6s) and internal dead air >1.8s is split
  out, so punchlines are never sliced and no 25s silent stall survives.
- word-count law         — narration durations are COMPUTED from the script's
  word count (~2.6 w/s); the LLM's dur_s is never trusted.
- quote-grounded refine  — the actual ASR lines inside the selected windows
  (+/-60s) go back to the LLM once, so it trims tangents and writes specific
  commentary against real quotes. Any failure keeps the snapped plan.

Phase B (Tool Execution Loop) — the validated macro plan is expanded into
`timeline.insert_raw_clip` / `timeline.add_narration` / `cards.add` tool steps
and executed act-by-act through the checkpointed registry (so every mutation
stays durable + logged). Then an LLM refinement loop (max_turns >= 25, never
terminating while the duration invariants are unmet) polishes: trims, splits,
blur boxes, narration rewrites.

Hard invariants (plan_invariants.py) gate every exit: a plan that cannot reach
its length-scaled floors is retried with failure feedback, then replaced by
the deterministic multi-act fallback — the session never returns a truncated
timeline again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from director_os.agent_runtime.agents import PlannerExecutor, ToolStep
from director_os.agent_runtime.doc_scale import (
    HOOK_MAX_END_S,
    HOOK_MAX_START_S,
    PLAN_WPS,
    STING_MAX_S,
    WORDS_COLD,
    WORDS_MID,
    WORDS_OUTRO,
    DocScale,
    doc_scale,
)
from director_os.agent_runtime.plan_invariants import (
    TimelineComposition,
    audit_invariants,
    coverage_span,
    timeline_composition,
)
from director_os.agent_runtime.prompting import compose_agent_prompt
from director_os.agent_runtime.story_map import StoryBucket, story_map_prompt
from pydantic import BaseModel, Field, ValidationError

# ------------------------------------------------------------ Phase A models
BlockKind = Literal["raw", "narration"]

# narration duration = computed from word count (the LLM's dur_s is ignored)
NARR_TAIL_PAD_S = 0.8
NARR_DUR_MIN_S = 8.0
NARR_DUR_MAX_S = 95.0
# ASR-boundary snapping (AI-studio review law)
SNAP_PAD_BEFORE_S = 0.5
SNAP_PAD_AFTER_S = 0.6
SNAP_BACK_MAX_S = 8.0  # max a start edge may pull back to complete an utterance
SNAP_UTT_MERGE_GAP_S = 1.2  # utterances closer than this are one run
DEAD_AIR_SPLIT_S = 1.8  # internal silence above this splits the clip
PIECE_MIN_S = 4.0  # speech runs shorter than this are dropped (if any run qualifies)
MAX_PIECES_PER_CLIP = 6


def narr_words(script: str) -> int:
    return len(script.split())


def narr_est_s(script: str) -> float:
    """Deterministic narration duration from word count (~2.6 w/s + tail).

    The LLM's dur_s is NEVER trusted (the acceptance review caught 34-43 word
    scripts labeled 25s). The post-TTS retime replaces this with the real clip
    duration anyway; the estimate only has to be honest.
    """
    words = max(1, narr_words(script))
    return max(NARR_DUR_MIN_S, min(NARR_DUR_MAX_S, words / PLAN_WPS + NARR_TAIL_PAD_S))


class PlanRawClip(BaseModel):
    type: Literal["raw"] = "raw"
    source_start_s: float = Field(ge=0.0)
    dur_s: float = Field(gt=0.0, le=600.0)
    note: str = Field(default="", max_length=300)


class PlanNarration(BaseModel):
    type: Literal["narration"] = "narration"
    script: str = Field(min_length=8, max_length=2000)
    # accepted but never trusted — expand_macro_plan recomputes from word count
    dur_s: float | None = Field(default=None, ge=3.0, le=180.0)
    broll_source_start_s: float | None = Field(default=None, ge=0.0)
    # 2-3 sequential muted b-roll windows under ONE spoken block (TTS stays
    # continuous — the visuals cut, the voice does not)
    broll_source_starts: list[float] | None = Field(default=None, max_length=4)

    @property
    def est_s(self) -> float:
        return narr_est_s(self.script)


PlanBlock = PlanRawClip | PlanNarration


class PlanAct(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    blocks: list[PlanBlock] = Field(min_length=1)


class PlanCard(BaseModel):
    template: Literal["location_intro", "evidence_fact", "case_outcome", "evidence_quote"]
    title: str = Field(min_length=1, max_length=120)
    subtitle: str = Field(default="", max_length=200)
    at_act: int = Field(ge=0)
    at_block: int = Field(default=0, ge=0)
    dur_s: float = Field(default=4.0, ge=1.0, le=15.0)


class MacroPlan(BaseModel):
    """The complete documentary outline (Phase A product)."""

    title: str = Field(default="", max_length=200)
    # act-count is a _plan_problems check (retry/repair), not a parse error: a
    # truncated-but-parseable plan must reach the Phase B repair loop
    acts: list[PlanAct] = Field(min_length=1)
    cards: list[PlanCard] = Field(default_factory=list)

    @property
    def raw_count(self) -> int:
        return sum(1 for a in self.acts for b in a.blocks if isinstance(b, PlanRawClip))

    @property
    def narration_count(self) -> int:
        return sum(1 for a in self.acts for b in a.blocks if isinstance(b, PlanNarration))

    @property
    def total_words(self) -> int:
        return sum(
            len(b.script.split())
            for a in self.acts
            for b in a.blocks
            if isinstance(b, PlanNarration)
        )

    @property
    def total_dur_s(self) -> float:
        """HONEST duration: narration blocks are word-count computed, raw
        blocks use their (possibly snapped) spans — never the LLM's guesses."""
        return sum(
            (b.est_s if isinstance(b, PlanNarration) else b.dur_s)
            for a in self.acts
            for b in a.blocks
        )

    def clip_span(self, source_duration_s: float) -> tuple[float | None, float | None]:
        first, last = None, None
        for act in self.acts:
            for b in act.blocks:
                if isinstance(b, PlanRawClip):
                    first = b.source_start_s if first is None else min(first, b.source_start_s)
                    end = b.source_start_s + b.dur_s
                    last = end if last is None else max(last, end)
        return first, (min(last, source_duration_s) if last is not None else None)


MACRO_PLAN_SCHEMA_DOC = """Return ONLY a JSON object (no prose, no code fences):
{"title": "working title",
 "acts": [
   {"name": "Cold Open", "blocks": [
      {"type": "raw", "source_start_s": 1234.0, "dur_s": 3.5, "note": "climax sting, raw audio ON"},
      {"type": "narration", "script": "hook line(s) — who/what/where/when",
       "broll_source_start_s": 1240.0,
       "broll_source_starts": [1240.0, 1290.0]}
   ]},
   {"name": "Act 1: ...", "blocks": [ ... ]},
   ... remaining acts ...
 ],
 "cards": [
   {"template": "location_intro", "title": "...", "subtitle": "...",
    "at_act": 1, "at_block": 0, "dur_s": 4.0},
   {"template": "evidence_fact", "title": "...", "subtitle": "...",
    "at_act": 3, "at_block": 0, "dur_s": 4.0},
   {"template": "case_outcome", "title": "...", "subtitle": "...",
    "at_act": 6, "at_block": 0, "dur_s": 5.0}
 ]}

dur_s on narration blocks is IGNORED — the system computes each block's real
duration from its word count (~2.6 words/sec). Raw clips snap to real speech
boundaries automatically; keep windows tight around the dialogue you want."""


def _target_lo(source_duration_s: float) -> float:
    return doc_scale(source_duration_s).target_lo


def _target_hi(source_duration_s: float) -> float:
    return doc_scale(source_duration_s).target_hi


NARRATION_PROMPT_LAW = """## Narration law (author by WORD COUNT; Brian +14% speaks ~2.9 words/sec)
- One narration block = ONE complete spoken script. NEVER split one narration
  across consecutive blocks; for visual movement under a longer block give
  `broll_source_starts` (a list of 2-3 muted b-roll source starts) and the
  renderer cuts the visuals while the voice stays continuous.
- Total narration across all blocks: {words_total_min}+ words (~30-40% of runtime).
- Cold Open narration: {cold_lo}-{cold_hi} words. Mid-video commentary: {mid_lo}-{mid_hi}
  words each. Final outro block: {outro_lo}-{outro_hi} words.
- Tone: calm, observant, analytical true-crime documentary voice — zero melodrama,
  zero snark; surface contradictions factually, never editorialize guilt.
- Facts come ONLY from the dossier; hint at mystery rather than invent.
- ANTI-FILLER: every block must reveal unseen dossier context, expose a specific
  contradiction against exact words/visuals, or raise the stakes for the next
  clip — never blandly summarize what the viewer just heard.
- NEVER spoil a raw clip's punchline right before it plays: tease the setup
  before the clip, or analyze the contradiction AFTER the viewer hears it.
- PACE: short declarative sentences, 1-2 clauses max, no throat-clearing openers
  ("what happens next will shock you" is banned) — the narration must catch
  attention, never lull it.
- J/L-cut: at most 0.3-0.6s into silent padding — narration NEVER overlaps
  active Officer/Suspect dialogue.
- B-roll windows must sit in QUIET, non-dialogue moments (verified automatically
  against the transcript) — never mute an active dramatic or funny exchange."""

HOOK_PROMPT_LAW = """## HOOK-FIRST LAW (non-negotiable — think like a real pro editor)
- The FIRST thing the viewer hears must be narration. acts[0].blocks[0] is
  either the hook narration itself OR a raw climax sting of ≤{sting}s (raw audio
  on) followed IMMEDIATELY by the hook narration.
- The hook narration starts ≤{hook_start}s into the video and ENDS by {hook_end}s.
  Inside those first 30 seconds the viewer learns WHO is on camera, WHERE, WHEN,
  and WHAT the tension is (all verifiable from the dossier/title facts only).
- Open mid-action: the sting/hook b-roll is the single most dramatic moment
  of the whole tape (the shout, the collapse, the cuffs) — never a wide
  establishing shot, never a title card first.
- A plan whose first narration starts after {hook_start}s is REJECTED before
  execution."""


def macro_plan_user_prompt(
    buckets: list[StoryBucket],
    source_duration_s: float,
    dossier_json: dict[str, Any] | None = None,
) -> str:
    scale = doc_scale(source_duration_s)
    dossier_block = ""
    if dossier_json:
        dossier_block = (
            "\n\n## Verified case dossier (use ONLY these facts; never invent "
            "charges, names, or verdicts)\n" + json.dumps(dossier_json, ensure_ascii=False)[:4000]
        )
    return (
        story_map_prompt(buckets, source_duration_s)
        + dossier_block
        + f"""

## Your task
Plan a COMPLETE {scale.target_lo:.0f}-{scale.target_hi:.0f}s documentary
(Cold Open + {scale.act_min}-{scale.act_max} acts) of this
{source_duration_s / 60:.0f}-min video. The structure scales with the source
length — a shorter tape gets fewer, tighter acts; a long tape gets the full
arc. Never plan longer than the source.

Structure (acts[0] is the Cold Open) — the GENERAL arrest/incident arc, not a
template:
1. Cold Open Hook ({STING_MAX_S:.0f}s sting + the hook narration, context lands
   ≤30s): a ≤{STING_MAX_S:.0f}s climax raw sting (PULLED FROM LATER IN THE
   VIDEO, raw audio on) + the {WORDS_COLD[0]}-{WORDS_COLD[1]} word narration hook
   setting date/location/mystery from the dossier only, over muted climax b-roll.
2+. Body acts: arrival/contact -> investigation/friction -> escalation/turning
   point -> arrest/detention -> aftermath/booking -> verified outcome — include
   the beats this footage ACTUALLY has, in order, and drop what it doesn't.
   Miranda readings, policy violations, and sobriety/field tests are HIGHLIGHTS
   to surface whenever present — never force them into a beat they don't belong
   to. Trim boilerplate (a Miranda reading = intro + waiver, never the full recital).

{NARRATION_PROMPT_LAW.format(
    words_total_min=scale.words_total_min,
    cold_lo=WORDS_COLD[0], cold_hi=WORDS_COLD[1],
    mid_lo=WORDS_MID[0], mid_hi=WORDS_MID[1],
    outro_lo=WORDS_OUTRO[0], outro_hi=WORDS_OUTRO[1])}

{HOOK_PROMPT_LAW.format(sting=STING_MAX_S, hook_start=HOOK_MAX_START_S, hook_end=HOOK_MAX_END_S)}

Hard rules:
- Raw clips: {scale.raw_min}-{scale.raw_max} speech-anchored stretches totaling
  {scale.target_lo * 0.55:.0f}-{scale.target_hi * 0.7:.0f}s of source, mostly
  15-{scale.clip_max_s:.0f}s per stretch; every stretch must be built around
  dialogue or decisive action (dead air is cut automatically, so pick windows
  by what is SAID, not by the clock).
- Clip starts must span the WHOLE file: earliest within the first 15%, latest
  ending inside the final 25% (never drop the arrest/booking ending).
- Cut dead air and logistics (keys, phone details, radio spelling); KEEP
  ironic, emotional, and incriminating exchanges — and keep them COMPLETE
  (a clip may extend past its window to finish the punchline; say so in `note`).
- Cards: location_intro on the first body act's narration; 1-3 evidence_fact
  mid-doc on concrete contradictions/BAC/policy facts; case_outcome on the
  Outro. If the dossier has a verified verdict, state it cleanly; only use
  "presumed innocent" framing when the verdict is pending/unknown.
- Never mention video timestamps/timecodes (like "at 04:12"); real case
  durations ("a 20-minute observation period") are fine.
- Source seconds come from the story index [mm:ss] markers; mm:ss -> seconds.

{MACRO_PLAN_SCHEMA_DOC}"""
    )


def parse_macro_plan(text: str) -> MacroPlan:
    """Lenient parse: strip code fences, take the outermost JSON object."""
    from director_os.agent_runtime.llm import strip_code_fences

    cleaned = strip_code_fences(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("macro plan is not a JSON object")
    return MacroPlan.model_validate(data)


# -------------------------------------------------- deterministic truth passes
def speech_utterances(transcript: dict | None) -> list[tuple[float, float]]:
    """Merged speech runs [(start, end), ...] from the transcript.

    Segments are preferred (diarized, sentence-shaped); word timings are the
    fallback. Utterances closer than SNAP_UTT_MERGE_GAP_S merge into one run
    (a breath is not dead air).
    """
    if not transcript:
        return []
    spans: list[tuple[float, float]] = []
    segments = transcript.get("segments") or []
    if segments:
        for seg in segments:
            try:
                s, e = float(seg.get("start_s", 0.0)), float(seg.get("end_s", 0.0))
            except (TypeError, ValueError):
                continue
            if e > s:
                spans.append((s, e))
    else:
        for w in transcript.get("words") or []:
            if not (w.get("text") or w.get("word")):
                continue
            try:
                s, e = float(w.get("start_s", 0.0)), float(w.get("end_s", 0.0))
            except (TypeError, ValueError):
                continue
            if e > s:
                spans.append((s, e))
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + SNAP_UTT_MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def _clip_pieces(
    start: float, dur: float, utts: list[tuple[float, float]], source_end: float
) -> list[tuple[float, float]]:
    """Snap one planned window to speech: utterance-edge boundaries, pads, and
    dead-air splitting (>DEAD_AIR_SPLIT_S silence splits the window).

    Edge law (the acceptance review's "punchlines never slice"): the END edge
    always extends to the last overlapping utterance's true end + pad_after —
    a clip may run past its window to FINISH the line it started. The START
    edge snaps to the utterance start - pad (killing leading dead air), and
    when the window begins mid-utterance it pulls back at most
    SNAP_BACK_MAX_S so a deliberately late in-point survives.
    """
    w0, w1 = max(0.0, start), min(start + dur, source_end)
    if w1 - w0 < 2.0:
        return [(w0, w1 - w0)] if w1 > w0 else []
    overlapping = [u for u in utts if u[0] < w1 and u[1] > w0]
    if not overlapping:
        return [(w0, w1 - w0)]  # silent footage (action/visuals): keep as planned
    pieces: list[list[float]] = []
    for u0, u1 in overlapping:
        if u0 >= w0:
            s0 = max(0.0, u0 - SNAP_PAD_BEFORE_S)  # leading dead air: snap forward
        else:  # window starts mid-utterance: pull back, bounded
            s0 = max(0.0, max(u0 - SNAP_PAD_BEFORE_S, w0 - SNAP_BACK_MAX_S))
        s1 = min(source_end, u1 + SNAP_PAD_AFTER_S)  # punchline completes, always
        pieces.append([s0, s1])
    # merge pieces whose gap <= DEAD_AIR_SPLIT_S (breath is not dead air)
    merged = [pieces[0]]
    for p0, p1 in pieces[1:]:
        if p0 - merged[-1][1] <= DEAD_AIR_SPLIT_S:
            merged[-1][1] = max(merged[-1][1], p1)
        else:
            merged.append([p0, p1])
    runs = [(a, b) for a, b in merged if b - a >= PIECE_MIN_S]
    if not runs:  # everything tiny: keep the single largest run
        runs = [max(merged, key=lambda p: p[1] - p[0])]
    runs = sorted(runs)[:MAX_PIECES_PER_CLIP]
    return [(a, b - a) for a, b in runs if b > a]


def snap_plan_to_speech(
    plan: MacroPlan, transcript: dict | None, source_duration_s: float
) -> tuple[MacroPlan, int]:
    """Deterministic ASR-boundary pass over every raw clip in the plan.

    Returns (plan', number_of_pieces_after_snapping). Narration blocks pass
    through untouched. A plan with NO transcript rides through unchanged.
    """
    utts = speech_utterances(transcript)
    if not utts:
        return plan, plan.raw_count
    pieces_total = 0
    for act in plan.acts:
        new_blocks: list[PlanBlock] = []
        for blk in act.blocks:
            if isinstance(blk, PlanRawClip):
                src_end = max(source_duration_s, blk.source_start_s + blk.dur_s)
                pieces = _clip_pieces(blk.source_start_s, blk.dur_s, utts, src_end)
                if not pieces:
                    continue
                pieces_total += len(pieces)
                for p0, p_dur in pieces:
                    new_blocks.append(
                        PlanRawClip(
                            source_start_s=round(p0, 2),
                            dur_s=round(p_dur, 2),
                            note=blk.note,
                        )
                    )
            else:
                new_blocks.append(blk)
        act.blocks = new_blocks
    return plan, pieces_total


def _quotes_block(
    plan: MacroPlan, transcript: dict | None, source_duration_s: float, budget_chars: int = 3500
) -> str:
    """The actual ASR dialogue inside/around the selected windows (+/-60s)."""
    if not transcript:
        return ""
    segments = [
        (float(s.get("start_s", 0.0)), float(s.get("end_s", 0.0)),
         str(s.get("speaker") or "?"), str(s.get("text", "")).strip())
        for s in (transcript.get("segments") or [])
        if str(s.get("text", "")).strip()
    ]
    if not segments:
        return ""
    windows: list[tuple[float, float]] = [
        (b.source_start_s, b.source_start_s + b.dur_s)
        for a in plan.acts
        for b in a.blocks
        if isinstance(b, PlanRawClip)
    ]
    if not windows:
        return ""
    lines: list[str] = []
    used = 0
    seen: set[int] = set()
    for w0, w1 in windows:
        near = [
            (i, s) for i, s in enumerate(segments)
            if s[0] < w1 + 60.0 and s[1] > w0 - 60.0 and i not in seen
        ]
        for i, (s0, s1, spk, text) in near[:8]:
            if used >= budget_chars:
                break
            snippet = " ".join(text.split())[:180]
            lines.append(f"[{s0:.0f}s] {spk}: {snippet}")
            seen.add(i)
            used += len(lines[-1]) + 2
        if used >= budget_chars:
            break
    if not lines:
        return ""
    return (
        "\n\n## ACTUAL TRANSCRIPT inside/around your selected windows (+/-60s) — "
        "trim tangents, complete punchlines (extend a window to finish the last "
        "line), and reference these exact words in the narration:\n"
        + "\n".join(lines)
    )


def refine_plan_quotes(
    plan: MacroPlan,
    transcript: dict | None,
    backend: Any,
    source_duration_s: float,
    problems_fn=None,
) -> tuple[MacroPlan, str]:
    """The Quote-Grounded Refinement pass (one extra completion, soft-fail).

    Feeds the real dialogue inside the selected windows back to the LLM so it
    can trim tangents, extend windows to complete punchlines, and rewrite
    generic narration against actual quotes. Guardrails: the returned plan is
    re-snapped to speech, every revised raw clip must stay within +/-90s of
    its original window, and the plan must still pass `_plan_problems` — any
    violation keeps the deterministic snapped plan (never a worse plan).
    """
    quote_block = _quotes_block(plan, transcript, source_duration_s)
    if not quote_block or backend is None or getattr(backend, "complete_json", None) is None:
        return plan, ""
    windows = [
        (b.source_start_s, b.source_start_s + b.dur_s)
        for a in plan.acts
        for b in a.blocks
        if isinstance(b, PlanRawClip)
    ]
    if not windows:
        return plan, ""
    system = compose_agent_prompt(agent="director_agent", role="director", niche="bodycam_documentary")
    user = (
        "## Quote-grounded refinement of an accepted documentary plan\n"
        "Here is the ACCEPTED plan JSON. Improve it using the transcript quotes: "
        "trim boring tangents, extend raw windows so the last utterance in a "
        "window finishes (pad lives after), sharpen narration against exact "
        "words, keep the same act structure and count. Return ONLY the revised "
        "JSON object.\n\n"
        + plan.model_dump_json()
        + quote_block
    )
    try:
        text = backend.complete_json(system, user)
        revised = parse_macro_plan(text)
    except Exception:  # noqa: BLE001 — a failed refine never breaks planning
        return plan, "refine parse failed — kept snapped plan"
    # guardrail: structure + window proximity
    if len(revised.acts) != len(plan.acts):
        return plan, "refine changed act count — kept snapped plan"
    orig_centers = [(a + b) / 2 for a, b in windows]
    for act in revised.acts:
        for blk in act.blocks:
            if isinstance(blk, PlanRawClip):
                center = blk.source_start_s + blk.dur_s / 2
                if not any(abs(center - c) <= 90.0 for c in orig_centers):
                    return plan, "refine wandered outside its windows — kept snapped plan"
                blk.dur_s = min(max(blk.dur_s, 3.0), 600.0)
    revised, _ = snap_plan_to_speech(revised, transcript, source_duration_s)
    if problems_fn is not None:
        problems = problems_fn(revised)
        if problems:
            return plan, "refine failed pre-flight: " + "; ".join(problems)
    return revised, ""


# ------------------------------------------------------------ Phase B expansion
def _as_block(blk: PlanRawClip | PlanNarration | dict) -> PlanRawClip | PlanNarration:
    """Coerce a stray dict into its model (defensive against mutated plans)."""
    if isinstance(blk, dict):
        if blk.get("type") == "narration":
            return PlanNarration.model_validate(blk)
        return PlanRawClip.model_validate(blk)
    return blk


def expand_macro_plan(plan: MacroPlan, source_duration_s: float) -> list[ToolStep]:
    """Expand the outline into ordered tool steps against the scaffold timeline.

    The scaffold's single full-length raw clip (item 0) is repurposed as the
    first planned clip (trim), everything else appends; the scaffold is removed
    at the end if it was not consumed. Card positions are computed from the
    cumulative block durations (absolute timeline seconds) with card_anchors
    set to each act's TRUE START — the old code anchored to the act's last
    block boundary, shifting every card forward by nearly a full act.
    """
    steps: list[ToolStep] = []
    scaffold_consumed = False
    if not plan.acts or not plan.acts[0].blocks:
        raise ValueError("macro plan has no first-act blocks")

    first = _as_block(plan.acts[0].blocks[0])
    if isinstance(first, PlanRawClip):
        steps.append(
            ToolStep(
                tool="timeline.trim_clip",
                args={
                    "item_index": 0,
                    "new_start_s": first.source_start_s,
                    "new_dur_s": first.dur_s,
                },
                checkpoint_note="cold open hook 1 onto scaffold clip",
            )
        )
        scaffold_consumed = True
        rest = plan.acts[0].blocks[1:]
    else:
        rest = plan.acts[0].blocks
    blocks_before_scaffold = 0 if scaffold_consumed else 1

    # act markers ride tool args; cards need absolute positions -> track them
    card_anchors: dict[int, float] = {}
    # the scaffold-consumed sting is the master's first 4s: t starts THERE so
    # every card anchor is a true master-clock position
    t = first.dur_s if scaffold_consumed else 0.0
    for act_i, act in enumerate(plan.acts):
        card_anchors[act_i] = t  # the TRUE start of this act
        blocks = [_as_block(b) for b in (rest if act_i == 0 else act.blocks)]
        for blk in blocks:
            if isinstance(blk, PlanRawClip):
                steps.append(
                    ToolStep(
                        tool="timeline.insert_raw_clip",
                        args={
                            "source_start_s": round(blk.source_start_s, 2),
                            "dur_s": round(
                                min(blk.dur_s, source_duration_s - blk.source_start_s), 2
                            ),
                            "act_name": act.name,
                        },
                        checkpoint_note=act.name,
                    )
                )
            else:
                args: dict[str, Any] = {
                    "script_text": blk.script,
                    # word-count law: the LLM's dur_s is never trusted
                    "dur_s": round(blk.est_s, 2),
                    "act_name": act.name,
                }
                starts = [s for s in (blk.broll_source_starts or []) if s is not None]
                if not starts and blk.broll_source_start_s is not None:
                    starts = [blk.broll_source_start_s]
                if starts:
                    # b-roll windows must FIT the source: a start at/after EOF
                    # slices zero frames (empty-chunk render crash)
                    win = min(blk.est_s / max(1, len(starts)) + 2.0, 30.0)
                    starts = [
                        round(max(0.0, min(s, source_duration_s - min(win, source_duration_s))), 2)
                        for s in starts
                    ]
                if len(starts) >= 2:
                    args["broll_windows"] = [
                        [s, round(s + min(blk.est_s / len(starts) + 2.0, 30.0), 2)]
                        for s in starts
                    ]
                elif starts:
                    args["broll_source_start_s"] = round(starts[0], 2)
                steps.append(
                    ToolStep(tool="timeline.add_narration", args=args, checkpoint_note=act.name)
                )
            t += blk.est_s if isinstance(blk, PlanNarration) else blk.dur_s

    if not scaffold_consumed:
        # the untouched full-length scaffold clip sits at index 0 -> drop it
        steps.append(
            ToolStep(
                tool="timeline.remove_clip",
                args={"item_index": blocks_before_scaffold - 1},
                checkpoint_note="drop unused full-length scaffold clip",
            )
        )

    # cards: absolute start = act's true start + offset of the target block
    for card in plan.cards:
        if card.at_act >= len(plan.acts):
            continue
        act_blocks = [_as_block(b) for b in plan.acts[card.at_act].blocks]
        offset = 0.0
        for b in act_blocks[: card.at_block]:
            offset += b.est_s if isinstance(b, PlanNarration) else b.dur_s
        start = card_anchors[card.at_act] + offset
        args: dict[str, Any] = {
            "template": card.template,
            "title": card.title,
            "start_s": round(start, 2),
            "dur_s": card.dur_s,
        }
        if card.subtitle:
            args["subtitle"] = card.subtitle
        steps.append(ToolStep(tool="cards.add", args=args, checkpoint_note=card.template))
    return steps


def director_state_summary(
    timeline, source_duration_s: float, transcript: dict | None = None
) -> str:
    """Per-clip state the refinement loop sees each turn (index-stable).

    With a transcript, each raw clip carries its first/last REAL quote so the
    refinement LLM argues against actual words, not guesses.
    """
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import is_gap, item_duration_s

    utts = speech_utterances(transcript)

    def _quote_for(src0: float, src1: float) -> str:
        if not utts:
            return ""
        inside = [
            (a, b) for a, b in utts if a < src1 and b > src0
        ]
        if not inside:
            return " (no speech)"
        a0, b0 = inside[0]
        a1, b1 = inside[-1]
        return f" quotes {a0:.0f}-{b0:.0f}s..{a1:.0f}-{b1:.0f}s"

    comp = TimelineComposition()
    lines: list[str] = []
    try:
        track = master_track(timeline)
    except Exception:
        return "timeline unreadable"
    for i, item in enumerate(track):
        if is_gap(item):
            continue
        dur = item_duration_s(item)
        meta = item.metadata.get("director_os") or {}
        kind = str(meta.get("kind", "?"))
        comp.final_duration_s += dur
        sr = getattr(item, "source_range", None)
        src = sr.start_time.to_seconds() if sr is not None else 0.0
        act = str(meta.get("act_name", "") or "-")
        if kind == "narration":
            comp.narrator_segment_count += 1
            script = str(meta.get("script_text", ""))[:60]
            lines.append(f"[{i}] NARRATION {dur:.1f}s broll@{src:.0f}s act={act} '{script}'")
        elif kind == "raw":
            comp.raw_segment_count += 1
            comp.first_source_s = (
                src if comp.first_source_s is None else min(comp.first_source_s, src)
            )
            end = src + dur
            comp.last_source_end_s = (
                end if comp.last_source_end_s is None else max(comp.last_source_end_s, end)
            )
            lines.append(
                f"[{i}] RAW {dur:.1f}s src {src:.1f}-{end:.1f}s act={act}"
                + _quote_for(src, end)
            )
    failures = audit_invariants(comp, source_duration_s)
    span = coverage_span(comp, source_duration_s)
    status = (
        "ALL INVARIANTS MET"
        if not failures
        else "UNMET: " + "; ".join(f"{f.code}" for f in failures)
    )
    return (
        f"master: {comp.final_duration_s:.1f}s | raw={comp.raw_segment_count} "
        f"narration={comp.narrator_segment_count} coverage={span:.2f}"
        + _targets_suffix(source_duration_s)
        + f" — {status}\n" + "\n".join(lines[:40])
    )


def _targets_suffix(source_duration_s: float) -> str:
    scale = doc_scale(source_duration_s)
    return (
        f" (target {scale.min_final_s:.0f}s+/{scale.narr_lo}+ narr/{scale.raw_min}+ raw/"
        f"{scale.min_coverage_span:.2f}+ span)"
    )


# ------------------------------------------------------------ the session
@dataclass
class DirectorSessionReport:
    ok: bool
    phase: str = ""  # macro | refine | fallback
    provider: str = ""
    plan: MacroPlan | None = None
    turns: int = 0
    steps_ok: int = 0
    steps_failed: int = 0
    composition: TimelineComposition | None = None
    failures: list[str] = field(default_factory=list)
    detail: str = ""


class DirectorSession:
    """Two-phase Director run with hard duration/coverage invariants."""

    def __init__(
        self,
        registry,
        store,
        project_id: str,
        backend,  # FallbackChain | GeminiRestBackend | OpenAICompatBackend
        source_duration_s: float,
        max_turns: int = 30,  # >= 25 mandated: the loop may not die at 1-2 calls
        phase_a_retries: int = 2,
        transcript: dict | None = None,  # enables ASR snap + quote refinement
        quote_refine: bool = True,
    ):
        self.registry = registry
        self.store = store
        self.project_id = project_id
        self.backend = backend
        self.source_duration_s = source_duration_s
        self.scale = doc_scale(source_duration_s)
        self.max_turns = max_turns
        self.phase_a_retries = phase_a_retries
        self.transcript = transcript
        self.quote_refine = quote_refine
        self.executor = PlannerExecutor(
            registry, store, project_id, agent="director_agent", role="director"
        )

    # ---------------------------------------------------------------- phase A
    def plan_macro(
        self, buckets: list[StoryBucket], dossier_json: dict[str, Any] | None = None
    ) -> tuple[MacroPlan | None, str]:
        """One JSON completion against the routing chain; retries with feedback.

        Returns (None, error) only for UNPARSEABLE plans after all retries — a
        parseable-but-truncated plan is returned as best effort so Phase B's
        repair loop (and the deterministic fallback behind it) can fix it.
        """
        system = compose_agent_prompt(
            agent="director_agent", role="director", niche="bodycam_documentary"
        )
        user = macro_plan_user_prompt(buckets, self.source_duration_s, dossier_json)
        feedback = ""
        last_err = ""
        best: MacroPlan | None = None
        for _attempt in range(self.phase_a_retries + 1):
            text = self.backend.complete_json(
                system, user + (f"\n\n## Previous attempt failed\n{feedback}" if feedback else "")
            )
            try:
                plan = parse_macro_plan(text)
                plan, _pieces = snap_plan_to_speech(plan, self.transcript, self.source_duration_s)
                problems = self._plan_problems(plan)
                if problems:
                    feedback = "; ".join(problems)
                    last_err = feedback
                    if best is None or self._honest_total(plan) > self._honest_total(best):
                        best = plan
                    continue
                if self.transcript and self.quote_refine:
                    plan, refine_note = refine_plan_quotes(
                        plan, self.transcript, self.backend, self.source_duration_s,
                        problems_fn=self._plan_problems,
                    )
                    if refine_note:
                        last_err = refine_note
                return plan, ""
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                feedback, last_err = f"unparseable plan ({e})", str(e)
                continue
        if best is not None:
            return best, last_err  # structurally valid but truncated: Phase B repairs
        return None, last_err

    def _honest_total(self, plan: MacroPlan) -> float:
        total = 0.0
        for act in plan.acts:
            for b in act.blocks:
                total += b.est_s if isinstance(b, PlanNarration) else b.dur_s
        return total

    def _plan_problems(self, plan: MacroPlan) -> list[str]:
        """Cheap pre-flight on the plan JSON (the timeline is the real judge).

        Scale-driven (doc_scale): every floor derives from the SOURCE LENGTH.
        Narration durations are word-count computed — an inflated LLM dur_s
        cannot buy its way past these checks.
        """
        scale = self.scale
        problems: list[str] = []
        if len(plan.acts) < scale.act_min:
            problems.append(
                f"only {len(plan.acts)} acts — need the Cold Open + {scale.act_min}-{scale.act_max} "
                "act structure for this source length"
            )
        honest_total = self._honest_total(plan)
        if honest_total < scale.min_final_s:
            problems.append(
                f"total plan duration {honest_total:.0f}s < {scale.min_final_s:.0f}s "
                "(narration durations are computed from word count: write enough words)"
            )
        if not scale.narr_lo <= plan.narration_count <= scale.narr_hi:
            problems.append(
                f"{plan.narration_count} narration blocks — need {scale.narr_lo}-{scale.narr_hi} "
                f"({scale.words_total_min}+ words total)"
            )

        # --- word-count law (never trust LLM dur_s)
        narr_blocks = [b for a in plan.acts for b in a.blocks if isinstance(b, PlanNarration)]
        total_words = sum(narr_words(b.script) for b in narr_blocks)
        if plan.narration_count and total_words < scale.words_total_min:
            problems.append(
                f"total narration {total_words} words < {scale.words_total_min} — write "
                "fuller commentary (dur_s is computed from word count, inflating it does nothing)"
            )
        for idx, blk in enumerate(narr_blocks):
            w = narr_words(blk.script)
            if idx == 0:
                lo, hi = WORDS_COLD
                kind = "cold open"
            elif idx == len(narr_blocks) - 1 and len(narr_blocks) > 2:
                lo, hi = WORDS_OUTRO
                kind = "outro"
            else:
                lo, hi = WORDS_MID
                kind = "mid-video"
            if not lo <= w <= hi:
                problems.append(
                    f"{kind} narration block #{idx + 1} is {w} words — need {lo}-{hi} "
                    f"(~{w / PLAN_WPS:.0f}s at real TTS pace)"
                )

        # --- narration ratio on HONEST durations
        raw_total = sum(
            b.dur_s for a in plan.acts for b in a.blocks if isinstance(b, PlanRawClip)
        )
        narr_total = sum(b.est_s for b in narr_blocks)
        if narr_total + raw_total > 0:
            ratio = narr_total / (narr_total + raw_total)
            if not 0.30 <= ratio <= 0.52:
                problems.append(
                    f"narration ratio {ratio:.2f} outside 0.30..0.52 "
                    "(rich commentary, not a sprinkle, not a podcast)"
                )
        if plan.raw_count < scale.raw_min:
            problems.append(f"only {plan.raw_count} raw clips (< {scale.raw_min})")
        if plan.raw_count > scale.raw_max:
            problems.append(
                f"{plan.raw_count} raw clips > {scale.raw_max} — consolidate tiny fragments"
            )

        # --- b-roll quiet-window law (verified against the transcript)
        if self.transcript:
            utts = speech_utterances(self.transcript)
            if utts:
                for idx, blk in enumerate(narr_blocks):
                    if blk.broll_source_start_s is None:
                        continue
                    b0 = blk.broll_source_start_s
                    b1 = b0 + min(blk.est_s, 30.0)
                    overlap = sum(max(0.0, min(b1, u1) - max(b0, u0)) for u0, u1 in utts)
                    span = max(0.1, b1 - b0)
                    if overlap / span > 0.5:
                        problems.append(
                            f"narration block #{idx + 1} b-roll at {b0:.0f}s rides ACTIVE "
                            "dialogue — pick a quiet, non-dialogue window (tell-instead-of-show "
                            "is banned)"
                        )

        # --- HOOK-FIRST LAW (Phase A pre-flight): narration ≤8s, hook done ≤32s
        if scale.hook_law:
            first_narr_start = 0.0
            found_narr = False
            sting_ok = True
            for blk in plan.acts[0].blocks:
                if isinstance(blk, PlanNarration):
                    found_narr = True
                    break
                first_narr_start += blk.dur_s
                if first_narr_start > STING_MAX_S + 0.5:
                    sting_ok = False
                    break
            if not found_narr:
                problems.append("plan has no narration in the cold open — the hook is missing")
            elif first_narr_start > HOOK_MAX_START_S:
                problems.append(
                    f"first narration starts at {first_narr_start:.1f}s > {HOOK_MAX_START_S:.0f}s — "
                    "open with the hook narration (a ≤4s climax sting may precede it)"
                )
            elif not sting_ok:
                problems.append(
                    f"cold-open leads with {first_narr_start:.1f}s of raw before narration — "
                    "only a ≤4s climax sting may precede the hook"
                )
            else:
                hook = narr_blocks[0]
                hook_end = first_narr_start + hook.est_s
                if hook_end > HOOK_MAX_END_S:
                    problems.append(
                        f"hook narration ends at {hook_end:.1f}s > {HOOK_MAX_END_S:.0f}s — "
                        "who/what/where/when must land inside 30s"
                    )

        first, last = plan.clip_span(self.source_duration_s)
        if first is None or last is None:
            problems.append("plan has no raw clips")
        else:
            span = (last - first) / self.source_duration_s
            if span < scale.min_coverage_span:
                problems.append(
                    f"clip span {span:.2f} < {scale.min_coverage_span:.2f} "
                    f"(start {first:.0f}s end {last:.0f}s)"
                )
            if first > self.source_duration_s * 0.15:
                problems.append(f"earliest clip starts at {first:.0f}s (> first 15% of raw)")
            if last < self.source_duration_s * 0.75:
                problems.append(f"latest clip ends at {last:.0f}s (< final 25% of raw)")
        return problems

    # ---------------------------------------------------------------- phase B
    def execute(self, plan: MacroPlan, timeline) -> tuple[Any, DirectorSessionReport]:
        """Expand the plan act-by-act, then run the refinement loop to target."""
        report = DirectorSessionReport(ok=False, phase="macro", plan=plan)
        steps = expand_macro_plan(plan, self.source_duration_s)
        run = self.executor.execute_plan(steps, timeline)
        report.steps_ok = run.steps_ok
        report.steps_failed = len(run.steps) - run.steps_ok
        report.composition = timeline_composition(run.timeline)
        timeline = run.timeline
        if not run.ok:
            # a failed expansion step rolled back; surface and let the loop repair
            report.failures.append(
                f"expansion step failed: {run.steps[-1].detail if run.steps else '?'}"
            )

        report.phase = "refine"
        turns = 0
        while turns < self.max_turns:
            comp = timeline_composition(timeline)
            failures = audit_invariants(comp, self.source_duration_s)
            if not failures and turns >= 1:
                break  # at least one refinement pass validated the structure
            if not failures and turns == 0 and not self._has_refinable(timeline):
                break  # invariants met pre-loop and nothing to refine
            state = director_state_summary(timeline, self.source_duration_s, self.transcript)
            nudge = (
                ""
                if not failures
                else "\n\nINVARIANT FAILURES — fix these with tool calls (add clips via "
                "timeline.insert_raw_clip from the story index, add narrator bridges via "
                "timeline.add_narration): " + "; ".join(f"{f.code}: {f.detail}" for f in failures)
            )
            try:
                steps = self.backend.propose(state + nudge, "director")
            except Exception as e:  # chain exhausted this turn
                report.failures.append(f"propose failed at turn {turns}: {e}")
                break
            turns += 1
            self.store.log_tool(
                "director_agent",
                "llm.propose",
                {"role": "director", "turn": turns},
                result_summary=f"{len(steps)} step(s) proposed",
                project_id=self.project_id,
                ok=True,
            )
            if not steps:
                if not failures:
                    break
                if turns >= 3:  # model insists it is done but invariants disagree
                    report.failures.append("model stopped proposing while invariants unmet")
                    break
                continue  # re-prompt with the failure nudge
            run = self.executor.execute_plan(steps, timeline)
            timeline = run.timeline
            report.steps_ok += run.steps_ok
            report.steps_failed += len(run.steps) - run.steps_ok
            record = getattr(self.backend, "record_results", None)
            if callable(record):
                record(run.steps and steps or steps, [sr.detail for sr in run.steps])

        report.turns = turns
        report.composition = timeline_composition(timeline)
        report.failures.extend(
            f"{f.code}: {f.detail}"
            for f in audit_invariants(report.composition, self.source_duration_s)
        )
        report.ok = not report.failures
        return timeline, report

    @staticmethod
    def _has_refinable(timeline) -> bool:
        comp = timeline_composition(timeline)
        return comp.raw_segment_count + comp.narrator_segment_count > 0
