"""Deterministic multi-act documentary fallback (no-LLM path).

Mandatory change #2: when no LLM rung is reachable, planning must NOT emit a
1-minute slice or an 8s-grained full dump — it samples the whole story arc.

HOOK-FIRST LAW (owner rule 2026-11-19 — "think like a real pro editor"):
the old layout opened with two ~28s raw clips and let narration drift in at
~50s — a guaranteed bail-out. The new layout opens COLD:

  t=0.0  ≤4s climax sting (raw audio ON — the ear-catch)
  t≈4s   HOOK narration over muted climax b-roll — narration starts ≤8s,
         and the hook script itself lands the who/what/where/when so the
         viewer has full context well inside 30s
  then   chronological 6-act body (narration bridges bracket speech-anchored
         raw clips), outro close.

Clips are speech-anchored (top speech intervals per act window, so dead air
falls out by construction) and sized to land the 600-900s target window for
long sources. Narrator bridges are generic, fact-free scripting unless seed
scripts (e.g. a previous LLM-authored plan) are supplied — the deterministic
path never invents case facts beyond the `case_facts` line the caller hands it.
"""

from __future__ import annotations

from dataclasses import dataclass

from director_os.agent_runtime.doc_scale import DOCUMENTARY_MIN_SOURCE_S
from director_os.otio_engine.contract import (
    CardMeta,
    NarratorClipMeta,
    RawClipMeta,
)
from director_os.otio_engine.timeline import ClipSpec, GapSpec, TimelineSpec, TrackSpec

FALLBACK_MIN_SOURCE_S = DOCUMENTARY_MIN_SOURCE_S  # below this the v1 auto-cut stays
STING_MAX_S = 4.0  # the cold-open raw sting before narration — narration ≤8s
HOOK_MAX_START_S = 8.0  # first narration block must START by this master time
HOOK_MAX_END_S = 32.0  # ... and the hook (context) must END by this time
NARRATION_WPS = 3.3  # MEASURED pace: Brian +14% with the pause-compression chain
# (the plan's estimate must match reality — the post-TTS retime closes the
# residual gap, and the floor top-up below adds the safety margin)
HOOK_REGION = (0.60, 0.80)  # climax hooks come from later in the video

# Act layouts by act count — every layout covers the SAME full-arc semantic
# sequence (contact -> investigation -> escalation -> arrest -> aftermath ->
# outcome) compressed to the acts the source's LENGTH can carry. bridge_key /
# close_key point into the generic 6-act bridges (the merged acts inherit the
# opening bridge of their first member and the closing line of their last).
ACT_SETS: dict[int, tuple[tuple[str, float, float, str, str], ...]] = {
    3: (
        (
            "Act 1: Contact & Investigation",
            0.02,
            0.38,
            "Act 1: Arrival & Contact",
            "Act 2: Investigation & Friction",
        ),
        ("Act 2: Escalation & Arrest", 0.38, 0.74, "Act 3: Escalation", "Act 4: The Arrest"),
        (
            "Act 3: Aftermath & Outcome",
            0.74,
            1.00,
            "Act 5: Aftermath & Booking",
            "Act 6: Verified Outcome",
        ),
    ),
    4: (
        (
            "Act 1: Arrival & Contact",
            0.02,
            0.25,
            "Act 1: Arrival & Contact",
            "Act 1: Arrival & Contact",
        ),
        (
            "Act 2: Investigation & Friction",
            0.25,
            0.50,
            "Act 2: Investigation & Friction",
            "Act 2: Investigation & Friction",
        ),
        ("Act 3: Escalation & Arrest", 0.50, 0.74, "Act 3: Escalation", "Act 4: The Arrest"),
        (
            "Act 4: Aftermath & Outcome",
            0.74,
            1.00,
            "Act 5: Aftermath & Booking",
            "Act 6: Verified Outcome",
        ),
    ),
    5: (
        (
            "Act 1: Arrival & Contact",
            0.02,
            0.15,
            "Act 1: Arrival & Contact",
            "Act 1: Arrival & Contact",
        ),
        (
            "Act 2: Investigation & Friction",
            0.15,
            0.38,
            "Act 2: Investigation & Friction",
            "Act 2: Investigation & Friction",
        ),
        ("Act 3: Escalation", 0.38, 0.58, "Act 3: Escalation", "Act 3: Escalation"),
        ("Act 4: The Arrest", 0.58, 0.80, "Act 4: The Arrest", "Act 4: The Arrest"),
        (
            "Act 5: Aftermath & Outcome",
            0.80,
            1.00,
            "Act 5: Aftermath & Booking",
            "Act 6: Verified Outcome",
        ),
    ),
}
ACT_SETS[6] = tuple(
    (name, lo, hi, name, name)
    for name, lo, hi in (
        ("Act 1: Arrival & Contact", 0.02, 0.15),
        ("Act 2: Investigation & Friction", 0.15, 0.38),
        ("Act 3: Escalation", 0.38, 0.58),
        ("Act 4: The Arrest", 0.58, 0.74),
        ("Act 5: Aftermath & Booking", 0.74, 0.88),
        ("Act 6: Verified Outcome", 0.88, 1.00),
    )
)

NARRATOR_BRIDGES = {
    "Cold Open": (
        "The bodycam was already rolling. What it caught next turned a routine "
        "night shift into a criminal case — and every second of it is on tape. "
        "Stay with the first thirty seconds, because the detail that decides "
        "this entire case appears before anyone says a word."
    ),
    "Act 1: Arrival & Contact": (
        "It starts like any other call. Small talk, routine questions — and the "
        "first small details that will not add up later. Watch how casually this "
        "begins; nothing about the next forty minutes is casual. The paperwork "
        "says one thing, the camera says another, and the gap between them is "
        "where this whole case lives."
    ),
    "Act 2: Investigation & Friction": (
        "Now the officers start comparing what they are seeing with what they "
        "are being told. The answers get a little too rehearsed. The excuses "
        "are not holding — and everyone in this frame knows it. Notice how "
        "the same question gets asked three different ways, and how the "
        "answer drifts a little further each time."
    ),
    "Act 3: Escalation": (
        "This is the turning point. The questions get sharper, the posture "
        "changes, and the tone flips from patience to procedure. From here, "
        "the encounter stops being a conversation. Everything said in these "
        "minutes gets read back later, line by line, under much less "
        "forgiving lighting."
    ),
    "Act 4: The Arrest": (
        "The arrest itself — caught in full, from both sides of the handcuffs. "
        "Listen closely to what gets said in these seconds; it is the part "
        "every report afterwards has to match. And listen for what is NOT "
        "said too — the silence here is just as loud as anything on the "
        "report."
    ),
    "Act 5: Aftermath & Booking": (
        "What follows the cuffs matters just as much: transport, processing, "
        "and the tests that lock the facts in place — the part of the record "
        "that decides how this case actually ends. Numbers do not get nervous "
        "and paperwork does not forget; this is where the story stops being "
        "about what anyone claims."
    ),
    "Act 6: Verified Outcome": (
        "Booking finishes, the tests come back, and the record closes the loop "
        "on everything you just watched. This is the verified outcome, exactly "
        "as the case files state it. Every claim from the last forty minutes "
        "now meets the one thing it cannot argue with — the result."
    ),
}

# Per-act CLOSE lines — every one distinct (identical TTS text repeats on
# screen and viewers notice). They hand momentum to the next act instead of
# restating it.
TRANSITION_BRIDGES = {
    "Act 1: Arrival & Contact": (
        "Nothing about these first minutes looks unusual — and that is exactly "
        "what makes the next part worth watching. Keep an eye on the small "
        "details, because the story turns on them. The next stretch of tape "
        "is where the account of this night starts to come apart at the "
        "seams, quietly at first."
    ),
    "Act 2: Investigation & Friction": (
        "By this point the account is starting to strain against what the "
        "camera shows. What comes next is the moment the patience runs out "
        "and the procedure takes over. Watch the body language change before "
        "anyone raises their voice — professionals signal the shift early."
    ),
    "Act 3: Escalation": (
        "Once that line is crossed there is no walking it back. The next "
        "stretch of tape is the part every report afterwards has to match, "
        "second by second. From here forward, every sentence spoken on this "
        "camera becomes evidence, and everyone involved seems to know it."
    ),
    "Act 4: The Arrest": (
        "With the cuffs on, the case shifts from the roadside to the "
        "paperwork — transport, processing, and the tests that turn suspicion "
        "into numbers. The roadside conversation is over; what replaces it "
        "is measured, recorded, and timestamped."
    ),
    "Act 5: Aftermath & Booking": (
        "The final stretch answers the only question that matters: how does "
        "the record actually close on this? The official outcome is next, "
        "and it lands differently once you have watched how the night "
        "actually unfolded."
    ),
    "Act 6: Verified Outcome": (
        "That is the full arc, start to finish, exactly as the tape recorded "
        "it. If this case told you something the headlines didn't, the next "
        "one is already on the channel — and it opens with the same rule "
        "this one proved: the camera keeps rolling whether anyone behaves "
        "or not."
    ),
}


# SHORT bridge variants — the deterministic short-documentary sizing law.
# Full-length scripts (8 x ~50-70 words) blow the narration-ratio invariant on
# short masters: a 480s tape plans a ~134-160s master and the fixed scripts
# alone carry ~108s of narration (ratio 0.67 > 0.52) — the fallback then
# failed its own retention pre-flight ("narration ratio=0.67", the any-length
# law broken). Same act keys, same tone, every line distinct, fewer words:
# the ratio law is enforced by construction instead of hoped for.
SHORT_NARRATOR_BRIDGES = {
    "Cold Open": (
        "The bodycam was already rolling. What it caught next turned a routine "
        "shift into a criminal case — and the detail that decides everything "
        "appears early."
    ),
    "Act 1: Arrival & Contact": (
        "It starts like any routine call — small talk, routine questions. But "
        "the first details already refuse to add up, and the tape keeps rolling."
    ),
    "Act 2: Investigation & Friction": (
        "Now the officers compare what they see with what they were told. The "
        "answers get rehearsed, and everyone in this frame knows it."
    ),
    "Act 3: Escalation": (
        "This is the turning point — sharper questions, a changing posture. The "
        "conversation ends here and the procedure takes over."
    ),
    "Act 4: The Arrest": (
        "The arrest caught in full, from both sides of the handcuffs — the part "
        "every report afterwards has to match."
    ),
    "Act 5: Aftermath & Booking": (
        "After the cuffs come transport, processing, and the tests that decide "
        "how this case actually ends."
    ),
    "Act 6: Verified Outcome": (
        "Booking finishes, the tests come back, and the record closes the loop "
        "on everything you just watched — the verified outcome, exactly as filed."
    ),
}

SHORT_TRANSITION_BRIDGES = {
    "Act 1: Arrival & Contact": (
        "Nothing here looks unusual yet — which is exactly what makes the next "
        "stretch worth watching."
    ),
    "Act 2: Investigation & Friction": (
        "Watch the answers drift as the same question comes back — the account "
        "starts straining against the camera."
    ),
    "Act 3: Escalation": (
        "Once that line is crossed there is no walking it back — every sentence "
        "from here is evidence."
    ),
    "Act 4: The Arrest": (
        "The cuffs turn a roadside conversation into measured, timestamped evidence."
    ),
    "Act 5: Aftermath & Booking": (
        "The final stretch answers the only question that matters: how does the "
        "record actually close?"
    ),
    "Act 6: Verified Outcome": (
        "If this case told you something the headlines didn't, the next one is "
        "already on the channel."
    ),
}


@dataclass
class _Block:
    kind: str  # raw | narration
    source_start_s: float
    dur_s: float
    script: str = ""
    act: str = ""


def _hook_script(case_facts: str, seed: str | None, short: bool = False) -> str:
    """The cold-open hook: 3 punchy lines carrying the who/what/where/when.

    Seeds (previously LLM-authored narration) win; the fact line composes a
    clean hook otherwise. Never editorializes guilt — public-record framing.
    """
    if seed:
        return " ".join(seed.split())
    facts = " ".join((case_facts or "").split())
    if short:
        if facts:
            return (
                f"{facts} The bodycam never stopped rolling — stay for the "
                "detail that decides this entire case."
            )
        return SHORT_NARRATOR_BRIDGES["Cold Open"]
    if facts:
        return (
            f"{facts} The bodycam never stopped rolling — and what it captured "
            "turned a routine shift into a criminal case. Every second of it "
            "is on tape."
        )
    return NARRATOR_BRIDGES["Cold Open"]


def _narr_dur(script: str, *, lo: float = 12.0, hi: float = 34.0) -> float:
    """Planned narration duration from word count at the retention-law pace.

    The post-TTS retime pass replaces this with the real clip duration; the
    estimate only has to be close enough for cards and invariants.
    """
    words = max(1, len(script.split()))
    return max(lo, min(hi, words / NARRATION_WPS + 1.0))


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


def _act_windows(source_duration_s: float):
    """Act layout scaled to the source length (doc_scale act count)."""
    from director_os.agent_runtime.doc_scale import doc_scale

    n = doc_scale(source_duration_s)
    n_acts = max(3, min(6, round((n.target_lo + n.target_hi) / 2 / 110)))
    return ACT_SETS.get(n_acts, ACT_SETS[6])


def plan_fallback_blocks(
    source_duration_s: float,
    transcript: dict,
    *,
    seed_scripts: dict[str, str] | None = None,
    seed_raw_ranges: list[tuple[float, float]] | None = None,
    case_facts: str = "",
) -> list[_Block]:
    """Pure deterministic planner: the full-arc block list (HOOK-FIRST law).

    v0.5 layout: ≤4s climax sting (raw audio) -> hook narration over muted
    climax b-roll (first narration ≤8s, context lands ≤30s) -> chronological
    multi-act body scaled to the SOURCE LENGTH (narration bridge leads each
    act, a distinct transition line closes it, speech-anchored raw clips
    between), outro close. Seed scripts (a previous LLM-authored plan) and
    seed raw ranges win over the generic bridges / speech-anchored picks when
    supplied.

    CONTENT SAFETY LAW (ADR-0013, owner directive 2026-11-24): the case gate
    runs first (violent/off-menu cases are refused, never planned), and no
    raw clip is cut from a transcript window carrying a violent marker —
    candidate intervals, seed ranges, hook fallbacks, and the floor top-up
    all respect the veto ranges.

    FOREIGN NARRATION LAW (owner directive 2026-11-24): YouTube-sourced tapes
    carry the source channel's narration/commentary. Every foreign-narration
    window is excluded from RAW (audio-on) clips the same way the violence
    veto excludes windows — the beats are replaced by our own narration,
    with the source pixels available only as muted b-roll.
    """
    from director_os.agent_runtime.content_policy import (
        ContentPolicyError,
        clip_safe_span,
        drop_vetoed,
        evaluate_case,
        first_safe_window,
        vetoed_ranges,
    )
    from director_os.agent_runtime.doc_scale import doc_scale
    from director_os.agent_runtime.foreign_media import foreign_narration_ranges

    verdict = evaluate_case(case_facts)
    if not verdict.allowed:
        raise ContentPolicyError(verdict.reason)
    veto = vetoed_ranges(transcript)
    foreign = foreign_narration_ranges(transcript)
    # audio-exclusion list: violence vetoes + foreign narrator windows —
    # every RAW (audio-on) clip decision below respects this list
    excluded = [*veto, *foreign]

    scale = doc_scale(source_duration_s)
    target = max(scale.target_lo, min(scale.target_hi, source_duration_s * 0.28))
    # SHORT-DOC LAW: below a ~400s master the full-length bridge scripts carry
    # more narration than the ratio law allows — switch to the short variants
    # (hook/outro keep the arc) and let the estimate bounds follow. The
    # threshold rides the 50-65% keep band (owner-ratified 2026-11-25): an
    # 8-min tape now plans a ~240s master, where full bridges measured 0.58
    # narration ratio (the 251s tape-1 lineage) — short bridges keep it legal.
    short_doc = target < 400.0
    narrators = SHORT_NARRATOR_BRIDGES if short_doc else NARRATOR_BRIDGES
    transitions = SHORT_TRANSITION_BRIDGES if short_doc else TRANSITION_BRIDGES
    narr_lo = 5.0 if short_doc else 14.0
    narr_hi = 22.0 if short_doc else 34.0
    intervals = drop_vetoed(_speech_intervals(transcript), excluded)
    seeds = {k: " ".join(v.split()) for k, v in (seed_scripts or {}).items() if v and v.strip()}
    seed_ranges = [r for r in (seed_raw_ranges or []) if r[1] > r[0]]
    act_windows = _act_windows(source_duration_s)

    blocks: list[_Block] = []

    # --- COLD OPEN (hook-first): climax sting + hook narration ≤8s in
    hook_a, hook_b = HOOK_REGION
    hooks = _top_intervals(intervals, (hook_a, hook_b), source_duration_s, 4)[:2]
    if hooks:
        climax = hooks[0]
    else:
        safe = clip_safe_span(source_duration_s * hook_a, 24.0, excluded) or first_safe_window(
            source_duration_s, 24.0, excluded
        )
        climax = safe
    # 1) the sting: ≤STING_MAX_S of the climax WITH its raw audio — the ear-catch
    sting_start = max(0.0, climax[0])
    # a veto-trimmed fallback climax may be shorter than the sting — the
    # sting may never overrun the safe window back into vetoed footage
    sting_dur = min(STING_MAX_S, max(0.5, climax[1]))
    blocks.append(_Block("raw", sting_start, sting_dur, act="Cold Open"))
    # 2) the hook narration, b-roll riding the climax moment (muted bed)
    hook = _hook_script(case_facts, seeds.get("hook") or seeds.get("Cold Open"), short=short_doc)
    blocks.append(
        _Block("narration", climax[0], _narr_dur(hook, lo=narr_lo, hi=26.0), hook, "Cold Open")
    )
    # 3) one more short climax beat, audio on, to ride out of the cold open
    if len(hooks) > 1 and scale.target_lo >= 240.0:
        b_start = max(0.0, hooks[1][0])
        blocks.append(
            _Block("raw", b_start, min(24.0, max(10.0, hooks[1][1] - hooks[1][0])), act="Cold Open")
        )

    # --- acts: narration bridge LEADS each act, a distinct transition line
    #     CLOSES it (context before footage, momentum after), speech-anchored
    #     raw clips between; the last act's closing line is the outro
    n_acts = len(act_windows)
    narration_total = _narr_dur(
        _hook_script(
            case_facts,
            (seed_scripts or {}).get("hook") or (seed_scripts or {}).get("Cold Open"),
            short=short_doc,
        ),
        lo=narr_lo,
        hi=26.0,
    )
    narration_total += sum(
        _narr_dur(seeds.get(bridge_key) or narrators[bridge_key], lo=narr_lo, hi=narr_hi)
        + _narr_dur(transitions[close_key], lo=narr_lo, hi=30.0 if not short_doc else 22.0)
        for _name, _lo, _hi, bridge_key, close_key in act_windows
    )
    raw_wanted = max(scale.min_final_s * 0.35, target - narration_total)
    # ratio fit BOTH ways (narration-ratio law 0.30..0.52): the old one-sided
    # cap only stopped the ratio falling BELOW 0.30 — narration-heavy short
    # docs sailed over 0.52 and failed retention. Floor raw so narration
    # shares <= 0.48, cap it so the share stays >= 0.32.
    raw_floor_for_ratio = narration_total * (1.0 - 0.48) / 0.48
    raw_cap_for_ratio = narration_total * (1.0 - 0.32) / 0.32
    raw_total = max(raw_wanted, raw_floor_for_ratio, scale.min_final_s * 0.35)
    raw_total = min(raw_total, raw_cap_for_ratio)
    per_act = raw_total / n_acts
    clips_per_act = max(1, min(3, round(per_act / 35.0)))
    per_clip = min(scale.clip_max_s, max(16.0, per_act / clips_per_act))

    for name, lo, hi, bridge_key, close_key in act_windows:
        bridge = seeds.get(bridge_key) or narrators[bridge_key]
        blocks.append(
            _Block(
                "narration",
                source_duration_s * lo,
                _narr_dur(bridge, lo=narr_lo, hi=narr_hi),
                bridge,
                name,
            )
        )
        act_ranges = _seed_ranges_in(
            seed_ranges, lo, hi, source_duration_s, max_ranges=clips_per_act, veto_ranges=excluded
        ) or _top_intervals(intervals, (lo, hi), source_duration_s, clips_per_act)
        if act_ranges:
            for a, b in act_ranges:
                # budget-driven length: the speech anchor justifies the window,
                # the act budget sizes it (long-tail dead air falls out)
                d = min(per_clip, max(per_clip - 3.0, b - a + 1.5))
                raw_span = clip_safe_span(max(0.0, a - 0.5), d, excluded)
                if raw_span is None:
                    continue
                blocks.append(_Block("raw", raw_span[0], raw_span[1], act=name))
        else:  # fixed grid fallback inside the window
            w = (hi - lo) * source_duration_s
            for k in range(clips_per_act):
                start = lo * source_duration_s + k * (w / (clips_per_act + 0.2))
                grid_span = clip_safe_span(start, min(per_clip, w / clips_per_act), excluded)
                if grid_span is not None:
                    blocks.append(_Block("raw", grid_span[0], grid_span[1], act=name))
        closing = transitions[close_key]
        if close_key == "Act 6: Verified Outcome":
            closing = seeds.get("outro") or seeds.get(close_key) or closing
        blocks.append(
            _Block(
                "narration",
                source_duration_s * hi,
                _narr_dur(closing, lo=narr_lo, hi=40.0 if not short_doc else 22.0),
                closing,
                name,
            )
        )

    # --- floor top-up: rounding + skipped cold-open beats + the TTS re-time
    #     shrink (real Brian speaks faster than the plan estimate) can leave
    #     the SHIPPED master under the length-scaled floor — extend the raw
    #     blocks proportionally with a measured-pace safety margin. Raw spans
    #     are speech-anchored exacts; a few extra seconds ride into the
    #     following pause, never a silent stall.
    floor = scale.min_final_s
    aim = floor + max(25.0, floor * 0.05)
    for _pass in range(3):
        total = sum(blk.dur_s for blk in blocks)
        if total >= aim:
            break
        raws = [blk for blk in blocks if blk.kind == "raw"]
        raw_sum = sum(r.dur_s for r in raws)
        if not raws:
            break
        need = aim - total
        factor = min(1.35, (raw_sum + need) / raw_sum)
        for blk in raws:
            # the extension may never cross into a veto or foreign-narration range
            grown = min(75.0, blk.dur_s * factor)
            grown_safe = clip_safe_span(blk.source_start_s, grown, excluded)
            blk.dur_s = min(grown, grown_safe[1]) if grown_safe else blk.dur_s
    return blocks


def _seed_ranges_in(
    seed_ranges: list[tuple[float, float]],
    lo: float,
    hi: float,
    source_duration_s: float,
    max_ranges: int = 3,
    veto_ranges: list[tuple[float, float, str]] | None = None,
) -> list[tuple[float, float]]:
    """Seed raw clips (source-clock) that fall inside the [lo, hi] act window.

    Seed ranges come from a previous LLM-authored plan and bypass the
    interval filter, so every seed is trimmed off the veto ranges before it
    can become a raw clip (a vetoed seed shrinks or drops entirely).
    """
    a0, a1 = lo * source_duration_s, hi * source_duration_s
    picked = [
        (max(a, a0), min(b, a1))
        for a, b in seed_ranges
        if b > a0 and a < a1 and min(b, a1) - max(a, a0) >= 4.0
    ]
    if veto_ranges:
        from director_os.agent_runtime.content_policy import clip_safe_span

        trimmed = [s for s in (clip_safe_span(a, b - a, veto_ranges) for a, b in picked) if s]
        picked = [(a, a + d) for a, d in trimmed]
    picked.sort(key=lambda p: -(p[1] - p[0]))
    return sorted(picked[:max_ranges])


def build_fallback_documentary(
    asset_id: str,
    source_duration_s: float,
    transcript: dict,
    fps: int,
    width: int,
    height: int,
    project_id: str = "fallback",
    *,
    seed_scripts: dict[str, str] | None = None,
    seed_raw_ranges: list[tuple[float, float]] | None = None,
    case_facts: str = "",
) -> TimelineSpec:
    """The deterministic 6-act documentary timeline spec (QA-invariant-safe)."""
    blocks = plan_fallback_blocks(
        source_duration_s,
        transcript,
        seed_scripts=seed_scripts,
        seed_raw_ranges=seed_raw_ranges,
        case_facts=case_facts,
    )

    master: list[ClipSpec | GapSpec] = []
    for i, b in enumerate(blocks):
        if b.kind == "raw":
            end = min(b.source_start_s + b.dur_s, source_duration_s - 0.5)
            # the ≤4s cold-open sting keeps its planned length (the 8s floor
            # would push the hook narration past the 8s hook law)
            floor_dur = 8.0 if b.dur_s > STING_MAX_S else min(b.dur_s, STING_MAX_S)
            dur = max(floor_dur, end - b.source_start_s)
            master.append(
                ClipSpec(
                    name=f"raw_{i}",
                    source_start_s=b.source_start_s,
                    dur_s=dur,
                    meta=RawClipMeta(asset_id=asset_id, act_name=b.act),
                )
            )
        else:
            # b-roll must FIT the source: a window starting at/after EOF slices
            # zero frames (the acceptance run caught the outro narration at
            # 2627.08s = source end) — clamp the start so the window lands
            broll_start = max(0.0, min(b.source_start_s, max(0.0, source_duration_s - b.dur_s)))
            master.append(
                ClipSpec(
                    name=f"narration_{i}",
                    source_start_s=broll_start,
                    dur_s=b.dur_s,
                    meta=NarratorClipMeta(
                        script_text=b.script,
                        broll_asset_id=asset_id,
                        broll_range=(broll_start, broll_start + b.dur_s),
                        broll_audio_volume=0.0,
                        act_name=b.act,
                    ),
                )
            )

    # hook b-roll lives at the CLIMAX moment the sting came from (muted bed);
    # every other narration block rides its own planned source window

    # cards: location_intro on the first body act's narration, evidence_fact
    # mid-doc, case_outcome on the Outro — placed at the narration block's
    # start + 0.5s. Targets derive from the SCALED act layout (any length).
    act_windows = _act_windows(source_duration_s)
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

    body_acts = [name for name, _lo, _hi, _b, _c in act_windows]
    first_body = body_acts[0]
    mid_body = body_acts[len(body_acts) // 2]
    last_body = body_acts[-1]
    _card_at(
        first_body,
        "location_intro",
        "CASE FILE",
        "Bodycam footage — public record",
        4.0,
    )
    _card_at(
        mid_body,
        "evidence_fact",
        "THE EVIDENCE",
        "From the raw transcript",
        4.0,
    )
    _card_at(last_body, "case_outcome", "CASE OUTCOME", "Per official records", 5.0)

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
