"""Foreign narration detection — replace other channels' voices with ours.

Owner directive (2026-11-24): YouTube-sourced tapes often carry the SOURCE
channel's narration and commentary segments (intro hooks, mid-roll analysis,
outros, CTA). Our documentary replaces those beats with the Director's own
narration — the law has two mechanical halves:

1. NO foreign voice may leak into the master as raw-clip audio. The planner
   therefore never cuts a raw clip from a foreign-narration window (the same
   exclusion machinery as the violence veto).

2. Foreign narration pixels MAY still be used as muted b-roll under our
   narration blocks (``broll_audio_volume=0``), and the caption law already
   guarantees narration blocks carry ONLY our narrator's cues — so the
   foreign narrator's words can never reach the screen either.

Detection is transcript-based and deliberately conservative — two signals:

- MARKER phrases: channel-greeting / CTA / video-framing language that bodycam
  dialogue never uses ("welcome back", "subscribe", "in this video",
  "comment below", "let's get right into it"...).
- NARRATOR SPEAKER (diarization): a speaker who mostly monologues — long
  turns, rarely alternating with another speaker within 2.5s — while the
  officers/subjects converse in short alternating turns. Every range spoken
  by that speaker is foreign narration.

The detector is honest about uncertainty: it only excludes what the
transcript supports, and every excluded range carries its reason label so
the production report can show exactly which windows were replaced.
"""

from __future__ import annotations

import bisect
import re

# --- marker phrases: channel language, never bodycam dialogue ---------------
FOREIGN_NARRATION_MARKERS: tuple[tuple[str, str], ...] = (
    (
        "channel greeting",
        r"\bwelcome back\b|\bthanks for (watching|tuning in|joining)\b"
        r"|\bthank you for (watching|tuning in|joining)\b|\bwelcome (to|back to) "
        r"(the )?(channel|video|episode)\b",
    ),
    (
        "subscribe cta",
        r"\bsubscrib(e|ing|ed)\b|\bsmash that like\b|\bring the bell\b"
        r"|\blike (and|&) subscribe\b|\bdon'?t forget to (like|subscribe|share|comment)\b"
        r"|\bhit the (like|subscribe|notification)\b",
    ),
    (
        "video framing",
        r"\bin (this|today'?s|tonight'?s) (video|episode|clip|documentary)\b"
        r"|\bon (this|today'?s) (video|episode)\b|\bthis (video|episode|documentary)\b"
        r"|\btoday'?s (video|episode)\b|\bthis clip (comes|is) from\b",
    ),
    (
        "channel reference",
        r"\b(the|this|our|my) channel\b|\bnew videos? (every|each)\b"
        r"|\bcheck out (the|our) channel\b|\bfull (video|version) (is )?(on|at) (our|the)\b",
    ),
    (
        "comment prompt",
        r"\blet me know (in the comments|what you think|below)\b|\bdrop a comment\b"
        r"|\bcomment (below|down below)\b|\bin the comments (section|below)\b"
        r"|\blet'?s (get )?(right )?into (it|the video|today)\b",
    ),
    (
        "news framing",
        r"\bbreaking (news|tonight)\b|\bwe begin (tonight|this evening)\b"
        r"|\bgood evening\b|\bcoming up (tonight|next)\b",
    ),
)

_COMPILED_MARKERS = tuple(
    (label, re.compile(pat, re.IGNORECASE)) for label, pat in FOREIGN_NARRATION_MARKERS
)

# --- narrator-speaker heuristics (diarized segments) -------------------------
NARRATOR_MIN_TOTAL_S = 45.0  # a narrator carries the video, not one line
NARRATOR_MIN_MEAN_TURN_S = 8.0  # monologue turns, not conversational beats
NARRATOR_MAX_ALTERNATION = 0.35  # <35% of turns start right after ANOTHER speaker
ALTERNATION_GAP_S = 2.5


def _merge_ranges(
    ranges: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    if not ranges:
        return []
    out = sorted(ranges)
    merged: list[list[tuple[float, float, str]]] = [[out[0]]]
    last = merged[-1][0]
    for a, b, label in out[1:]:
        if a <= last[1]:
            last = (last[0], max(last[1], b), last[2])
            merged[-1][0] = last
        else:
            merged.append([(a, b, label)])
            last = (a, b, label)
    return [r[0] for r in merged]


def _segment_turns(transcript: dict) -> list[tuple[float, float, str, str]]:
    """(start, end, speaker, text) from diarized segments; falls back to words."""
    turns: list[tuple[float, float, str, str]] = []
    for seg in transcript.get("segments", []) or []:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        spk = str(seg.get("speaker", "?"))
        turns.append((float(seg.get("start_s", 0.0)), float(seg.get("end_s", 0.0)), spk, text))
    if turns:
        return sorted(turns, key=lambda t: t[0])
    # word fallback: group words into pseudo-turns by speaker runs
    words = [w for w in transcript.get("words", []) if w.get("text")]
    cur_start: float = 0.0
    cur_end: float = 0.0
    cur_spk: str = ""
    cur_text: str = ""
    cur_open = False
    for w in sorted(words, key=lambda w: float(w.get("start_s", 0.0))):
        spk = str(w.get("speaker", "?"))
        start, end = float(w.get("start_s", 0.0)), float(w.get("end_s", 0.0))
        if cur_open and cur_spk == spk and start - cur_end <= 1.8:
            cur_end = end
            cur_text += " " + str(w["text"]).strip()
        else:
            if cur_open:
                turns.append((cur_start, cur_end, cur_spk, cur_text))
            cur_start, cur_end, cur_spk = start, end, spk
            cur_text = str(w["text"]).strip()
            cur_open = True
    if cur_open:
        turns.append((cur_start, cur_end, cur_spk, cur_text))
    return sorted(turns, key=lambda t: t[0])


def detect_narrator_speaker(transcript: dict) -> str | None:
    """The one speaker that monologues while everyone else converses.

    Bodycam dialogue alternates: officers ask, subjects answer, gaps are
    short. A source channel's narrator runs long solo blocks over muted
    scene audio. Returns the speaker id, or None when no speaker fits the
    profile (the honest answer for a pure raw bodycam release).
    """
    turns = _segment_turns(transcript)
    if len(turns) < 6 or len({t[2] for t in turns}) < 2:
        return None
    speakers: dict[str, list[tuple[float, float, str]]] = {}
    for a, b, spk, text in turns:
        speakers.setdefault(spk, []).append((a, b, text))
    best: tuple[str, float] | None = None
    for spk, utts in speakers.items():
        total = sum(b - a for a, b, _t in utts)
        if total < NARRATOR_MIN_TOTAL_S:
            continue
        mean_turn = total / len(utts)
        if mean_turn < NARRATOR_MIN_MEAN_TURN_S:
            continue
        # alternation: fraction of this speaker's turns that start within
        # ALTERNATION_GAP_S of a DIFFERENT speaker's turn ENDING
        ends = sorted(b for a, b, other, _t in turns if other != spk)
        alt = 0
        for a, _b, _t in utts:
            i = bisect.bisect_left(ends, a)
            if i and a - ends[i - 1] <= ALTERNATION_GAP_S:
                alt += 1
        frac = alt / len(utts)
        if frac > NARRATOR_MAX_ALTERNATION:
            continue
        score = total * (1.0 - frac)
        if best is None or score > best[1]:
            best = (spk, score)
    return best[0] if best else None


def foreign_narration_ranges(
    transcript: dict, pad_s: float = 1.0
) -> list[tuple[float, float, str]]:
    """Source-clock windows that carry a source-channel narrator's voice.

    Marker hits veto their utterance window (padded); a detected narrator
    speaker vetoes ALL their speech windows. Output is merged and safe to
    hand to the planner as an audio-exclusion list.
    """
    turns = _segment_turns(transcript)
    ranges: list[tuple[float, float, str]] = []
    narrator = detect_narrator_speaker(transcript)
    for a, b, spk, text in turns:
        if narrator is not None and spk == narrator:
            ranges.append((max(0.0, a - pad_s), b + pad_s, f"narrator speaker S{spk}"))
            continue
        for label, regex in _COMPILED_MARKERS:
            if regex.search(text):
                ranges.append((max(0.0, a - pad_s), b + pad_s, label))
                break
    return _merge_ranges(ranges)
