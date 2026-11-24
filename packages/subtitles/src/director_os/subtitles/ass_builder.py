"""2-row speaker-colored `.ass` builder — the COMMITTED v4/v6 caption law.

History: the v5 regression burned Roboto 20pt under a 1920x1080 PlayRes onto
1280x720 masters (~13px illegible text, commit 0d56deb "caption truth on the
master clock"). The approved typography recovered from that commit is law:

- DejaVu Sans 40pt @ PlayRes 1280x720 (libass scales proportionally to any
  render size — 60px on a 1080p master; always far above the QA >=26px gate)
- role colors: Officer=white / Suspect=gold / Narrator=cyan; UNKNOWN speakers
  keep a deterministic palette (hash) so multi-party scenes stay distinguishable
- two lanes for simultaneous speakers (Row 1 `MarginV=46`, Row 2 `MarginV=96`),
  never clamped onto one row
- karaoke `\\k` centisecond beats from aligned word timings, `\\fad(40,30)`
- outline 2.9 / shadow 1.6 / spacing 0.4 / margins L+R 80 / bold
- cue law: <=5 words / <=27 chars / break on a >1s word gap (caption_v4)
- flash guard: a cue shorter than 0.22s is a subliminal flash — it is extended
  to 0.30s (word-preserving variant of the v6 drop rule; nothing is lost)
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pysubs2
from pydantic import BaseModel, Field
from pysubs2 import Alignment

LANE1_MARGIN_V = 46
LANE2_MARGIN_V = 96
# committed v4/v6 typography (see module docstring)
FONT_NAME = "DejaVu Sans"
FONT_SIZE = 40
PLAY_RES_X = 1280
PLAY_RES_Y = 720
OUTLINE = 2.9
SHADOW = 1.6
SPACING = 0.4
MARGIN_LR = 80
FLASH_MIN_S = 0.30

# committed v6 role colors (ASS PrimaryColour is &HAABBGGRR; here as RGB hex)
ROLE_COLORS: dict[str, str] = {
    "officer": "#FFFFFF",  # white
    "suspect": "#FFD84D",  # gold (&H004DD8FF in the committed header)
    "narrator": "#70E4FF",  # cyan (&H00FFE470 in the committed header)
}

PALETTE = [
    "#FFFFFF",
    "#3EE6C1",
    "#FFD166",
    "#7EB6FF",
    "#FF9E7D",
    "#C792EA",
]

# cue law (caption_v4): a caption beat is a short readable chunk
MAX_CUE_WORDS = 5
MAX_CUE_CHARS = 27
CUE_GAP_BREAK_S = 1.0

PROFANITY = {
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "bastard",
    "dick",
    "cunt",
}


class WordTiming(BaseModel):
    word: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class CaptionBeat(BaseModel):
    """One caption cue: 3–5 words, one speaker, absolute ms timeline."""

    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: str = "Unknown"
    words: list[WordTiming] = Field(default_factory=list)


class SubtitleBuildError(RuntimeError):
    pass


def _speaker_color(speaker: str) -> str:
    """Committed role colors first; unknown speakers keep the hash palette."""
    role = ROLE_COLORS.get(speaker.strip().lower())
    if role:
        return role
    idx = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % len(PALETTE)
    return PALETTE[idx]


def chunk_cue_words(words: Sequence[dict]) -> list[list[dict]]:
    """The caption_v4 cue law: <=5 words / <=27 chars / break on a >1s gap.

    `words` are dicts with `text`, `start_s`, `end_s` and an optional `speaker`
    (a speaker change always starts a new cue). Returns the chunk list.
    """
    chunks: list[list[dict]] = []
    run: list[dict] = []
    run_chars = 0
    speaker = None
    for w in words:
        text = str(w.get("text") or w.get("word") or "").strip()
        if not text:
            continue
        spk = str(w.get("speaker") or "") or None
        gap = float(w.get("start_s", 0.0)) - float(run[-1].get("end_s", 0.0)) if run else 0.0
        if run and (
            (spk is not None and speaker is not None and spk != speaker)
            or gap > CUE_GAP_BREAK_S
            or len(run) >= MAX_CUE_WORDS
            or run_chars + len(text) + 1 > MAX_CUE_CHARS
        ):
            chunks.append(run)
            run, run_chars = [], 0
        if run:
            run_chars += len(text) + 1
        else:
            run_chars = len(text)
        run.append({**w, "text": text})
        speaker = spk if spk is not None else speaker
    if run:
        chunks.append(run)
    return chunks


def _mask_profanity(text: str) -> str:
    out = []
    for token in text.split():
        stripped = token.strip(",.!?\"'").lower()
        if stripped in PROFANITY:
            out.append("*" * max(3, len(token)))
        else:
            out.append(token)
    return " ".join(out)


def _assign_rows(beats: Sequence[CaptionBeat]) -> list[int]:
    """Row assignment: different-speaker overlap -> opposite lane; same speaker sticks
    to their last lane; otherwise Lane 1 first."""
    rows: list[int] = []
    lane_free_at = {1: -1, 2: -1}
    speaker_lane: dict[str, int] = {}
    for b in beats:
        # same speaker keeps their lane if it is free
        lane = speaker_lane.get(b.speaker)
        if lane is not None and lane_free_at[lane] <= b.start_ms:
            rows.append(lane)
            lane_free_at[lane] = b.end_ms
            continue
        # lane 1 if free, else lane 2 if free, else the lane with the earlier free time
        if lane_free_at[1] <= b.start_ms:
            lane = 1
        elif lane_free_at[2] <= b.start_ms:
            lane = 2
        else:
            lane = 1 if lane_free_at[1] <= lane_free_at[2] else 2
        rows.append(lane)
        lane_free_at[lane] = b.end_ms
        speaker_lane[b.speaker] = lane
    return rows


def _karaoke_text(beat: CaptionBeat) -> str:
    words = beat.words or [WordTiming(word=w, start_ms=0, end_ms=0) for w in beat.text.split()]
    if len(words) < 2:
        return beat.text
    total = max(1, beat.end_ms - beat.start_ms)
    weights = [max(1, len(w.word)) for w in words]
    tw = sum(weights)
    tags = []
    for k, w in enumerate(words):
        if w.end_ms > w.start_ms:
            cs = max(1, round((w.end_ms - w.start_ms) / 10))
        else:
            share = total * weights[k] / tw
            cs = max(1, round(share / 10))
        tags.append(r"{\k" + str(int(cs)) + "}" + w.word)
    return " ".join(tags)


def _apply_flash_guard(beats: list[CaptionBeat]) -> list[CaptionBeat]:
    """Extend subliminal flashes (< FLASH_MIN_S) — words survive, flashes die."""
    out: list[CaptionBeat] = []
    for b in beats:
        dur_ms = b.end_ms - b.start_ms
        if 0 < dur_ms < int(FLASH_MIN_S * 1000):
            b = b.model_copy(update={"end_ms": b.start_ms + int(FLASH_MIN_S * 1000)})
        out.append(b)
    return out


def build_ass(
    beats: Sequence[CaptionBeat],
    preset: str = "doc_2row_karaoke",
    profanity_mode: str = "mask",
) -> str:
    """Build the .ass file content. Rows never contain overlapping cues."""
    if preset != "doc_2row_karaoke":
        raise SubtitleBuildError(f"unknown style preset: {preset}")
    if profanity_mode not in {"mask", "off"}:
        raise SubtitleBuildError(f"unknown profanity_mode: {profanity_mode}")

    subs = pysubs2.SSAFile()
    # committed v4/v6 PlayRes: 1280x720 — libass scales proportionally, so the
    # 40pt law holds at any render resolution (60px on a 1080p master)
    subs.info["PlayResX"] = str(PLAY_RES_X)
    subs.info["PlayResY"] = str(PLAY_RES_Y)
    subs.info["WrapStyle"] = "2"
    subs.styles.clear()

    flash_safe = _apply_flash_guard(list(beats))
    rows = _assign_rows(flash_safe)
    for beat, lane in zip(flash_safe, rows, strict=False):
        if profanity_mode == "mask":
            masked_words = []
            for w in beat.words:
                masked_words.append(w.model_copy(update={"word": _mask_profanity(w.word)}))
            beat = beat.model_copy(
                update={"text": _mask_profanity(beat.text), "words": masked_words}
            )

        color_hex = _speaker_color(beat.speaker).lstrip("#")
        color = pysubs2.Color(
            int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        )
        style_name = f"Lane{lane}_{beat.speaker}".replace(" ", "_")
        if style_name not in subs.styles:
            subs.styles[style_name] = pysubs2.SSAStyle(
                fontname=FONT_NAME,
                fontsize=FONT_SIZE,
                primarycolor=color,
                bold=True,
                alignment=Alignment.BOTTOM_CENTER,  # numpad 2
                # committed v4 look: heavy readable outline + soft shadow
                outline=OUTLINE,
                shadow=SHADOW,
                spacing=SPACING,
                marginv=LANE1_MARGIN_V if lane == 1 else LANE2_MARGIN_V,
                marginl=MARGIN_LR,
                marginr=MARGIN_LR,
            )
        event = pysubs2.SSAEvent(
            start=beat.start_ms,
            end=beat.end_ms,
            text=r"{\fad(40,30)}" + _karaoke_text(beat),
        )
        event.style = style_name
        subs.events.append(event)

    _assert_lane_non_overlap(subs)
    return subs.to_string("ass")


def _assert_lane_non_overlap(subs: pysubs2.SSAFile) -> None:
    """Guardrail 6 invariant: no temporal overlap between cues sharing a lane."""
    by_lane: dict[int, list[pysubs2.SSAEvent]] = {}
    for ev in subs.events:
        lane = 1 if (subs.styles[ev.style].marginv == LANE1_MARGIN_V) else 2
        by_lane.setdefault(lane, []).append(ev)
    for lane, events in by_lane.items():
        events.sort(key=lambda e: e.start)
        for prev, cur in zip(events, events[1:], strict=False):
            if cur.start < prev.end:
                raise SubtitleBuildError(
                    f"lane {lane} overlap: {prev.start}-{prev.end} vs {cur.start}-{cur.end}"
                )
