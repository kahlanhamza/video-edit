"""2-row speaker-colored `.ass` builder — PLAN guardrail 6 + §4.6.

Simultaneous speakers ride two lanes (Row 1 `MarginV=46`, Row 2 `MarginV=96`) so
both stay legible; we never clamp overlapping lines onto one row with forced
minimums. Karaoke active-word highlight uses `\\k` centisecond tags from aligned
word timings (3–5 word beats).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pysubs2
from pydantic import BaseModel, Field
from pysubs2 import Alignment

LANE1_MARGIN_V = 46
LANE2_MARGIN_V = 96
FONT_NAME = "Roboto"
FONT_SIZE = 20

PALETTE = [
    "#FFFFFF",
    "#3EE6C1",
    "#FFD166",
    "#7EB6FF",
    "#FF9E7D",
    "#C792EA",
]

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
    idx = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % len(PALETTE)
    return PALETTE[idx]


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
    subs.info["PlayResX"] = "1920"
    subs.info["PlayResY"] = "1080"
    subs.styles.clear()

    rows = _assign_rows(beats)
    for beat, lane in zip(beats, rows, strict=False):
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
                marginv=LANE1_MARGIN_V if lane == 1 else LANE2_MARGIN_V,
                marginl=60,
                marginr=60,
            )
        event = pysubs2.SSAEvent(
            start=beat.start_ms,
            end=beat.end_ms,
            text=_karaoke_text(beat),
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
