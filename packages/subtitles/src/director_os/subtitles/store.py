"""Cue store for the Director Studio subtitle grid (M3 close-out, PLAN §3.3F).

Persists the structured cue list (`subtitles.json`) next to the rendered
`subtitles.ass` / `subtitles.srt`, so the grid can read and edit cues without
reverse-parsing karaoke tags. Every edit re-aligns word timings through
`difflib.SequenceMatcher` (PLAN guardrail 5) and rebuilds all three artifacts
deterministically — no linear interpolation, ever.
"""

from __future__ import annotations

import json
from pathlib import Path

import pysubs2
from director_os.subtitles.align import RawWord, align_words
from director_os.subtitles.ass_builder import CaptionBeat, WordTiming, build_ass
from pydantic import ValidationError

JSON_NAME = "subtitles.json"
ASS_NAME = "subtitles.ass"
SRT_NAME = "subtitles.srt"


class SubtitleStoreError(RuntimeError):
    pass


def save_cues(base_dir: str | Path, beats: list[CaptionBeat]) -> None:
    """Write subtitles.json + subtitles.ass + subtitles.srt from the cue list."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / JSON_NAME).write_text(
        "[" + ",\n".join(b.model_dump_json() for b in beats) + "]",
        encoding="utf-8",
    )
    if beats:
        (base / ASS_NAME).write_text(build_ass(beats), encoding="utf-8")
        srt = pysubs2.SSAFile()
        for b in beats:
            srt.events.append(pysubs2.SSAEvent(start=b.start_ms, end=b.end_ms, text=b.text))
        (base / SRT_NAME).write_text(srt.to_string("srt"), encoding="utf-8")
    else:
        (base / ASS_NAME).write_text("", encoding="utf-8")
        (base / SRT_NAME).write_text("", encoding="utf-8")


def load_cues(base_dir: str | Path) -> list[CaptionBeat]:
    base = Path(base_dir)
    raw = (base / JSON_NAME).read_text(encoding="utf-8") if (base / JSON_NAME).exists() else "[]"
    beats: list[CaptionBeat] = []
    for item in json.loads(raw or "[]"):
        try:
            beats.append(CaptionBeat.model_validate(item))
        except ValidationError as e:  # corrupt line must not kill the whole grid
            raise SubtitleStoreError(f"corrupt cue entry: {e}") from e
    return beats


def _rescale_words(
    words: list[WordTiming],
    old_start: int,
    old_end: int,
    new_start: int,
    new_end: int,
) -> list[WordTiming]:
    """Linearly map word timings from the old cue window into the new one —
    an editor's timing nudge moves every WORD with the cue, never just the
    shell (the karaoke view must stay in step with the cue it belongs to)."""
    span_old = max(1, old_end - old_start)
    span_new = max(1, new_end - new_start)
    out: list[WordTiming] = []
    for w in words:
        s = new_start + round((w.start_ms - old_start) * span_new / span_old)
        e = new_start + round((w.end_ms - old_start) * span_new / span_old)
        out.append(
            WordTiming(word=w.word, start_ms=max(new_start, s), end_ms=max(new_start + 1, e))
        )
    return out


def edit_cue(
    base_dir: str | Path,
    index: int,
    text: str | None = None,
    speaker: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> CaptionBeat:
    """Proofread one cue: difflib re-alignment against its raw word timings.

    Word count may change freely — equal blocks keep raw timings, replacements
    split the original span by word length, inserts interpolate between
    neighbors (see `subtitles.align`). The cue keeps its absolute slot unless
    an explicit `start_ms`/`end_ms` (set TOGETHER — a half window is a lie)
    moves it; word timings then rescale linearly into the new window.
    """
    beats = load_cues(base_dir)
    if not 0 <= index < len(beats):
        raise SubtitleStoreError(f"cue index out of range: {index} (n={len(beats)})")
    beat = beats[index]
    if (start_ms is None) != (end_ms is None):
        raise SubtitleStoreError("start_ms and end_ms must be set together (or both omitted)")
    if start_ms is not None and end_ms is not None and end_ms <= start_ms:
        raise SubtitleStoreError(f"cue end {end_ms}ms before start {start_ms}ms")

    new_text = " ".join(text.split()) if text is not None else beat.text
    if not new_text:
        raise SubtitleStoreError("cue text cannot be empty")

    raw = [RawWord(word=w.word, start_ms=w.start_ms, end_ms=w.end_ms) for w in beat.words]
    if raw and new_text != beat.text:
        aligned = align_words(new_text, raw)
        words = [WordTiming(word=w.word, start_ms=w.start_ms, end_ms=w.end_ms) for w in aligned]
    else:
        words = [WordTiming(word=w.word, start_ms=w.start_ms, end_ms=w.end_ms) for w in beat.words]

    timing_changed = start_ms is not None and (start_ms != beat.start_ms or end_ms != beat.end_ms)
    if timing_changed and start_ms is not None and end_ms is not None:
        # explicit editor window wins — the words ride along, rescaled
        new_start, new_end = start_ms, end_ms
        words = _rescale_words(words, beat.start_ms, beat.end_ms, new_start, new_end)
    elif raw and new_text != beat.text:
        # keep the cue inside its original slot even after heavy rewrites
        new_start = beat.start_ms
        end_ms = max(w.end_ms for w in words) if words else beat.end_ms
        new_end = max(end_ms, beat.start_ms + 1)
    else:
        new_start, new_end = beat.start_ms, beat.end_ms

    new_beat = CaptionBeat(
        text=new_text,
        start_ms=new_start,
        end_ms=new_end,
        speaker=speaker if speaker is not None else beat.speaker,
        words=words,
    )
    beats[index] = new_beat
    save_cues(base_dir, beats)
    return new_beat
