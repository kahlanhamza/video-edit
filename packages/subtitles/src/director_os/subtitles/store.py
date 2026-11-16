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


def edit_cue(
    base_dir: str | Path,
    index: int,
    text: str,
    speaker: str | None = None,
) -> CaptionBeat:
    """Proofread one cue: difflib re-alignment against its raw word timings.

    Word count may change freely — equal blocks keep raw timings, replacements
    split the original span by word length, inserts interpolate between
    neighbors (see `subtitles.align`). The cue keeps its absolute slot.
    """
    beats = load_cues(base_dir)
    if not 0 <= index < len(beats):
        raise SubtitleStoreError(f"cue index out of range: {index} (n={len(beats)})")
    beat = beats[index]
    new_text = " ".join(text.split())
    if not new_text:
        raise SubtitleStoreError("cue text cannot be empty")

    raw = [RawWord(word=w.word, start_ms=w.start_ms, end_ms=w.end_ms) for w in beat.words]
    if raw:
        aligned = align_words(new_text, raw)
        words = [WordTiming(word=w.word, start_ms=w.start_ms, end_ms=w.end_ms) for w in aligned]
    else:
        words = []
    # keep the cue inside its original slot even after heavy rewrites
    if words:
        end_ms = max(w.end_ms for w in words) if new_text != beat.text else beat.end_ms
        end_ms = max(end_ms, beat.start_ms + 1)
    else:
        end_ms = beat.end_ms

    new_beat = CaptionBeat(
        text=new_text,
        start_ms=beat.start_ms,
        end_ms=end_ms,
        speaker=speaker if speaker is not None else beat.speaker,
        words=words,
    )
    beats[index] = new_beat
    save_cues(base_dir, beats)
    return new_beat
