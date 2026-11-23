"""Caption op write-through — the store IS the baked truth (caption ops law).

A caption op that never reaches the cue store would display text/timing the
render path never bakes — a lie between the op log and the program. The law
(mirrors the blur dual-write): a caption op is only valid WITH a store write;
the fold's `caption_overrides` stay a display lens over a store that already
contains the edit. Both the API (`POST /ops` kind=caption) and the agent tool
(`ops.propose` kind=caption) go through `apply_caption_op` here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from director_os.subtitles.ass_builder import CaptionBeat
from director_os.subtitles.store import SubtitleStoreError, edit_cue, load_cues


class CaptionOpError(ValueError):
    """A caption op whose write-through failed — the op must not be created."""


def apply_caption_op(sub_dir: str | Path, params: Mapping[str, object]) -> CaptionBeat:
    """Write one caption op's params through to the cue store.

    Requires `cue_index`; accepts any of `text` / `speaker` / `start_ms` +
    `end_ms` (the timing pair — set together or both omitted, end > start).
    Returns the updated beat; raises `CaptionOpError` on any invalid input or
    store failure (the caller must then NOT create the op row).
    """
    idx_raw = params.get("cue_index")
    if idx_raw is None:
        raise CaptionOpError("caption op requires params.cue_index (which cue is edited?)")
    if isinstance(idx_raw, bool) or not isinstance(idx_raw, (int, float, str)):
        raise CaptionOpError(f"caption op params.cue_index must be an int, got {idx_raw!r}")
    try:
        idx = int(idx_raw)
    except (TypeError, ValueError) as e:
        raise CaptionOpError(f"caption op params.cue_index must be an int, got {idx_raw!r}") from e

    text = params.get("text")
    speaker = params.get("speaker")
    start_ms = params.get("start_ms")
    end_ms = params.get("end_ms")
    if text is None and speaker is None and start_ms is None and end_ms is None:
        raise CaptionOpError(
            "caption op carries no edit: params need text, speaker, or the "
            "start_ms/end_ms timing pair"
        )
    if text is not None and not str(text).strip():
        raise CaptionOpError("caption op text cannot be empty")

    def _ms(v: object) -> int | None:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise CaptionOpError(f"caption op timing must be integer ms, got {v!r}")
        try:
            return int(v)
        except (TypeError, ValueError) as e:
            raise CaptionOpError(f"caption op timing must be integer ms, got {v!r}") from e

    try:
        cues = load_cues(sub_dir)
        if not cues:
            raise CaptionOpError("no cues yet — run the pipeline before caption ops")
        return edit_cue(
            sub_dir,
            idx,
            str(text) if text is not None else None,
            speaker=str(speaker) if speaker is not None else None,
            start_ms=_ms(start_ms),
            end_ms=_ms(end_ms),
        )
    except SubtitleStoreError as e:
        raise CaptionOpError(str(e)) from e
