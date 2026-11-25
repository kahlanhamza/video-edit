"""Source->master clock mapping for QA and post-fx audio measurement.

ONE shared implementation — the same law the v6 captions ride
(subtitles/master_cues.py: raw speech remaps through each kept raw clip's
source->master window; narration blocks carry no raw speech). The acoustic
gates were found measuring SOURCE-clock spans against the SHIPPED reordered
master (tape-1 v2, 2026-11-25): the spans landed on narration b-roll beds and
wrong moments, so a green enhance stage "failed" with a false speech RMS
-22 dBFS and a false clean_snr 9.0 dB. The shipped file is still measured
itself (M5-R1 law) — only the spans change clocks.
"""

from __future__ import annotations

from typing import Any


def master_clock_speech_intervals(
    timeline: Any,
    source_intervals: list[tuple[float, float]],
    *,
    include_narration_blocks: bool = False,
) -> list[tuple[float, float]]:
    """Source-clock speech spans -> MASTER clock, through the kept raw clips.

    Each kept raw clip maps [src0, src0+dur) -> its master-clock window;
    narration blocks and gaps carry no raw dialogue, so their spans drop out.
    Speech cut with the material it lived in drops out too — it is not on the
    master.

    `include_narration_blocks=True` additionally treats each narration block's
    full master window as program speech: the SNR noise floor must never
    sample the TTS voice as "noise" (it is program speech; the b-roll bed
    under it is muted by the narrator-on-broll isolation law).
    """
    from director_os.otio_engine.contract import (
        NarratorClipMeta,
        RawClipMeta,
        parse_clip_meta,
    )
    from director_os.otio_engine.timeline import (
        MASTER_TRACK,
        is_gap,
        item_duration_s,
        track_by_name,
    )

    master = track_by_name(timeline, MASTER_TRACK)
    if master is None:
        return sorted(source_intervals)
    out: list[tuple[float, float]] = []
    t_master = 0.0
    for item in master:
        dur = item_duration_s(item)
        if dur <= 0:
            continue
        if is_gap(item):
            t_master += dur
            continue
        try:
            meta = parse_clip_meta(item)
        except Exception:  # noqa: BLE001 — unknown item kinds still advance time
            meta = None
        if isinstance(meta, RawClipMeta):
            sr = getattr(item, "source_range", None)
            src0 = sr.start_time.to_seconds() if sr is not None else 0.0
            for s, e in source_intervals:
                lo, hi = max(s, src0), min(e, src0 + dur)
                if hi > lo:
                    out.append((t_master + (lo - src0), t_master + (hi - src0)))
        elif isinstance(meta, NarratorClipMeta) and include_narration_blocks:
            out.append((t_master, t_master + dur))
        t_master += dur
    return sorted(out)
