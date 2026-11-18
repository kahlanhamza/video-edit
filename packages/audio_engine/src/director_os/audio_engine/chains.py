"""FFmpeg audio chain builders (PLAN §4.5 "Audio Mastering & Utilities").

Single source of truth for the loudness/48 kHz law on segment masters; higher-level
mix/duck chains arrive with the M2 audio worker.
"""

from __future__ import annotations

from director_os.core.ffmpeg_args import (
    audio_output_args,
    exact_slice_args,
    loudnorm_chain,
)

__all__ = [
    "audio_output_args",
    "exact_slice_args",
    "loudnorm_chain",
    "duck_chain",
    "tts_pause_trim_chain",
    "narration_bed_chain",
]


def duck_chain(duck_db: float = -18.0) -> str:
    """Sidechain ducking of a music/bed track under narration (audio.mix_ducking)."""
    return f"volume={duck_db:g}dB"


def tts_pause_trim_chain(
    start_threshold_db: float = -50.0,
    start_silence_s: float = 0.30,
    end_silence_s: float = 0.40,
    *,
    compress_pauses: bool = True,
    pause_min_s: float = 0.30,
    pause_keep_s: float = 0.15,
    pause_threshold_db: float = -45.0,
) -> str:
    """Trim TTS dead padding on both ends + COMPRESS internal pauses.

    Pattern: silence at head -> reverse -> silence at tail -> reverse
    -> internal pause compression (owner rule 2026-11-19: the narration must
    catch attention — pauses longer than `pause_min_s` are shortened to
    `pause_keep_s`, so a sleepy 1.2s mid-sentence gap becomes a 150ms beat).
    """
    chain = (
        f"silenceremove=start_periods=1:start_threshold={start_threshold_db:g}dB"
        f":start_silence={start_silence_s:g},"
        "areverse,"
        f"silenceremove=start_periods=1:start_threshold={start_threshold_db:g}dB"
        f":start_silence={end_silence_s:g},"
        "areverse"
    )
    if compress_pauses:
        chain += (
            f",silenceremove=stop_periods=-1:stop_duration={pause_min_s:g}"
            f":stop_threshold={pause_threshold_db:g}dB:stop_silence={pause_keep_s:g}"
        )
    return chain


def narration_bed_chain(mute_head_s: float | None = None) -> str:
    """Narration bed: optional hard mute for a spill head, then bed level -6 dB."""
    chain = ""
    if mute_head_s and mute_head_s > 0:
        chain += f"volume=enable='lt(t,{mute_head_s:.2f})':volume=0,"
    return chain + "volume=-6dB"
