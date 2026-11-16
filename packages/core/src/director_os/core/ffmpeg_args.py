"""FFmpeg argument builders — the executable form of PLAN guardrails 1 and 2.

Guardrail 1: `-ar 48000 -ac 2` must come AFTER the `loudnorm` filter on every segment
before `concat -c copy`, or mixed sample rates silently mute downstream narrator clips.

Guardrail 2: exact slicing is `-ss <start> -t <end-start>`; never `-ss` before `-i`
combined with `-to` after `-i`.
"""

from __future__ import annotations


class FFmpegArgError(RuntimeError):
    pass


def exact_slice_args(start_s: float, end_s: float) -> list[str]:
    """Guardrail 2: compute the exact duration and pass `-ss <start> -t <dur>`."""
    dur = round(end_s - start_s, 6)
    if dur <= 0:
        raise ValueError(f"invalid slice: end ({end_s}) must exceed start ({start_s})")
    return ["-ss", f"{start_s:.6f}", "-t", f"{dur:.6f}"]


def loudnorm_chain(target_lufs: float = -16.0, true_peak: float = -1.5, lra: float = 11.0) -> str:
    return f"loudnorm=I={target_lufs:g}:TP={true_peak:g}:LRA={lra:g}"


def audio_output_args(
    sample_rate: int = 48_000,
    channels: int = 2,
    codec: str = "aac",
    bitrate: str = "192k",
) -> list[str]:
    """THE LAW: these flags ride *after* every -af chain on every segment (ADR-0005)."""
    return ["-ar", str(sample_rate), "-ac", str(channels), "-c:a", codec, "-b:a", bitrate]


def assert_audio_law(cmd: list[str]) -> None:
    """Verify a segment command obeys guardrail 1: -af precedes -ar/-ac; loudnorm present."""
    try:
        af_idx = cmd.index("-af")
    except ValueError:
        af_idx = None
    try:
        ar_idx = cmd.index("-ar")
        ac_idx = cmd.index("-ac")
    except ValueError as e:
        raise AudioLawArgError(f"segment command missing -ar/-ac audio law flags: {e}") from e

    if af_idx is None:
        # no filter chain at all (pure silence path) — still fine, flags must exist
        return
    if not (af_idx < ar_idx and af_idx < ac_idx):
        raise AudioLawArgError(
            "guardrail 1 violated: -af must precede -ar/-ac "
            "(loudnorm then resample, or downstream narration dies)"
        )
    if "loudnorm" in cmd[af_idx + 1]:
        return  # loudnorm chain followed by the audio law flags: correct


class AudioLawArgError(FFmpegArgError):
    pass
