"""Speech-gated audio leveling — dialogue sits in -12..-6 dBFS, sirens don't rule.

The problem with plain `loudnorm`: it integrates EVERYTHING — sirens, door
slams, cruiser A/C — so a loud siren 10 s away drags dialogue gain down and
officers end up whispered. This module measures RMS/peaks STRICTLY on ASR
spoken-word intervals (officer / suspect / narrator), applies one gain so
speech lands in the target window, and bricks the ceiling at -1.5 dBTP with
an `alimiter` look-ahead so the added gain never clips.

Pure ffmpeg + numpy: one PCM decode, exact per-interval math, no heavy SDK.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# speech target window (issue #15): dialogue RMS must sit inside it
SPEECH_LO_DBFS = -12.0
SPEECH_HI_DBFS = -6.0
DEFAULT_TARGET_DBFS = -9.0  # center of the window
MAX_GAIN_DB = 12.0  # never boost dead-air hiss into audibility
# -1.5 dBTP ceiling as a linear alimiter threshold: 10^(-1.5/20) = 0.8414
LIMITER_DBTP = -1.5
LIMITER_LINEAR = 0.8414


class LevelingError(RuntimeError):
    pass


def decode_pcm_mono(path: str | Path, sr: int = 48000) -> np.ndarray:
    """One ffmpeg PCM decode -> float32 mono at `sr` (exact sample math)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise LevelingError(f"pcm decode failed: {r.stderr[-300:].decode(errors='replace')}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS in dBFS of a sample slice (empty/silent -> -inf clamped to -120)."""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return max(-120.0, 20.0 * np.log10(rms + 1e-12))


def peak_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    return max(-120.0, 20.0 * np.log10(float(np.max(np.abs(samples))) + 1e-12))


def merge_intervals(
    intervals: list[tuple[float, float]], pad_s: float = 0.15
) -> list[tuple[float, float]]:
    """Merged speech intervals with a small pad so word edges are not cut."""
    padded = [(max(0.0, s - pad_s), e + pad_s) for s, e in intervals if e > s]
    if not padded:
        return []
    padded.sort()
    out = [padded[0]]
    for s, e in padded[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def measure_speech_levels(
    path: str | Path,
    speech_intervals: list[tuple[float, float]],
    sr: int = 48000,
) -> dict:
    """RMS/peak measured ONLY on spoken-word spans (+ whole-file for contrast).

    `speech_rms_dbfs` is the leveling input; `file_rms_dbfs` exists so QA can
    prove the siren was NOT what got measured.
    """
    pcm = decode_pcm_mono(path, sr)
    speech = merge_intervals(speech_intervals)
    slices: list[np.ndarray] = []
    for s, e in speech:
        a, b = int(s * sr), min(int(e * sr), pcm.size)
        if b > a:
            slices.append(pcm[a:b])
    if slices:
        joined = np.concatenate(slices)
    else:
        joined = np.empty(0, dtype=np.float32)
    return {
        "speech_rms_dbfs": rms_dbfs(joined),
        "speech_peak_dbfs": peak_dbfs(joined),
        "file_rms_dbfs": rms_dbfs(pcm),
        "speech_seconds": joined.size / sr,
    }


def speech_gated_gain(
    measured_dbfs: float,
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    max_gain_db: float = MAX_GAIN_DB,
) -> float:
    """One gain moves speech to target; clamped both ways (no hiss inflation)."""
    gain = target_dbfs - measured_dbfs
    return round(max(-max_gain_db, min(max_gain_db, gain)), 2)


def speech_gated_chain(gain_db: float, limit_linear: float = LIMITER_LINEAR) -> str:
    """FFmpeg fragment: the one gain, then the -1.5 dBTP look-ahead brickwall.

    `alimiter` runs AFTER the gain so the boosted dialogue can never poke
    through the ceiling — the limiter is the guarantee, not an afterthought.
    """
    return (
        f"volume={gain_db:+g}dB,"
        f"alimiter=level_in=1:level_out=1:limit={limit_linear:g}:attack=5:release=50"
    )


def leveled_chain_for(
    path: str | Path,
    speech_intervals: list[tuple[float, float]],
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    sr: int = 48000,
) -> tuple[str, dict]:
    """Measure -> gain -> chain, one call (the worker seam for audio leveling)."""
    levels = measure_speech_levels(path, speech_intervals, sr)
    gain = speech_gated_gain(levels["speech_rms_dbfs"], target_dbfs)
    levels["gain_db"] = gain
    return speech_gated_chain(gain), levels


def check_speech_window(
    measured_dbfs: float,
    lo_dbfs: float = SPEECH_LO_DBFS,
    hi_dbfs: float = SPEECH_HI_DBFS,
) -> bool:
    return lo_dbfs <= measured_dbfs <= hi_dbfs
