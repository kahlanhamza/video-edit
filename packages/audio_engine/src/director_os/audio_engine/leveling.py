"""Speech-gated audio leveling — dialogue sits in -12..-6 dBFS, sirens don't rule.

The problem with plain `loudnorm`: it integrates EVERYTHING — sirens, door
slams, cruiser A/C — so a loud siren 10 s away drags dialogue gain down and
officers end up whispered. This module measures RMS/peaks STRICTLY on ASR
spoken-word intervals (officer / suspect / narrator), applies one gain so
speech lands in the target window, and bricks the ceiling with a deterministic
numpy look-ahead limiter (`ceiling_limit_pcm`).

CEILING LAW (2026-11-18): ffmpeg 7.1's `alimiter` MANGLES the program in this
environment — measured correlation 0.063 against its own input on the Eustis
chain (fades-only control: 0.999), on every argument variant. The ceiling
moved to `ceiling_limit_pcm`: pure numpy, sample-exact (|out| <= ceiling by
construction), release-smoothed so crest reduction rides instead of pumping.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# speech target window (issue #15): dialogue RMS must sit inside it
SPEECH_LO_DBFS = -12.0
SPEECH_HI_DBFS = -6.0
# Target the window's LOWER half (user direction, 2026-11-18): -9 sat "in your
# face" for conversational bodycam footage; -11 keeps margin above the -12
# floor and eases the limiter's crest work.
DEFAULT_TARGET_DBFS = -11.0
MAX_GAIN_DB = 32.0  # measured ceiling, Nov 2026 (Eustis MossFormer2 harvest):
# the extraction leaves speech RMS ~-25 dBFS with ~19 dB crest, and the
# -1.5 dBTP limiter eats the peaks the one-shot gain adds — at a 20 dB clamp
# speech landed -14.2 dBFS, BELOW the -12..-6 law window (speech_gated_loudness
# audit FAIL); at 28 dB the refinement loop converged to -12.55 (still 0.55
# short) because the limiter response is concave (~0.2 dB RMS per gain dB once
# it engages). 32 dB lets the loop land dialogue at ~-11.8, inside the window
# with crest ~10.5 (inside the natural-dynamics gate). The old "never boost
# dead-air hiss" rationale is superseded: Bus-2 ambience is OFF by default
# (2026-11-18 ear verdict) and the speech-gated -14 dB duck protects the
# pauses; the limiter stays the absolute ceiling. Gain is always measured, so
# loud sources are unaffected — only very quiet sources reach further.
# -1.5 dBTP ceiling as a linear alimiter threshold: 10^(-1.5/20) = 0.8414
LIMITER_DBTP = -1.5
LIMITER_LINEAR = 0.8414


class LevelingError(RuntimeError):
    pass


def ceiling_limit_pcm(
    pcm: np.ndarray,
    sr: int,
    ceiling_linear: float,
    channels: int = 2,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
) -> np.ndarray:
    """Deterministic look-ahead ceiling limiter (the alimiter replacement).

    - Sample-exact by construction: the gain at frame i is capped by the
      smallest requirement over [i, i+attack] (LOOKAHEAD), so the gain is
      already down when a peak arrives — |out| never exceeds the ceiling.
    - Joint L/R decision: the frame peak is max(|L|, |R|), so identical gain
      rides both channels and the stereo image never wobbles.
    - Release smoothing: the gain falls instantly to any requirement but
      recovers at most exponentially (`release_ms`), vectorized as a running
      maximum in log-gain domain — crest reduction rides instead of pumping.
      (The original running-minimum envelope was INVERTED — attack became
      rate-limited and release instant, so valley-structured speech rode
      through the ceiling; found live on tape-1 v2 and pinned by the valley
      regression test.)
    """
    if pcm.size == 0:
        return pcm
    frames = pcm.reshape(-1, channels)
    peak = np.max(np.abs(frames), axis=1)
    # NEVER above 1.0: content already under the ceiling must pass untouched.
    # (Without this clamp a silent frame's cap explodes to ceiling/1e-9 and
    # the exp(-h) gain AMPLIFIES silence into clipping — caught by the run
    # where the refine loop walked gain to -32 dB while peaks hit +10 dBFS.)
    cap = np.minimum(1.0, ceiling_linear / np.maximum(peak, 1e-9))
    la = max(1, int(attack_ms * sr / 1000))
    if la > 1 and cap.size > la:
        # required suppression for the NEXT attack window (look-ahead min)
        cap = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([cap, np.full(la - 1, cap[-1])]), la
        ).min(axis=1)
    # log-gain domain: h >= 0. ATTACK IS INSTANT and release is bounded:
    # h_env[i] = max_{j<=i}(h[j] - (i-j)*a) — a peak's requirement engages at
    # once (the LOOKAHEAD above delivered it `attack_ms` early) and h then
    # decays at most `a` per sample. The previous envelope was inverted
    # (minimum-accumulate: attack rate-limited, release instant), so every
    # inter-syllable valley RESET the gain and hot peaks rode through up to
    # +3.9 dB above the ceiling — found live on tape-1 v2, where the shipped
    # master decoded at +3.37 dBFS against a -1.5 dBTP ceiling. With the max
    # envelope, gain[i] <= cap_la[i] <= ceiling/|x[i]| holds for EVERY sample:
    # the sample-exact contract is finally provable, not aspirational.
    h = -np.log(np.maximum(cap, 1e-9))
    a = 1000.0 / (release_ms * sr)
    idx = np.arange(h.size, dtype=np.float64)
    h_env = np.maximum.accumulate(h + idx * a) - idx * a
    gain = np.exp(-h_env)
    return (frames * gain[:, None]).reshape(pcm.shape)


def decode_pcm_channels(path: str | Path, sr: int = 48000) -> tuple[np.ndarray, int]:
    """One ffmpeg PCM decode -> interleaved float32 at `sr`, NATIVE channels.

    No `-ac 1`: ffmpeg's automatic stereo->mono downmix uses a 1/sqrt(2) law,
    which BOOSTS identical-channel material by +3.01 dB and would poison every
    dBFS measurement built on top. Callers downmix explicitly:
    - `decode_pcm_mono` averages channels (amplitude-preserving), so RMS/gain
      math sees the true signal level;
    - `snr.peak_true_dbfs` takes the max over channels (true stream peak).
    """
    import json as _json

    r_info = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
    )
    channels = 1
    try:
        payload = _json.loads(r_info.stdout or "{}")
        channels = int(payload["streams"][0].get("channels") or 1)
    except (ValueError, KeyError, IndexError):
        channels = 1
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
        "-ar",
        str(sr),
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise LevelingError(f"pcm decode failed: {r.stderr[-300:].decode(errors='replace')}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy(), channels


def decode_pcm_mono(path: str | Path, sr: int = 48000) -> np.ndarray:
    """float32 MONO at `sr` — channels AVERAGED (no downmix gain inflation)."""
    pcm, ch = decode_pcm_channels(path, sr)
    if ch <= 1:
        return pcm
    return pcm.reshape(-1, ch).mean(axis=1).astype(np.float32)


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
    """FFmpeg fragment: the one gain. Gain-only — see `ceiling_limit_pcm`.

    ffmpeg 7.1's alimiter MANGLES the program in this environment (measured
    correlation 0.063 against its own input; fades-only control 0.999), so
    no in-graph limiter may ride any chain. The brickwall is the
    deterministic numpy ceiling (`ceiling_limit_pcm`) applied by
    `run_two_bus_mix` after the mix — sample-exact |out| <= ceiling.
    """
    return f"volume={gain_db:+g}dB"


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
