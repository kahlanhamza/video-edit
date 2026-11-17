"""Quantitative acoustic SNR verification — proves the noise actually dropped.

The audit law (audio.isolate_2bus): a render may not ship "cleaned" audio on
faith. We measure, on the SAME ASR word intervals:

- `speech_rms_db` — RMS strictly inside spoken-word spans (dialogue energy)
- `noise_floor_db` — robust floor of the NON-speech gaps: 100 ms frame RMS
  across gap samples, then the MEDIAN frame (median rejects a passing siren or
  door slam that would inflate an average)
- `snr_db = speech_rms_db - noise_floor_db`

Gates (all must hold on the clean master):
- `clean_snr_db >= 14.0 dB` — dialogue is louder than what remains between words
- noise dropped: `delta_noise_db <= -8.0 dB` (floor fell ≥ 8 dB) OR the SNR
  improved ≥ 6 dB (`clean_snr - raw_snr >= 6.0`)
- zero clipping: sample peak stays ≤ -1.0 dBTP on the final master (the
  -1.5 dBTP limiter gives 0.5 dB headroom above this gate)
- stream uniformity: every audio stream 48000 Hz stereo (checked in qa_probe)
"""

from __future__ import annotations

import numpy as np
from director_os.audio_engine.leveling import decode_pcm_mono, merge_intervals, rms_dbfs

FRAME_S = 0.10
MIN_CLEAN_SNR_DB = 14.0
MAX_DELTA_NOISE_DB = -8.0  # noise floor must FALL by at least 8 dB ...
MIN_SNR_IMPROVEMENT_DB = 6.0  # ... or SNR must improve by at least 6 dB
MAX_PEAK_DBTP = -1.0  # sample-peak ceiling on the final master


def noise_floor_db(
    pcm: np.ndarray,
    sr: int = 48_000,
    speech_intervals: list[tuple[float, float]] | None = None,
    frame_s: float = FRAME_S,
) -> float:
    """Robust noise floor: median 100 ms-frame RMS over NON-speech samples."""
    if pcm.size == 0:
        return -120.0
    speech = merge_intervals(speech_intervals or []) if speech_intervals else []
    frame_n = max(1, int(frame_s * sr))
    n_frames = pcm.size // frame_n
    if n_frames == 0:
        return rms_dbfs(pcm)
    frames = pcm[: n_frames * frame_n].reshape(n_frames, frame_n).astype(np.float64)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    # frame is non-speech iff it does not overlap any merged speech span
    frame_t0 = np.arange(n_frames) * frame_s
    frame_t1 = frame_t0 + frame_s
    loud = np.zeros(n_frames, dtype=bool)
    for s, e in speech:
        loud |= (frame_t1 > s) & (frame_t0 < e)
    quiet_rms = frame_rms[~loud]
    if quiet_rms.size == 0:
        return rms_dbfs(pcm)
    floor = float(np.median(quiet_rms))
    return max(-120.0, 20.0 * np.log10(floor + 1e-12))


def speech_rms_db(pcm: np.ndarray, sr: int, speech_intervals: list[tuple[float, float]]) -> float:
    """RMS strictly on merged spoken-word spans (sirens never measured)."""
    slices = []
    for s, e in merge_intervals(speech_intervals):
        a, b = int(s * sr), min(int(e * sr), pcm.size)
        if b > a:
            slices.append(pcm[a:b])
    if not slices:
        return -120.0
    return rms_dbfs(np.concatenate(slices))


def measure_snr(
    path: str,
    speech_intervals: list[tuple[float, float]],
    sr: int = 48_000,
) -> dict:
    """Decode one file -> {speech_rms_db, noise_floor_db, snr_db, duration_s}."""
    pcm = decode_pcm_mono(path, sr)
    speech = speech_rms_db(pcm, sr, speech_intervals)
    floor = noise_floor_db(pcm, sr, speech_intervals)
    return {
        "speech_rms_db": speech,
        "noise_floor_db": floor,
        "snr_db": speech - floor,
        "duration_s": pcm.size / sr if sr else 0.0,
        "samples": int(pcm.size),
    }


def snr_gate(
    raw_snr_db: float,
    clean_snr_db: float,
    delta_noise_db: float,
    min_clean_snr: float = MIN_CLEAN_SNR_DB,
    max_delta_noise: float = MAX_DELTA_NOISE_DB,
    min_improvement: float = MIN_SNR_IMPROVEMENT_DB,
) -> tuple[bool, str]:
    """The production gate: clean floor AND proven noise reduction."""
    improvement = clean_snr_db - raw_snr_db
    reasons: list[str] = []
    if clean_snr_db < min_clean_snr:
        reasons.append(f"clean_snr {clean_snr_db:.1f} dB < {min_clean_snr:g} dB")
    if not (delta_noise_db <= max_delta_noise or improvement >= min_improvement):
        reasons.append(
            f"noise reduction insufficient: delta_noise {delta_noise_db:+.1f} dB "
            f"(need <= {max_delta_noise:g}) and snr improvement {improvement:+.1f} dB "
            f"(need >= {min_improvement:g})"
        )
    ok = not reasons
    detail = (
        f"raw_snr={raw_snr_db:.1f} clean_snr={clean_snr_db:.1f} "
        f"delta_noise={delta_noise_db:+.1f} improvement={improvement:+.1f}"
        if ok
        else "; ".join(reasons)
    )
    return ok, detail


def peak_true_dbfs(path: str, sr: int = 48_000) -> float:
    """TRUE stream peak in dBFS: max over ALL channels of the final master.

    Decodes without downmix gain (ffmpeg's auto mono downmix is a 1/sqrt(2)
    law that reads +3 dB hot on identical-channel material — see
    `leveling.decode_pcm_channels`). Clipping gate: must be <= -1.0 dBTP.
    """
    from director_os.audio_engine.leveling import decode_pcm_channels

    pcm, ch = decode_pcm_channels(path, sr)
    if pcm.size == 0:
        return -120.0
    x = pcm.reshape(-1, ch) if ch > 1 else pcm.reshape(-1, 1)
    return max(-120.0, 20.0 * np.log10(float(np.max(np.abs(x))) + 1e-12))
