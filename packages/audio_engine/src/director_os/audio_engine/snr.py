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

Timbre gate (the "deep voice" law): neural speech extractors shave the top
end hardest — measured on the real MossFormer2 harvest, presence (2-5 kHz)
lands -19.9 dB and air (5-12 kHz) -23.9 dB below the 500-2k mid band, which
perceptually reads as muffled/deep voices. `speech_band_tilts` + `timbre_gate`
quantify the shipped master's spectral balance on word intervals and FAIL
extreme muffling (presence tilt below MIN_PRESENCE_TILT_DB).
"""

from __future__ import annotations

import numpy as np
from director_os.audio_engine.leveling import decode_pcm_mono, merge_intervals, rms_dbfs

FRAME_S = 0.10
MIN_CLEAN_SNR_DB = 14.0
MAX_DELTA_NOISE_DB = -8.0  # noise floor must FALL by at least 8 dB ...
MIN_SNR_IMPROVEMENT_DB = 6.0  # ... or SNR must improve by at least 6 dB
MAX_PEAK_DBTP = -1.0  # sample-peak ceiling on the final master

# spectral-tilt bands (measured deep-voice signature — see module docstring)
TILT_LOW_HZ = (120.0, 250.0)
TILT_MID_HZ = (500.0, 2000.0)
TILT_PRESENCE_HZ = (2000.0, 5000.0)
TILT_AIR_HZ = (5000.0, 12000.0)
MIN_PRESENCE_TILT_DB = -24.0  # extreme-muffling floor (pre-fix real render: ~-20)


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


def speech_band_tilts(
    path: str,
    speech_intervals: list[tuple[float, float]],
    sr: int = 48_000,
) -> dict[str, float]:
    """Spectral balance of the VOICE on the merged word spans (the timbre QA).

    Concatenates the samples inside spoken-word spans, FFTs them, and reports
    each band's mean energy density relative to the 500-2k mid band:
    `tilt_lowmid_db` (120-250), `tilt_presence_db` (2-5k), `tilt_air_db`
    (5-12k). Per-Hz normalization makes the numbers level- and duration-
    independent. Raises when there is no speech to measure (caller decides).
    """
    pcm = decode_pcm_mono(path, sr)
    slices = []
    for s, e in merge_intervals(speech_intervals):
        a, b = int(s * sr), min(int(e * sr), pcm.size)
        if b > a:
            slices.append(pcm[a:b])
    if not slices:
        raise ValueError("no speech samples inside the given intervals")
    x = np.concatenate(slices).astype(np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)

    def band_db(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        n = int(mask.sum())
        if n == 0:
            return -120.0
        p = float(np.mean(spec[mask]))
        return max(-120.0, 10.0 * np.log10(p + 1e-24))

    mid = band_db(*TILT_MID_HZ)
    return {
        "tilt_lowmid_db": band_db(*TILT_LOW_HZ) - mid,
        "tilt_presence_db": band_db(*TILT_PRESENCE_HZ) - mid,
        "tilt_air_db": band_db(*TILT_AIR_HZ) - mid,
    }


def timbre_gate(
    tilts: dict[str, float],
    min_presence_tilt_db: float = MIN_PRESENCE_TILT_DB,
) -> tuple[bool, str]:
    """The timbre law: the shipped voice must not be extremely muffled.

    FAILS when the presence band (2-5 kHz) falls more than `floor` dB below
    the mid band inside word spans — the measured "deep/muffled voice"
    signature. Low-mid and air tilts ride in the detail for evidence; they
    are recorded, not gated (voice f0 and mic proximity legitimately vary
    them between officers and vehicles).
    """
    tilt = tilts["tilt_presence_db"]
    ok = tilt >= min_presence_tilt_db
    detail = (
        f"presence {tilt:+.1f}dB air {tilts['tilt_air_db']:+.1f}dB "
        f"lowmid {tilts['tilt_lowmid_db']:+.1f}dB vs mid (gate >= {min_presence_tilt_db:g}dB)"
    )
    if not ok:
        detail += " | VOICE MUFFLED: presence band collapsed — check the Bus-1 timbre chain"
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
