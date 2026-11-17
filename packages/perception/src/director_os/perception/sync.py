"""Multi-camera audio sync — cross-correlation offsets + clock drift (PLAN §4.4, M4).

FOIA packets routinely mix devices whose clocks disagree by seconds and whose
sample clocks drift by tens of ppm. `sync_multicam` aligns every angle to one
reference track by cross-correlating their 16 kHz mono audio with FFT, refining
the peak to sub-sample precision (parabolic interpolation -> ~1 ms at 16 kHz),
and estimating residual clock drift by re-correlating a second window deep into
the recordings (lag slope over wall time).

Sign convention: `offset_s > 0` means the angle started recording LATER than
the reference — its clock reads `ref_time - offset_s`. `drift_ppm > 0` means
the angle's clock runs SLOW (it falls further behind over time). Consumers
align an angle to reference wall time `t` by seeking the angle at
`t - offset_s - drift_ppm * t / 1e6`.

Pure numpy DSP — no model downloads, no services. Deterministic on synthetic
inputs and pinned by tests to ±1 ms offset accuracy.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SYNC_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_S = 30.0
DEFAULT_SEARCH_S = 120.0
DEFAULT_PROBE_S = 6.0
N_DRIFT_PROBES = 5
MIN_CONFIDENCE_FOR_DRIFT = 0.3
DRIFT_PROBE_GAP_S = 60.0  # how deep past the first window the drift probe sits


class SyncError(RuntimeError):
    pass


@dataclass(frozen=True)
class SyncResult:
    asset_id: str
    offset_s: float
    drift_ppm: float
    confidence: float

    def angle_source_time(self, ref_time_s: float) -> float:
        """Where to seek this angle so it plays the reference's `ref_time_s`."""
        return ref_time_s - self.offset_s - self.drift_ppm * ref_time_s / 1e6


@dataclass(frozen=True)
class MultiCamSync:
    reference_asset_id: str
    sample_rate: int
    results: list[SyncResult] = field(default_factory=list)

    def for_asset(self, asset_id: str) -> SyncResult | None:
        return next((r for r in self.results if r.asset_id == asset_id), None)


# ---------------------------------------------------------------- audio IO


def extract_audio(
    video_path: str | Path, out_wav: str | Path, sample_rate: int = SYNC_SAMPLE_RATE
) -> Path:
    """Mono 16-bit WAV at the sync sample rate via ffmpeg (video ignored)."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise SyncError(f"audio extraction failed for {video_path}: {r.stderr[-500:]}")
    return out_wav


def load_samples(wav_path: str | Path) -> np.ndarray:
    """Read a 16-bit PCM mono WAV into float32 in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as w:
        if w.getnchannels() != 1:
            raise SyncError(f"{wav_path}: expected mono, got {w.getnchannels()} channels")
        if w.getsampwidth() != 2:
            raise SyncError(f"{wav_path}: expected 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------- correlation


def _next_pow2(n: int) -> int:
    return 1 << max(1, (n - 1).bit_length())


def cross_correlate(
    ref_window: np.ndarray, other_window: np.ndarray, fs: int, search_s: float
) -> tuple[float, float]:
    """Best alignment of `ref_window` inside `other_window` within ±search_s.

    Returns (lag_s, confidence): `lag_s` is the shift k in ref[i] ≈ other[i + k]
    (positive = matching content sits later in the other track). Confidence is
    normalized over the OVERLAPPING region only — 1.0 when the overlap is
    identical, regardless of how much of either window falls outside it.
    """
    if len(ref_window) < 64 or len(other_window) < 64:
        raise SyncError("correlation windows too short (<64 samples)")
    n = _next_pow2(len(ref_window) + len(other_window))
    fa = np.fft.rfft(ref_window, n)
    fo = np.fft.rfft(other_window, n)
    corr = np.fft.irfft(np.conj(fa) * fo, n)

    band = int(round(search_s * fs))
    band = min(band, n // 2 - 1)
    idx_pos = np.arange(0, band + 1)  # lags 0..+band
    idx_neg = np.arange(n - band, n)  # lags -band..-1 (wrapped)
    candidates = np.concatenate([idx_pos, idx_neg])
    peak_i = int(candidates[np.argmax(np.abs(corr[candidates]))])
    k = peak_i if peak_i <= band else peak_i - n  # wrapped -> negative lag

    # parabolic sub-sample refinement on the signed peak (real corr, not |corr|)
    c_m, c_0, c_p = corr[peak_i - 1], corr[peak_i], corr[(peak_i + 1) % n]
    denom = c_m - 2.0 * c_0 + c_p
    delta = float(np.clip(0.5 * (c_m - c_p) / denom, -1.0, 1.0)) if denom != 0 else 0.0

    lag_s = (k + delta) / fs
    conf = _overlap_confidence(float(corr[peak_i]), k, ref_window, other_window)
    return lag_s, conf


def _overlap_confidence(
    corr_peak: float, k: int, ref_window: np.ndarray, other_window: np.ndarray
) -> float:
    """NCC of the aligned overlap: corr_peak / (||ref_overlap|| * ||other_overlap||)."""
    i_lo = max(0, -k)
    i_hi = min(len(ref_window), len(other_window) - k)
    if i_hi - i_lo < 64:
        return 0.0
    ro = ref_window[i_lo:i_hi]
    oo = other_window[i_lo + k : i_hi + k]
    denom = float(np.linalg.norm(ro) * np.linalg.norm(oo))
    if denom <= 0:
        return 0.0
    return float(np.clip(corr_peak / denom, 0.0, 1.0))


def _window(samples: np.ndarray, start_s: float, dur_s: float, fs: int) -> np.ndarray:
    a = max(0, int(round(start_s * fs)))
    b = min(len(samples), a + int(round(dur_s * fs)))
    return samples[a:b]


def sync_pair(
    ref_samples: np.ndarray,
    other_samples: np.ndarray,
    fs: int = SYNC_SAMPLE_RATE,
    window_s: float = DEFAULT_WINDOW_S,
    search_s: float = DEFAULT_SEARCH_S,
    probe_s: float = DEFAULT_PROBE_S,
) -> tuple[float, float, float]:
    """Sync one angle against the reference: (offset_s, drift_ppm, confidence).

    Offset comes from correlating the reference's opening window inside the
    angle's head (±search_s). Clock drift is then estimated from several SHORT
    local probes spread across the overlap — a sample-clock mismatch shows up
    as lag slope over wall time, fitted by least squares. Short probes keep
    each correlation coherent even at 50+ ppm drift (a 30 s white-noise window
    would smear beyond its correlation length at that rate).
    """
    dur_ref = len(ref_samples) / fs
    dur_other = len(other_samples) / fs
    overlap = min(dur_ref, dur_other)
    win = min(window_s, overlap / 4.0)
    if win < 2.0:
        raise SyncError(f"overlap too short to sync: {overlap:.1f}s")

    lag_head, conf = cross_correlate(
        _window(ref_samples, 0.0, win, fs),
        _window(other_samples, 0.0, win + search_s, fs),
        fs,
        search_s,
    )
    # alignment: ref[i] ≈ other[i + lag]; other's wall start = -lag relative to ref
    offset_s = -lag_head

    drift_ppm = 0.0
    if conf >= MIN_CONFIDENCE_FOR_DRIFT:
        drift_ppm = _estimate_drift(ref_samples, other_samples, offset_s, overlap, fs, probe_s)
    return offset_s, drift_ppm, conf


def _estimate_drift(
    ref_samples: np.ndarray,
    other_samples: np.ndarray,
    offset_s: float,
    overlap: float,
    fs: int,
    probe_s: float,
) -> float:
    """Least-squares slope of lag over wall time from N short local probes.

    lag(t) = -offset(t) = -(offset_0 + drift*t), so drift_ppm = -slope * 1e6.
    Each probe correlates ref[t : t+p] against a ±2 s slack neighborhood around
    the expected alignment in the other track — a tiny FFT per probe.
    """
    slack = 2.0
    first = DRIFT_PROBE_GAP_S
    last = overlap - probe_s - slack
    if first + probe_s > last:
        return 0.0  # overlap too short for a meaningful slope
    times = np.linspace(first, last, N_DRIFT_PROBES)
    lags: list[float] = []
    kept: list[float] = []
    for t_p in times:
        r = _window(ref_samples, t_p, probe_s, fs)
        if len(r) < fs // 2:
            continue
        # other tau = wall - offset(t): probe window centers the expected peak
        head = max(0.0, t_p - offset_s - slack)
        o = _window(other_samples, head, probe_s + 2 * slack, fs)
        if len(o) < fs // 2:
            continue
        lag_p, conf_p = cross_correlate(r, o, fs, slack)
        if conf_p < MIN_CONFIDENCE_FOR_DRIFT:
            continue
        # r[i] ≈ o[i + lag_p]: wall(t_p) ↔ other(head + lag_p) -> lag_abs = -offset(t_p)
        lag_abs = lag_p + head - t_p
        lags.append(lag_abs)
        kept.append(t_p)
    if len(lags) < 2:
        return 0.0
    slope = float(np.polyfit(np.array(kept), np.array(lags), 1)[0])
    return -slope * 1e6


def sync_multicam(
    audio_paths: dict[str, str | Path],
    reference_asset_id: str,
    sample_rate: int = SYNC_SAMPLE_RATE,
    window_s: float = DEFAULT_WINDOW_S,
    search_s: float = DEFAULT_SEARCH_S,
    workdir: str | Path | None = None,
) -> MultiCamSync:
    """Align every angle's audio to the reference track.

    `audio_paths` maps asset ids to video OR wav files (videos are extracted
    with ffmpeg first). Returns per-angle `SyncResult`s including the
    reference itself at offset 0 / drift 0.
    """
    if reference_asset_id not in audio_paths:
        raise SyncError(f"reference '{reference_asset_id}' not among inputs")
    if len(audio_paths) < 2:
        raise SyncError("multi-cam sync needs at least two angles")

    ctx: tempfile.TemporaryDirectory[str] | None = None
    if workdir:
        work = Path(workdir)
    else:
        ctx = tempfile.TemporaryDirectory()
        work = Path(ctx.name)
    work.mkdir(parents=True, exist_ok=True)
    try:
        samples: dict[str, np.ndarray] = {}
        for asset_id, path in audio_paths.items():
            p = Path(path)
            if p.suffix.lower() == ".wav":
                wav = p
            else:
                wav = extract_audio(p, work / f"{asset_id}.wav", sample_rate)
            samples[asset_id] = load_samples(wav)

        ref = samples[reference_asset_id]
        results = [SyncResult(reference_asset_id, 0.0, 0.0, 1.0)]
        for asset_id, other in samples.items():
            if asset_id == reference_asset_id:
                continue
            offset_s, drift_ppm, conf = sync_pair(
                ref, other, sample_rate, window_s=window_s, search_s=search_s
            )
            results.append(
                SyncResult(asset_id, round(offset_s, 4), round(drift_ppm, 1), round(conf, 3))
            )
        return MultiCamSync(reference_asset_id, sample_rate, results)
    finally:
        if ctx is not None:
            ctx.cleanup()
