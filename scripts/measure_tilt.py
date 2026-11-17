"""Measure spectral band balance raw vs MossFormer2-enhanced (deep-voice diagnosis).

Reads the real artifacts from the Kaggle SE48K harvest run and computes per-band
RMS (dBFS) over the full file, plus band deltas. Confirms (or refutes) that the
neural enhancer darkened the voice by shaving HF.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

BANDS = [
    ("sub-low 60-120", 60, 120),
    ("boom 120-250", 120, 250),
    ("low-mid 250-500", 250, 500),
    ("mid 500-2k", 500, 2000),
    ("presence 2-5k", 2000, 5000),
    ("edge 5-8k", 5000, 8000),
    ("air 8-12k", 8000, 12000),
    ("top 12-20k", 12000, 20000),
]

SR = 48_000


def decode(path: str) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vn",
        "-ar",
        str(SR),
        "-ac",
        "1",
        "-f",
        "f32le",
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def band_rms_db(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x.astype(np.float64)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any() or spec[mask].size == 0:
        return -120.0
    p = float(np.mean(spec[mask] ** 2))
    return max(-120.0, 10.0 * np.log10(p + 1e-20))


def main() -> None:
    raw_p, enh_p = sys.argv[1], sys.argv[2]
    raw, enh = decode(raw_p), decode(enh_p)
    n = min(raw.size, enh.size)
    raw, enh = raw[:n], enh[:n]
    print(f"samples={n} ({n / SR:.1f}s @ {SR} Hz)")
    print(f"{'band':<18}{'raw dB':>10}{'enh dB':>10}{'delta':>9}")
    for name, lo, hi in BANDS:
        r, e = band_rms_db(raw, lo, hi), band_rms_db(enh, lo, hi)
        print(f"{name:<18}{r:>10.1f}{e:>10.1f}{e - r:>+9.1f}")
    # tilt: low vs presence, presence vs air — the "deep/muffled" signature
    for x, tag in ((raw, "raw"), (enh, "enh")):
        low = band_rms_db(x, 120, 250)
        pres = band_rms_db(x, 2000, 5000)
        air = band_rms_db(x, 5000, 12000)
        mid = band_rms_db(x, 500, 2000)
        print(
            f"{tag}: tilt(low-mid)={low - mid:+.1f} dB  tilt(pres-mid)={pres - mid:+.1f} dB  "
            f"tilt(air-mid)={air - mid:+.1f} dB"
        )


if __name__ == "__main__":
    main()
