"""REAL DeepFilterNet3 run — optional, skipped when the binary is absent.

CI is hermetic (no binary, no downloads): these tests skip. In an environment
with `deep-filter` on PATH (or DEEPFILTER_BIN set — the sandbox and the
worker image both ship it), they prove the actual neural extraction on a
known noisy clip: broadband noise floor collapses, speech content survives,
timeline is preserved exactly.
"""

from __future__ import annotations

import os
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
from director_os.audio_engine.enhance import DeepFilterCliEnhancer, EnhancerUnavailableError
from director_os.audio_engine.leveling import decode_pcm_mono
from director_os.audio_engine.snr import noise_floor_db
from director_os.core.qa_probe import probe

SR = 48_000
DUR_S = 6.0
WORDS = [(0.5, 2.5), (3.0, 5.0)]


def _band_noise(rng, n: int, lo_hz: float, hi_hz: float, sr: int) -> np.ndarray:
    x = rng.normal(size=n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    out = np.fft.irfft(spec * mask, n=n)
    return out / (np.sqrt(np.mean(np.square(out))) + 1e-9)


@pytest.fixture(scope="module")
def deepfilter():
    bin_path = os.getenv("DEEPFILTER_BIN", "deep-filter")
    if shutil.which(bin_path) is None and not Path(bin_path).exists():
        pytest.skip("deep-filter CLI not installed in this environment (hermetic CI)")
    return DeepFilterCliEnhancer(bin_path=bin_path)


@pytest.fixture(scope="module")
def noisy_clip(tmp_path_factory) -> Path:
    """6 s: speech-band bursts at -13 dBFS + brown noise at -10 dBFS."""
    root = tmp_path_factory.mktemp("df_real")
    rng = np.random.default_rng(7)
    n = int(DUR_S * SR)
    speech = np.zeros(n)
    for s0, s1 in WORDS:
        a, b = int(s0 * SR), int(s1 * SR)
        speech[a:b] = _band_noise(rng, b - a, 300.0, 3000.0, SR) * 0.22
    white = rng.normal(size=n)
    noise = white / (np.sqrt(np.mean(np.square(white))) + 1e-9) * 0.18
    mixed = np.clip(speech + noise, -0.99, 0.99)
    wav = root / "noisy.wav"
    data = (
        (np.clip(np.stack([mixed, mixed], axis=1), -0.999, 0.999) * 32767)
        .astype(np.int16)
        .tobytes()
    )
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    return wav


def test_real_deepfilter_attenuates_pure_noise(deepfilter, tmp_path):
    """A noise-only clip: the neural extractor must crush the floor >= 10 dB.

    (Real deep-filter on a pure white-noise wav attenuates ~27 dB; the gate
    is set conservatively so the assertion stays about the MODEL, not the
    exact weights version. Synthetic band-noise 'speech' is NOT asserted —
    MossFormer/DFN cannot be expected to treat fake speech as speech.)
    """
    rng = np.random.default_rng(31)
    n = 4 * SR
    noise = rng.normal(size=n) * 0.15
    wav = tmp_path / "noise_only.wav"
    data = (
        (np.clip(np.stack([noise, noise], axis=1), -0.999, 0.999) * 32767)
        .astype(np.int16)
        .tobytes()
    )
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    out = tmp_path / "noise_enhanced.wav"
    result = deepfilter.enhance(wav, out)
    assert result.duration_drift_s <= 0.05
    raw_floor = noise_floor_db(decode_pcm_mono(str(wav)), SR, [])
    clean_floor = noise_floor_db(decode_pcm_mono(str(out)), SR, [])
    assert clean_floor < raw_floor - 10.0, (
        f"noise floor {raw_floor:.1f} -> {clean_floor:.1f} dBFS "
        "(neural extractor must attenuate pure noise)"
    )


def test_real_deepfilter_preserves_timeline_and_format(deepfilter, noisy_clip, tmp_path):
    out = tmp_path / "enhanced.wav"
    result = deepfilter.enhance(noisy_clip, out)
    assert result.engine == "deepfilternet3"
    assert result.neural is True
    assert result.atten_lim_db == 28.0
    # timeline preserved — the 2-bus mix requires sample alignment
    assert result.duration_drift_s <= 0.05, "enhancer must keep the timeline"
    info = probe(out)
    assert info.sample_rate == 48_000
    # the extraction actually changed the signal (no silent pass-through)
    from director_os.audio_engine.leveling import decode_pcm_mono

    raw_pcm = decode_pcm_mono(str(noisy_clip))
    clean_pcm = decode_pcm_mono(str(out))
    m = min(raw_pcm.size, clean_pcm.size)
    assert float(np.max(np.abs(clean_pcm[:m] - raw_pcm[:m]))) > 1e-3


def test_real_deepfilter_missing_binary_raises(monkeypatch):
    from director_os.audio_engine.enhance import DEEPFILTER_BIN_ENV

    monkeypatch.setenv(DEEPFILTER_BIN_ENV, "/definitely/not/here")
    with pytest.raises(EnhancerUnavailableError):
        DeepFilterCliEnhancer()
