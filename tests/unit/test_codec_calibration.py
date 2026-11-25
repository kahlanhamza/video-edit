"""The codec reconstruction law — AAC overshoot compensation (2026-11-25).

The limiter's flat ceiling plateaus ring on AAC decode (192k measured up to
+4.5 dB, one window +7). The design previously assumed +1.1 dB and shipped
masters decoded at +2.27 dBFS against the -1.0 dBTP gate. `calibrate_mix_for_codec`
measures the ACTUAL encoder on the ACTUAL mix and trims until the decoded
peak fits CODEC_PROBE_TARGET_DBFS — the shipped file rides ONE codec
generation, so the probe predicts the shipped decode exactly.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from director_os.audio_engine.snr import peak_true_dbfs
from director_os.audio_engine.two_bus import CODEC_PROBE_TARGET_DBFS
from director_os.worker.audio_enhance import calibrate_mix_for_codec

SR = 48_000


def _write_s16_wav(path: Path, pcm: np.ndarray) -> Path:
    data = (np.clip(pcm, -0.999, 0.999) * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    return path


def _flat_ceiling_pcm(dur_s: float = 1.0) -> np.ndarray:
    """A limiter-flattened waveform: MDCT-heavy bursts ride the ceiling."""
    rng = np.random.default_rng(11)
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    voice = 0.98 * np.sign(np.sin(2 * np.pi * 2000.0 * t))  # MDCT-heavy square-ish
    ceiled = np.maximum(np.minimum(voice, 0.9), -0.9).astype(np.float32)  # flat plateau
    interleave = np.repeat(ceiled[:, None], 2, axis=1).reshape(-1)
    return interleave + rng.normal(0, 1e-4, size=interleave.size).astype(np.float32)


def test_calibrate_hot_mix_trims_decoded_peak_under_target(tmp_path):
    master = _write_s16_wav(tmp_path / "master_seg.wav", _flat_ceiling_pcm())
    pre = peak_true_dbfs(str(master))

    receipt = calibrate_mix_for_codec(master, log=lambda m: None)

    assert pre > CODEC_PROBE_TARGET_DBFS  # the hot mix starts over the target
    assert receipt["codec_probe_peak_dbfs"] <= CODEC_PROBE_TARGET_DBFS + 1e-6
    assert receipt["codec_trim_db"] < 0.0
    post = peak_true_dbfs(str(master))
    assert post <= CODEC_PROBE_TARGET_DBFS + 0.05
    # untouched artifacts cleaned up
    assert not (tmp_path / "master_seg.codec_probe.mp4").exists()
    assert not (tmp_path / "master_seg.codec_trim.wav").exists()


def test_calibrate_safe_mix_is_a_noop(tmp_path):
    rng = np.random.default_rng(5)
    quiet = (rng.normal(0, 0.01, size=int(0.5 * SR) * 2)).astype(np.float32)  # -40 dBFS
    master = _write_s16_wav(tmp_path / "master_seg.wav", quiet)
    before = master.read_bytes()

    receipt = calibrate_mix_for_codec(master, log=lambda m: None)

    assert receipt["codec_probe_peak_dbfs"] <= CODEC_PROBE_TARGET_DBFS
    assert receipt["codec_trim_db"] == 0.0
    assert master.read_bytes() == before  # the safe mix is never rewritten
