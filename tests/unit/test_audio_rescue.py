"""Muted-speech rescue (ENHANCE_CACHE_VERSION v4) — the enhancer may never
destroy dialogue.

DeepFilterNet3's post-mask can zero quiet far-field speech (measured 13/239
words on the Eustis 0-180s window). `_rescue_muted_speech` detects word spans
whose enhanced RMS collapsed >RESCUE_DROP_DB below the precleaned raw and
splices the preclean audio back with cosine ramps. These tests pin the
contract on synthetic PCM: destroy -> rescue, kept speech -> untouched.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
from director_os.audio_engine.two_bus import SAMPLE_RATE
from director_os.worker.audio_enhance import _rescue_muted_speech

SR = SAMPLE_RATE


def _read(path: Path) -> np.ndarray:
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ar",
            str(SR),
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    """Interleaved stereo f32 -> s16 WAV (the 2-bus law)."""
    s16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(s16.tobytes())


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.repeat(mono[:, None], 2, axis=1).astype(np.float32)


def _span_rms(pcm: np.ndarray, s: float, e: float) -> float:
    seg = pcm[int(s * SR) * 2 : int(e * SR) * 2]
    return float(20 * np.log10(np.sqrt(np.mean(seg**2)) + 1e-12))


def _make_pair(tmp_path: Path, mute: bool) -> tuple[Path, Path, dict]:
    """3s pair: a speech burst at 0.5-1.0s, noise floor elsewhere.

    `mute=True` simulates the DFN3 post-mask zeroing the burst in enhanced.
    """
    t = np.arange(int(3.0 * SR)) / SR
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(t.size) * 10 ** (-38 / 20)).astype(np.float32)
    burst = (np.sin(2 * np.pi * 220.0 * t) * 10 ** (-20 / 20)).astype(np.float32)
    in_burst = (t >= 0.5) & (t < 1.0)
    preclean_mono = noise + np.where(in_burst, burst, 0.0)
    enhanced_mono = np.where(in_burst, 0.0 if mute else burst, noise)
    preclean = tmp_path / "preclean.wav"
    enhanced = tmp_path / "enhanced.wav"
    _write_wav(preclean, _stereo(preclean_mono))
    _write_wav(enhanced, _stereo(enhanced_mono))
    transcript = {"words": [{"text": "sir", "start_s": 0.5, "end_s": 1.0, "speaker": "Officer"}]}
    return enhanced, preclean, transcript


def test_destroyed_speech_is_rescued(tmp_path: Path):
    enhanced, preclean, transcript = _make_pair(tmp_path, mute=True)
    before = _read(enhanced)
    assert _span_rms(before, 0.55, 0.95) < -90.0  # the post-mask zeroed the burst

    rescued = _rescue_muted_speech(enhanced, preclean, transcript, lambda m: None, "")

    assert rescued == 1
    after = _read(enhanced)
    # the burst is back at preclean level
    assert abs(_span_rms(after, 0.55, 0.95) - _span_rms(_read(preclean), 0.55, 0.95)) < 1.0
    # the noise floor outside the span is untouched (enhancer output kept)
    assert abs(_span_rms(after, 1.5, 2.5) - _span_rms(before, 1.5, 2.5)) < 1e-4


def test_kept_speech_is_untouched(tmp_path: Path):
    enhanced, preclean, transcript = _make_pair(tmp_path, mute=False)
    pcm_before = _read(enhanced)
    rescued = _rescue_muted_speech(enhanced, preclean, transcript, lambda m: None, "")
    assert rescued == 0  # a healthy enhancer keeps its speech: no splice
    assert np.array_equal(pcm_before, _read(enhanced))


def test_no_words_no_rescue(tmp_path: Path):
    enhanced, preclean, _ = _make_pair(tmp_path, mute=True)
    rescued = _rescue_muted_speech(enhanced, preclean, {"words": []}, lambda m: None, "")
    assert rescued == 0  # stub-ASR runs carry no word truth
