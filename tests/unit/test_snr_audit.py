"""Quantitative SNR verification + the production audio QA gates.

Covers: robust noise-floor math (median beats a passing siren), the SNR gate
thresholds, the timbre (deep-voice) tilt measurement + gate, and the
audit_technical checks that enforce the 2-bus law — enhancer_engine_used
(afftdn forbidden in production), snr_verification, timbre_not_muffled,
zero_clipping, stream_uniformity.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from director_os.audio_engine.snr import (
    measure_snr,
    noise_floor_db,
    peak_true_dbfs,
    snr_gate,
    speech_band_tilts,
    timbre_gate,
)
from director_os.core.qa_probe import AudioVerification, audit_technical

SR = 48_000


def _write_wav(path: Path, pcm: np.ndarray) -> Path:
    if pcm.ndim == 1:  # mono input -> duplicated stereo (a raw 1-D buffer
        pcm = np.stack([pcm, pcm], axis=1)  # declared as 2ch interleaves and
        # aliases the spectrum on decode — fatal for spectral-tilt asserts)
    data = (np.clip(pcm, -0.999, 0.999) * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    return path


def _band_noise(rng, n: int, lo_hz: float, hi_hz: float, sr: int) -> np.ndarray:
    x = rng.normal(size=n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    out = np.fft.irfft(spec * mask, n=n)
    return out / (np.sqrt(np.mean(np.square(out))) + 1e-9)


def _make_pcm(rng, speech_spans, dur_s=6.0, speech_amp=0.3, noise_amp=0.05, siren=None):
    """Speech = FFT band-limited noise bursts (300-7000 Hz, realistic RMS).

    The band reaches 7 kHz because REAL speech carries fricative/sibilant
    energy there — a 3 kHz-deep synthetic voice would be artificially muffled
    and trip the timbre gate for the wrong reason.
    """
    n = int(dur_s * SR)
    speech = np.zeros(n)
    for s0, s1 in speech_spans:
        a, b = int(s0 * SR), int(s1 * SR)
        speech[a:b] = _band_noise(rng, b - a, 300.0, 7000.0, SR) * speech_amp
    brown = np.cumsum(rng.normal(size=n))
    brown = brown / (np.sqrt(np.mean(np.square(brown))) + 1e-9)
    noise = brown * noise_amp
    if siren:
        for s0, s1, amp in siren:
            a, b = int(s0 * SR), int(s1 * SR)
            tt = np.arange(b - a) / SR
            noise[a:b] += amp * np.sin(2 * np.pi * 900 * tt)
    return speech, noise


WORDS = [(1.0, 2.5), (3.5, 5.0)]


class TestNoiseFloor:
    def test_median_floor_rejects_siren_burst(self):
        rng = np.random.default_rng(3)
        _, noise = _make_pcm(rng, WORDS, siren=[(5.2, 5.9, 0.25)])  # loud span in the gap
        pcm = noise.astype(np.float32)
        floor = noise_floor_db(pcm, SR, WORDS)
        floor_siren_excluded = noise_floor_db(pcm, SR, WORDS + [(5.2, 5.9)])
        assert floor < -20.0  # the siren did not drag the floor up
        # median stays within 5 dB of the uncontaminated floor (23% of gap frames)
        assert abs(floor - floor_siren_excluded) < 5.0

    def test_empty_pcm_clamps_to_minus120(self):
        import numpy as np

        assert noise_floor_db(np.empty(0, dtype=np.float32), SR, WORDS) == -120.0


class TestMeasureSnr:
    def test_raw_vs_clean_snr_fields(self, tmp_path):
        rng = np.random.default_rng(5)
        speech, noise = _make_pcm(rng, WORDS)
        raw = _write_wav(tmp_path / "raw.wav", (speech + noise).astype(np.float32))
        clean = _write_wav(tmp_path / "clean.wav", speech.astype(np.float32))
        r = measure_snr(str(raw), WORDS)
        c = measure_snr(str(clean), WORDS)
        assert c["snr_db"] - r["snr_db"] > 10.0  # removing known noise improves SNR
        assert c["noise_floor_db"] < r["noise_floor_db"] - 8.0


class TestSnrGate:
    def test_passing_scenario(self):
        ok, _ = snr_gate(raw_snr_db=6.0, clean_snr_db=22.0, delta_noise_db=-16.0)
        assert ok

    def test_fails_low_clean_snr(self):
        ok, detail = snr_gate(raw_snr_db=2.0, clean_snr_db=9.0, delta_noise_db=-20.0)
        assert not ok and "clean_snr" in detail

    def test_fails_when_neither_reduction_path_met(self):
        ok, detail = snr_gate(raw_snr_db=10.0, clean_snr_db=15.0, delta_noise_db=-2.0)
        assert not ok and "noise reduction insufficient" in detail

    def test_or_condition_snrs_improvement_only(self):
        ok, _ = snr_gate(raw_snr_db=4.0, clean_snr_db=14.5, delta_noise_db=-5.0)
        assert ok  # improvement +10.5 >= 6.0 despite delta_noise > -8


class TestNaturalDynamicsGate:
    def test_natural_crest_passes(self):
        from director_os.audio_engine.snr import natural_dynamics_gate

        ok, detail = natural_dynamics_gate(9.5)
        assert ok and "9.5" in detail

    def test_crushed_voice_fails(self):
        from director_os.audio_engine.snr import natural_dynamics_gate

        ok, detail = natural_dynamics_gate(3.5)
        assert not ok and "CRUSHED" in detail

    def test_speech_crest_of_a_known_signal(self, tmp_path):
        # constant-amplitude sine: crest of a full-scale sine = 3.01 dB
        from director_os.audio_engine.snr import speech_crest_db

        n = SR * 2
        t = np.arange(n) / SR
        pcm = 0.5 * np.sin(2 * np.pi * 440 * t)
        path = tmp_path / "sine.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((pcm * 32767).astype(np.int16).tobytes())
        crest = speech_crest_db(str(path), [(0.1, 1.9)])
        assert crest == pytest.approx(3.01, abs=0.05)


class TestTimbreTilt:
    def test_full_band_voice_measures_neutral_tilt(self, tmp_path):
        rng = np.random.default_rng(31)
        speech, noise = _make_pcm(rng, WORDS)  # 300-7000 Hz "voice"
        wav = _write_wav(tmp_path / "voice.wav", (speech + noise).astype(np.float32))
        tilts = speech_band_tilts(str(wav), WORDS)
        # flat density across mid/presence bands -> tilt near zero (±3 dB),
        # i.e. NOT muffled; air band is above the voice band so it dips some
        assert tilts["tilt_presence_db"] > -6.0
        assert tilts["tilt_air_db"] > -12.0

    def test_muffled_voice_fails_gate_and_audit(self, tmp_path):
        # the deep-voice signature measured on the real harvest: everything
        # above ~2 kHz sits ~40 dB down (shelf, not a hard cut) — extreme
        # muffling, well past the gate floor
        rng = np.random.default_rng(33)
        speech, _ = _make_pcm(rng, WORDS)
        freqs = np.fft.rfftfreq(speech.size, d=1.0 / SR)
        x = np.fft.irfft(np.fft.rfft(speech) * np.where(freqs < 2000.0, 1.0, 0.01), n=speech.size)
        wav = _write_wav(tmp_path / "muffled.wav", x.astype(np.float32))
        tilts = speech_band_tilts(str(wav), WORDS)
        ok, detail = timbre_gate(tilts)
        assert not ok and "VOICE MUFFLED" in detail

        master = tmp_path / "muffled.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=30:duration=6",
                "-i",
                str(wav),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "34",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                str(master),
            ],
            check=True,
            capture_output=True,
        )
        ver = AudioVerification(raw_audio_path=str(wav), enhancer_engine="deepfilternet3")
        audit = audit_technical(master, speech_intervals=WORDS, audio_verification=ver)
        check = next(c for c in audit.checks if c.name == "timbre_not_muffled")
        assert not check.ok and not audit.ok

    def test_no_speech_intervals_raise(self, tmp_path):
        rng = np.random.default_rng(35)
        speech, noise = _make_pcm(rng, WORDS)
        wav = _write_wav(tmp_path / "voice.wav", (speech + noise).astype(np.float32))
        with pytest.raises(ValueError):
            speech_band_tilts(str(wav), [])


class TestAuditAudioVerification:
    @pytest.fixture()
    def scene(self, tmp_path):
        rng = np.random.default_rng(9)
        # speech_amp kept so band-noise peaks (~4 sigma) stay under full scale —
        # a clean master must not clip before the limiter even exists
        speech, noise = _make_pcm(rng, WORDS, speech_amp=0.18, siren=[(2.8, 3.4, 0.2)])
        raw = _write_wav(tmp_path / "raw_48k.wav", (speech + noise).astype(np.float32))
        clean = _write_wav(tmp_path / "clean.wav", (speech * 0.9).astype(np.float32))
        # a playable master video carrying the clean audio (48 kHz stereo)
        master = tmp_path / "final.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=30:duration=6",
                "-i",
                str(clean),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "34",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                str(master),
            ],
            check=True,
            capture_output=True,
        )
        return raw, clean, master

    def test_production_legal_engine_passes_all_checks(self, scene):
        raw, _, master = scene
        ver = AudioVerification(
            raw_audio_path=str(raw),
            enhancer_engine="deepfilternet3",
            enhancer_neural=True,
            atten_lim_db=28.0,
        )
        audit = audit_technical(master, speech_intervals=WORDS, audio_verification=ver)
        by_name = {c.name: c for c in audit.checks}
        assert by_name["enhancer_engine_used"].ok
        assert by_name["snr_verification"].ok, by_name["snr_verification"].detail
        assert by_name["timbre_not_muffled"].ok, by_name["timbre_not_muffled"].detail
        assert by_name["zero_clipping"].ok
        assert by_name["stream_uniformity"].ok
        assert "raw_snr" in by_name["snr_verification"].detail

    def test_afftdn_in_production_fails_engine_check(self, scene):
        raw, _, master = scene
        ver = AudioVerification(
            raw_audio_path=str(raw),
            enhancer_engine="afftdn",
            enhancer_neural=False,
            allow_afftdn=False,
        )
        audit = audit_technical(master, speech_intervals=WORDS, audio_verification=ver)
        check = next(c for c in audit.checks if c.name == "enhancer_engine_used")
        assert not check.ok and "FORBIDDEN" in check.detail
        assert not audit.ok

    def test_dirty_master_fails_snr_gate(self, scene):
        raw, clean, _tmp = scene
        # "cleaned" master that is just the raw noisy audio again (fake cleanup)
        still_dirty = scene[2].parent / "dirty.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=30:duration=6",
                "-i",
                str(raw),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "34",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                str(still_dirty),
            ],
            check=True,
            capture_output=True,
        )
        ver = AudioVerification(raw_audio_path=str(raw), enhancer_engine="deepfilternet3")
        audit = audit_technical(still_dirty, speech_intervals=WORDS, audio_verification=ver)
        check = next(c for c in audit.checks if c.name == "snr_verification")
        assert not check.ok
        assert check.detail.startswith("raw_snr")  # metrics always visible

    def test_clipped_master_fails_zero_clipping(self, scene, tmp_path):
        raw, _, _ = scene
        rng = np.random.default_rng(13)
        hot_pcm = rng.normal(size=6 * SR) * 0.9  # white noise peaking past full scale
        hot = _write_wav(tmp_path / "hot.wav", hot_pcm.astype(np.float32))
        clipped = tmp_path / "clipped.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=30:duration=6",
                "-i",
                str(hot),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "34",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                str(clipped),
            ],
            check=True,
            capture_output=True,
        )
        ver = AudioVerification(raw_audio_path=str(raw), enhancer_engine="deepfilternet3")
        audit = audit_technical(clipped, speech_intervals=WORDS, audio_verification=ver)
        check = next(c for c in audit.checks if c.name == "zero_clipping")
        assert not check.ok

    def test_peak_helper_reports_hot_file(self, tmp_path):
        rng = np.random.default_rng(21)
        pcm = rng.normal(size=SR) * 0.9
        hot = _write_wav(tmp_path / "hot.wav", pcm.astype(np.float32))
        assert peak_true_dbfs(str(hot)) > -1.0
