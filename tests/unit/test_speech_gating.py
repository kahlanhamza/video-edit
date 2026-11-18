"""Speech-gated leveling — siren must not rule the meter (issue #15)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from director_os.audio_engine.leveling import (
    DEFAULT_TARGET_DBFS,
    LIMITER_LINEAR,
    MAX_GAIN_DB,
    ceiling_limit_pcm,
    check_speech_window,
    decode_pcm_channels,
    leveled_chain_for,
    measure_speech_levels,
    merge_intervals,
    peak_dbfs,
    rms_dbfs,
    speech_gated_chain,
    speech_gated_gain,
)
from director_os.core.qa_probe import audit_technical

SR = 48000


def _write_wav(path: Path, samples: np.ndarray) -> Path:
    import wave

    s16 = np.clip(samples, -1.0, 1.0)
    s16 = (s16 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(s16.tobytes())
    return path


def _tone(seconds: float, amp: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture()
def bodycam_mix(tmp_path: Path) -> tuple[Path, list[tuple[float, float]]]:
    """Quiet dialogue (-18 dBFS) at 1-3 s + LOUD siren (-3 dBFS) at 4-6 s.

    Plain loudnorm would average both; speech-gated leveling must measure
    ONLY the dialogue span and leave the siren out of the gain decision.
    """
    # sine amplitude -15 dB -> RMS = amp - 3.01 dB = -18 dBFS
    speech = _tone(2.0, 10 ** (-15 / 20))
    siren = _tone(2.0, 10 ** (-3 / 20), freq=800.0)
    pcm = np.concatenate(
        [
            np.zeros(int(1.0 * SR), dtype=np.float32),
            speech,
            np.zeros(int(1.0 * SR), dtype=np.float32),  # gap: pad must not reach siren
            siren,
        ]
    )
    wav = _write_wav(tmp_path / "mix.wav", pcm)
    return wav, [(1.0, 3.0)]


def test_measure_only_speech_span(bodycam_mix: tuple[Path, list[tuple[float, float]]]):
    wav, intervals = bodycam_mix
    levels = measure_speech_levels(wav, intervals)
    assert abs(levels["speech_rms_dbfs"] - (-18.0)) < 1.5, levels
    # file RMS includes the siren — proof the meter was gated, not global
    assert levels["file_rms_dbfs"] > levels["speech_rms_dbfs"] + 6.0


def test_gain_moves_speech_into_window(bodycam_mix: tuple[Path, list[tuple[float, float]]]):
    wav, intervals = bodycam_mix
    levels = measure_speech_levels(wav, intervals)
    gain = speech_gated_gain(levels["speech_rms_dbfs"])
    assert -MAX_GAIN_DB <= gain <= MAX_GAIN_DB
    # -18 dB + ~9 dB -> -9 dB: inside the -12..-6 window
    assert check_speech_window(levels["speech_rms_dbfs"] + gain)


def test_gain_clamped_against_hiss_inflation():
    assert speech_gated_gain(-60.0) == MAX_GAIN_DB  # 81 dB boost -> clamped
    assert speech_gated_gain(24.0) == -MAX_GAIN_DB  # 33 dB cut -> clamped
    assert speech_gated_gain(DEFAULT_TARGET_DBFS) == 0.0  # already at target
    assert speech_gated_gain(0.0) == DEFAULT_TARGET_DBFS  # modest cut stays unclamped


def test_chain_has_gain_then_ceiling_contract():
    """The ceiling is the numpy pass (`ceiling_limit_pcm`), not an in-graph filter.

    ffmpeg 7.1's alimiter mangles the program (corr 0.063 vs its own input) —
    the chain is gain-only; `run_two_bus_mix` bricks the ceiling after the mix.
    """
    chain = speech_gated_chain(9.0)
    assert chain == "volume=+9dB"
    assert "alimiter" not in chain  # the broken filter must not ride any graph


def test_leveled_chain_for_one_call(bodycam_mix: tuple[Path, list[tuple[float, float]]]):
    wav, intervals = bodycam_mix
    chain, levels = leveled_chain_for(wav, intervals)
    assert chain.startswith("volume=")
    assert "alimiter" not in chain
    assert levels["gain_db"] == pytest.approx(levels["gain_db"], abs=0.1)
    assert abs(levels["gain_db"] - (DEFAULT_TARGET_DBFS - levels["speech_rms_dbfs"])) < 0.3


def test_limiter_prevents_clipping(tmp_path: Path):
    """Hot dialogue + full gain -> numpy ceiling holds the -1.5 dBTP law."""
    hot = _tone(1.0, 0.95)  # near-full-scale dialogue
    wav = _write_wav(tmp_path / "hot.wav", hot)
    # no intervals: measurement returns file RMS; gain clamps to -12 dB
    chain, levels = leveled_chain_for(wav, [(0.0, 1.0)], target_dbfs=-9.0)
    # the chain is gain-only; the ceiling rides ceiling_limit_pcm (the same
    # post-pass run_two_bus_mix applies)
    pcm, ch = decode_pcm_channels(wav, SR)
    boosted = _apply_volume_and_ceiling(pcm, ch, chain, LIMITER_LINEAR)
    assert peak_dbfs(boosted) <= -1.4, "ceiling failed: output pokes through -1.5 dBTP"


def _apply_volume_and_ceiling(pcm, ch, chain, ceiling):
    import re

    m = re.match(r"volume=([+-][\d.]+)dB", chain)
    gain = 10 ** (float(m.group(1)) / 20.0)
    return ceiling_limit_pcm(pcm * gain, SR, ceiling, channels=ch)


def test_audit_speech_window_check(bodycam_mix: tuple[Path, list[tuple[float, float]]]):
    """audit_technical gains speech_gated_loudness when intervals are given."""
    wav, intervals = bodycam_mix
    # measured speech -18 dBFS is OUTSIDE -12..-6 -> audit fails loud
    report = audit_technical(wav, speech_intervals=intervals)
    check = next(c for c in report.checks if c.name == "speech_gated_loudness")
    assert not check.ok
    assert "speech RMS" in check.detail
    # after leveling by the measured gain, speech lands in-window -> pass
    levels = measure_speech_levels(wav, intervals)
    gain = speech_gated_gain(levels["speech_rms_dbfs"])
    out = wav.parent / "leveled.wav"
    r = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav),
            "-af",
            speech_gated_chain(gain),
            "-ar",
            str(SR),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report2 = audit_technical(out, speech_intervals=intervals)
    check2 = next(c for c in report2.checks if c.name == "speech_gated_loudness")
    assert check2.ok, check2.detail


def test_no_intervals_no_check(bodycam_mix: tuple[Path, list[tuple[float, float]]]):
    wav, _ = bodycam_mix
    report = audit_technical(wav)
    assert all(c.name != "speech_gated_loudness" for c in report.checks)


def test_merge_intervals_pads_and_unions():
    out = merge_intervals([(1.0, 2.0), (2.1, 3.0), (10.0, 11.0)])
    assert out == [(0.85, 3.15), (9.85, 11.15)]
    assert merge_intervals([]) == []


def test_rms_empty_is_silent():
    assert rms_dbfs(np.empty(0, dtype=np.float32)) == -120.0
    assert peak_dbfs(np.empty(0, dtype=np.float32)) == -120.0
