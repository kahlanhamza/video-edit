"""The 2-Bus Speech Isolation mix — envelope math, chains, and the real ffmpeg mix.

The fixture generates a KNOWN signal: speech-shaped bursts + broadband noise.
The `PerfectEnhancer` fake returns exactly the noise-free speech (the test's
own recipe), so the SNR gates pass by construction and the whole bus chain is
verified deterministically — no neural weights needed in unit tests.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from director_os.audio_engine.two_bus import (
    BUS2_LIMITER_LINEAR,
    DUCK_LINEAR,
    ENCODE_HEADROOM_LIMITER_LINEAR,
    SAMPLE_RATE,
    TwoBusMixError,
    apply_gate_inplace,
    build_two_bus_mix_cmd,
    bus1_tone_chain,
    bus2_chain,
    duck_envelope,
    master_chain,
    run_two_bus_mix,
    segment_fade_chain,
)

SR = SAMPLE_RATE


def _write_wav(path: Path, pcm: np.ndarray, channels: int = 1) -> None:
    data = (np.clip(pcm, -0.999, 0.999) * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


# --------------------------------------------------------------------- envelope
def test_duck_envelope_opens_speech_and_ducks_gaps():
    sr = 1000  # small scale for exact indexing
    env = duck_envelope(2000, sr, [(0.4, 0.8)])
    assert env[0] == pytest.approx(DUCK_LINEAR, abs=1e-6)  # pre-speech: ducked
    assert env[600] == pytest.approx(1.0, abs=1e-6)  # mid speech (past 150 ms ramp): open
    assert env[1999] == pytest.approx(DUCK_LINEAR, abs=1e-6)  # post speech: ducked


def test_duck_envelope_150ms_cosine_ramps_never_click():
    sr = 48_000
    env = duck_envelope(sr, sr, [(0.0, 1.0)])
    ramp_n = int(0.15 * sr)
    # monotonic attack within the ramp window (no instantaneous jump > 6 dB/frame)
    attack = env[:ramp_n]
    assert np.all(np.diff(attack) >= -1e-5), "attack must be monotonic up"
    # starts ducked, ends open
    assert attack[0] <= DUCK_LINEAR + 0.02
    assert attack[-1] == pytest.approx(1.0, abs=0.01)


def test_duck_envelope_short_span_reaches_full_open_at_center():
    sr = 48_000
    env = duck_envelope(sr, sr, [(0.5, 0.52)])  # 20 ms blip < ramp length
    a, b = int(0.5 * sr), int(0.52 * sr)
    # anti-click: rolls in/out from the duck floor, peaks fully open mid-blip
    assert env[a] == pytest.approx(DUCK_LINEAR, abs=0.01)
    assert env[(a + b) // 2] == pytest.approx(1.0, abs=0.01)
    assert env[b - 1] <= DUCK_LINEAR + 0.02


# --------------------------------------------------------------------- chains
def test_bus1_chain_carries_eq_and_compressor_law():
    chain = bus1_tone_chain()
    # mud cut (deepened -3.0 after the measured tilt showed bass-heavy output)
    assert "equalizer=f=300:width_type=o:width=1.2:g=-3" in chain
    # clarity lift (+3) and the timbre-restoration law: presence bell + air shelf
    assert "equalizer=f=2800:width_type=o:width=1.2:g=+3" in chain
    assert "equalizer=f=4500:width_type=o:width=1.4:g=+2.5" in chain
    assert "highshelf=f=6800:g=+3" in chain
    assert "acompressor=threshold=-18dB:ratio=3:attack=10:release=120" in chain


def test_bus2_chain_band_and_bed_level():
    chain = bus2_chain(volume=0.06)
    assert "highpass=f=150" in chain and "lowpass=f=4000" in chain
    assert "volume=0.06" in chain  # -24 dB bed (opt-in via AUDIO_BUS2_VOLUME)
    assert f"limit={BUS2_LIMITER_LINEAR:g}" in chain  # soft-limit (alimiter floor)


def test_master_chain_gain_limiter_and_48k_stereo():
    chain = master_chain(9.0)
    assert "volume=+9dB" in chain
    assert "limit=0.8414" in chain  # -1.5 dBTP
    assert "level=disabled" in chain  # auto-level would re-boost to 0 dBFS
    assert "aresample=48000" in chain
    assert "channel_layouts=stereo" in chain


def test_segment_fade_chain_is_20ms_with_encode_headroom():
    chain = segment_fade_chain(6.0)
    assert "afade=t=in:st=0:d=0.0200" in chain
    assert "afade=t=out:st=5.9800:d=0.0200" in chain
    # pre-codec headroom limiter: the shipped mp4 must decode under -1.0 dBTP
    assert f"limit={ENCODE_HEADROOM_LIMITER_LINEAR:g}" in chain
    assert "level=disabled" in chain
    # never fades more than half the segment
    assert "d=0.0050" in segment_fade_chain(0.01)


def test_build_mix_cmd_default_is_pure_voice_no_bus2():
    """The 2026-11-18 ear verdict: default master = isolated voice only."""
    cmd = build_two_bus_mix_cmd(Path("b1.wav"), Path("raw.wav"), Path("out.wav"), 3.0)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "amix" not in graph  # single-input graph, no ambience bed
    assert cmd.count("-i") == 1
    assert cmd[cmd.index("-ar") + 1] == "48000"
    assert cmd[cmd.index("-ac") + 1] == "2"


def test_build_mix_cmd_opt_in_bus2_amix_no_normalize():
    cmd = build_two_bus_mix_cmd(
        Path("b1.wav"), Path("raw.wav"), Path("out.wav"), 3.0, bus2_level=0.06
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[b1][b2]amix=inputs=2:duration=first:normalize=0[mix]" in graph
    assert cmd.count("-i") == 2


# --------------------------------------------------------------------- gate apply
def test_apply_gate_inplace_ducks_stereo(tmp_path: Path):
    n = SR  # 1 s
    loud = np.full(n, 0.5, dtype=np.float32)
    wav = tmp_path / "bus1.wav"
    _write_wav(wav, np.stack([loud, loud], axis=1), channels=2)
    env = duck_envelope(n, SR, [])  # fully ducked envelope
    apply_gate_inplace(wav, env)
    out = _read_wav(wav)
    assert np.max(np.abs(out)) < 0.5 * DUCK_LINEAR * 1.1
    with pytest.raises(TwoBusMixError):
        apply_gate_inplace(tmp_path / "missing.wav", env)


# --------------------------------------------------------------------- real mix
@pytest.fixture()
def noisy_scene(tmp_path: Path) -> tuple[Path, Path, list[tuple[float, float]]]:
    """6 s speech bursts + brown noise; returns (noisy_wav, clean_wav, words)."""
    rng = np.random.default_rng(11)
    n = 6 * SR
    speech = np.zeros(n)
    words = [(0.5, 2.0), (3.0, 5.0)]
    for _i, (s0, s1) in enumerate(words):
        a, b = int(s0 * SR), int(s1 * SR)
        carrier = rng.normal(size=b - a)
        for _ in range(3):
            carrier = np.convolve(carrier, np.ones(32) / 32, mode="same")
        speech[a:b] = carrier * 0.9
    brown = np.cumsum(rng.normal(size=n))
    brown = brown / (np.max(np.abs(brown)) + 1e-9) * 0.3
    noisy = np.clip(speech * 0.8 + brown, -0.99, 0.99)
    clean = speech * 0.8
    noisy_wav, clean_wav = tmp_path / "raw_48k.wav", tmp_path / "clean_ref.wav"
    _write_wav(noisy_wav, noisy)
    _write_wav(clean_wav, clean)
    return noisy_wav, clean_wav, words


def test_two_bus_mix_drops_noise_floor_and_keeps_speech(noisy_scene, tmp_path: Path):
    noisy_wav, clean_wav, words = noisy_scene
    # Bus 1 = the noise-free speech (the fake's receipt of a perfect enhance)
    bus1 = tmp_path / "bus1_gated.wav"
    _write_wav(bus1, _read_wav(clean_wav) * 0.5)
    out = tmp_path / "master_2bus.wav"
    run_two_bus_mix(bus1, noisy_wav, out, master_gain_db=6.0)

    from director_os.audio_engine.snr import measure_snr

    raw = measure_snr(str(noisy_wav), words)
    clean = measure_snr(str(out), words)
    # the mix reduced the between-words floor vs the raw source
    assert clean["noise_floor_db"] < raw["noise_floor_db"] - 8.0
    # speech survived louder than everything between the words
    assert clean["snr_db"] > raw["snr_db"] + 6.0
    assert clean["snr_db"] >= 14.0
    # output obeys the 48 kHz stereo law
    from director_os.core.qa_probe import probe

    info = probe(out)
    assert info.sample_rate == 48_000 and info.channels == 2
