"""Audio event detection (YAMNet CPU) — sirens, gunshots, shouting, glass (issue #12).

Span-building logic is pure and tested hermetically with crafted score matrices.
The real-model path runs when data/models/yamnet.onnx is present
(scripts/fetch_yamnet.py) and skips otherwise; the synthetic-siren recipe below
is proven to fire YAMNet's Siren class at mean 0.94 / max 0.997.
"""

from __future__ import annotations

import numpy as np
import pytest
from director_os.perception.audio_events import (
    CHUNK_WINDOWS,
    EVENT_GROUPS,
    FRAME_HOP_S,
    FRAME_WINDOW_S,
    OVERLAP_WINDOWS,
    SAMPLE_RATE,
    AudioEventBackend,
    AudioEventsNotAvailable,
    YamnetOnnxBackend,
    build_event_spans,
    decode_audio_pcm16k,
    resolve_yamnet_model_path,
)


def _scores(n_frames: int, classes: int = 521) -> np.ndarray:
    return np.zeros((n_frames, classes), dtype=np.float32)


def _set(frames: np.ndarray, cls: int, frame_slice: slice, value: float) -> None:
    frames[frame_slice, cls] = value


# ------------------------------------------------------------- group mapping


def test_event_groups_documented_indices():
    # the indices ARE the contract with yamnet_class_map.csv (fetch-script pinned)
    assert 390 in EVENT_GROUPS["siren"]  # Siren
    assert 317 in EVENT_GROUPS["siren"]  # Police car (siren)
    assert 421 in EVENT_GROUPS["gunshot"]  # Gunshot, gunfire
    assert 6 in EVENT_GROUPS["shout"]  # Shout
    assert 435 in EVENT_GROUPS["glass"]  # Glass
    assert 304 in EVENT_GROUPS["alarm"]  # Car alarm


# ------------------------------------------------------------- span building


def test_single_run_above_threshold_becomes_one_span():
    scores = _scores(20)
    _set(scores, 390, slice(4, 9), 0.8)  # siren frames 4..8
    spans = build_event_spans(scores, threshold=0.3)
    siren = [s for s in spans if s.kind == "siren"]
    assert len(siren) == 1
    s = siren[0]
    assert s.start_s == pytest.approx(4 * FRAME_HOP_S)
    assert s.end_s == pytest.approx(9 * FRAME_HOP_S + 0.96 - FRAME_HOP_S)
    assert s.peak_score == pytest.approx(0.8)


def test_short_gap_within_merge_tolerance_joins():
    scores = _scores(20)
    _set(scores, 390, slice(0, 3), 0.7)  # run A: frames 0..2
    _set(scores, 390, slice(5, 8), 0.9)  # run B: frames 5..7 (gap = 2 frames)
    # merge_gap 1.0 s ~= 2 frames at 0.48 s hop -> gap 2 <= 2 merges
    spans = build_event_spans(scores, threshold=0.3, merge_gap_s=1.0)
    siren = [s for s in spans if s.kind == "siren"]
    assert len(siren) == 1
    assert siren[0].start_s == pytest.approx(0.0)
    assert siren[0].peak_score == pytest.approx(0.9)  # peak over the merged run


def test_long_gap_stays_split():
    scores = _scores(40)
    _set(scores, 390, slice(0, 3), 0.7)
    _set(scores, 390, slice(30, 33), 0.9)  # 27-frame gap — two distinct events
    spans = build_event_spans(scores, threshold=0.3, merge_gap_s=1.0)
    assert len([s for s in spans if s.kind == "siren"]) == 2


def test_single_frame_blip_is_dropped():
    scores = _scores(20)
    _set(scores, 421, slice(5, 6), 0.95)  # one frame — detector noise
    spans = build_event_spans(scores, threshold=0.3, min_frames=2)
    assert spans == []


def test_two_frame_gunshot_survives():
    scores = _scores(20)
    _set(scores, 421, slice(10, 12), 0.85)
    spans = build_event_spans(scores, threshold=0.3, min_frames=2)
    gunshots = [s for s in spans if s.kind == "gunshot"]
    assert len(gunshots) == 1
    assert gunshots[0].peak_score == pytest.approx(0.85)


def test_group_max_member_wins():
    scores = _scores(20)
    _set(scores, 317, slice(2, 6), 0.4)  # Police car (siren) — member of siren group
    _set(scores, 390, slice(3, 5), 0.6)  # Siren proper
    spans = build_event_spans(scores, threshold=0.3)
    siren = [s for s in spans if s.kind == "siren"][0]
    assert siren.peak_score == pytest.approx(0.6)


def test_quiet_scene_has_no_spans():
    assert build_event_spans(_scores(20), threshold=0.3) == []


def test_malformed_scores_rejected():
    with pytest.raises(ValueError):
        build_event_spans(np.zeros(10, dtype=np.float32))


# -------------------------------------------------------------- port + decode


def test_protocol_is_structural():
    class FakeBackend:
        name = "fake"

        def detect_events(self, audio_path, threshold=0.30):
            from director_os.perception.audio_events import AudioEventSweep

            return AudioEventSweep(backend="fake", duration_s=1.0, threshold=threshold)

    backend: AudioEventBackend = FakeBackend()  # must type-check structurally
    assert backend.name == "fake"


def test_decode_audio_pcm16k_returns_mono_16k(tmp_path):
    import subprocess

    # 2 s of 440 Hz sine, stereo 48 kHz in a wav — decode must flatten to 32k samples
    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    wave = decode_audio_pcm16k(wav)
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert len(wave) == pytest.approx(32000, rel=0.01)  # 2 s * 16 kHz
    assert float(np.abs(wave).max()) > 0.1


def test_decode_bad_file_raises_soft():
    with pytest.raises(AudioEventsNotAvailable):
        decode_audio_pcm16k("/nonexistent/audio.mp4")


def test_model_resolution_missing_raises_with_hint(tmp_path, monkeypatch):
    import director_os.perception.audio_events as ae

    monkeypatch.setattr(ae, "_default_model_candidates", lambda _f: [])
    monkeypatch.delenv("DIRECTOR_YAMNET_PATH", raising=False)
    with pytest.raises(AudioEventsNotAvailable, match="fetch_yamnet"):
        ae.resolve_yamnet_model_path()


def test_model_resolution_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "yamnet.onnx"
    fake.write_bytes(b"onnx")
    monkeypatch.setenv("DIRECTOR_YAMNET_PATH", str(fake))
    assert resolve_yamnet_model_path() == fake


# --------------------------------------------------------- real model (optional)


def _yamnet_present() -> bool:
    try:
        resolve_yamnet_model_path()
        return True
    except AudioEventsNotAvailable:
        return False


def _siren_wave(seconds: float = 6.0) -> np.ndarray:
    """Synthetic wail proven to fire YAMNet Siren (mean 0.94): harmonics + AM."""
    sr = 16000
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    f = 550 + (t % 1.6) / 1.6 * 700
    phase = 2 * np.pi * np.cumsum(f) / sr
    w = (
        0.5 * np.sin(phase)
        + 0.22 * np.sin(2 * phase)
        + 0.12 * np.sin(3 * phase)
        + 0.05 * np.sin(4 * phase)
    )
    am = 0.75 + 0.25 * np.sin(2 * np.pi * 0.9 * t)
    noise = np.random.default_rng(7).normal(0, 0.015, len(w)).astype(np.float32)
    return np.tanh(1.6 * (w * am + noise)).astype(np.float32)


@pytest.mark.skipif(not _yamnet_present(), reason="yamnet.onnx not fetched")
def test_real_yamnet_siren_fires(tmp_path):
    import wave as wavemod

    wave = _siren_wave()
    path = tmp_path / "siren.wav"
    with wavemod.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((wave * 32767).astype("<i2").tobytes())
    sweep = YamnetOnnxBackend().detect_events(path)
    assert "siren" in sweep.kinds()
    siren = [s for s in sweep.events if s.kind == "siren"]
    assert siren[0].peak_score >= 0.5


@pytest.mark.skipif(not _yamnet_present(), reason="yamnet.onnx not fetched")
def test_real_yamnet_silence_is_quiet(tmp_path):
    import wave as wavemod

    wave = np.zeros(16000 * 4, dtype=np.float32)  # 4 s digital silence
    path = tmp_path / "silence.wav"
    with wavemod.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((wave * 32767).astype("<i2").tobytes())
    sweep = YamnetOnnxBackend().detect_events(path)
    assert sweep.events == []


@pytest.mark.skipif(not _yamnet_present(), reason="yamnet.onnx not fetched")
def test_full_tape_chunked_inference_keeps_events_and_memory(tmp_path):
    """A >61s tape rides the CHUNKED yamnet pass: the synthetic siren is found
    at its placement, and the frame math stays on the 0.48s grid. One giant
    session.run on 43-min audio OOM-killed the pipeline (4.1 GB) — this pins
    the chunk contract: same model, whole windows, frame-exact re-emission."""
    import resource
    import wave as wavemod

    sr = 16000
    src = np.zeros(130 * sr, dtype=np.float32)
    src[20 * sr : 45 * sr] = _siren_wave(25.0)
    path = tmp_path / "tape.wav"
    with wavemod.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((src * 32767).astype("<i2").tobytes())

    sweep = YamnetOnnxBackend().detect_events(path)
    assert sweep.duration_s == pytest.approx(130.0, rel=0.01)
    assert sweep.events, "the siren must survive chunking"
    siren = [s for s in sweep.events if s.kind == "siren"]
    assert siren, [e.kind for e in sweep.events]
    assert siren[0].start_s <= 21.0
    assert siren[0].end_s >= 44.0
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024 * 1024  # <2 GB


def test_chunk_math_is_window_and_hop_congruent():
    """chunk = 64 windows; step = 62 windows; every chunk start lands on a
    whole hop so dropped leading frames re-align frame-exactly."""
    window = round(FRAME_WINDOW_S * SAMPLE_RATE)
    hop = round(FRAME_HOP_S * SAMPLE_RATE)
    chunk = CHUNK_WINDOWS * window
    step = (CHUNK_WINDOWS - OVERLAP_WINDOWS) * window
    assert chunk % window == 0 and chunk % hop == 0
    assert step % window == 0 and step % hop == 0
    assert step < chunk, "chunks must overlap"
