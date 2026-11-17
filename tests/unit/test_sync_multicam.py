"""sync_multicam — synthetic multi-cam scenes with known offsets and drift.

Every fixture builds angles from one deterministic "scene" signal whose wall
placement is known exactly, so offsets are asserted to ±1 ms on pure audio and
±20 ms through lossy AAC extraction, drift to ±10 ppm.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from director_os.perception.sync import (
    SyncError,
    SyncResult,
    cross_correlate,
    extract_audio,
    load_samples,
    sync_multicam,
    sync_pair,
)

FS = 16_000


def _scene(dur_s: float, seed: int = 1234, lowpass_hz: float = 800.0) -> np.ndarray:
    """Bandlimited deterministic scene.

    Lowpass keeps a ~1.25 ms correlation length so short probes stay coherent
    under clock drift (pure white noise decorrelates at a 1-sample shift, which
    is exactly the failure mode drift estimation has to survive).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(dur_s * FS))
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / FS)
    spectrum[freqs > lowpass_hz] = 0.0
    y = np.fft.irfft(spectrum, len(x))
    y = y / (float(np.std(y)) + 1e-9) * 0.5
    return y.astype(np.float32)


def _ref_from_scene(scene: np.ndarray) -> np.ndarray:
    return scene  # reference records the scene from its own t=0


def _angle_from_scene(
    scene: np.ndarray, offset_s: float, dur_s: float, drift_ppm: float = 0.0
) -> np.ndarray:
    """Angle track whose clock reads `wall - offset_s - drift*t`, wall = ref time.

    other[n] = ref_at(t_o + offset_s + drift_ppm*t_o/1e6) — `offset_s > 0`
    means this angle started LATER than the reference (see sync.py docstring).
    """
    n = int(dur_s * FS)
    t_o = np.arange(n, dtype=np.float64) / FS
    wall = t_o + offset_s + drift_ppm * t_o / 1e6
    return np.interp(wall, np.arange(len(scene)) / FS, scene).astype(np.float32)


# ---------------------------------------------------------------- pure audio


def test_pair_recovers_positive_offset_sub_ms():
    ref = _ref_from_scene(_scene(40.0))
    other = _angle_from_scene(ref, offset_s=7.0, dur_s=33.0)
    offset_s, drift_ppm, conf = sync_pair(ref, other, FS, search_s=60.0)
    assert abs(offset_s - 7.0) <= 0.001
    assert abs(drift_ppm) <= 5.0
    assert conf > 0.4


def test_pair_recovers_negative_offset_other_started_earlier():
    scene = _scene(40.0)
    ref = _ref_from_scene(scene)
    preroll = _scene(12.0, seed=99)  # before the reference started rolling
    # other_time 0..12 = preroll; other_time 12..32 = scene[0..20]
    other = np.concatenate([preroll, scene[: int(20.0 * FS)]])
    offset_s, _, conf = sync_pair(ref, other, FS, search_s=60.0)
    assert abs(offset_s - (-12.0)) <= 0.001
    assert conf > 0.3


def test_drift_recovered_over_long_overlap():
    ref = _ref_from_scene(_scene(200.0, seed=7))
    other = _angle_from_scene(ref, offset_s=3.0, dur_s=190.0, drift_ppm=50.0)
    offset_s, drift_ppm, conf = sync_pair(ref, other, FS)
    assert abs(offset_s - 3.0) <= 0.05  # head window averages lag over its span
    assert abs(drift_ppm - 50.0) <= 10.0
    assert conf > 0.3


def test_unrelated_tracks_score_low_confidence():
    a = _scene(30.0, seed=1)
    b = _scene(30.0, seed=2)
    _, _, conf = sync_pair(a, b, FS, search_s=10.0)
    assert conf < 0.2


def test_cross_correlate_sign_convention():
    """ref[i] ≈ other[i + k] — a pure integer delay puts the peak at k > 0."""
    ref = _scene(5.0, seed=3)
    delay = 3000  # 187.5 ms
    other = np.concatenate([np.zeros(delay, dtype=np.float32), ref])
    lag_s, conf = cross_correlate(ref, other, FS, search_s=2.0)
    assert abs(lag_s - delay / FS) <= 1.0 / FS
    assert conf > 0.9


def test_sync_result_angle_source_time():
    r = SyncResult("b", offset_s=7.0, drift_ppm=0.0, confidence=0.9)
    assert r.angle_source_time(10.0) == 3.0  # seek the angle at ref 10s -> t=3s


def test_too_little_overlap_raises():
    ref = _ref_from_scene(_scene(3.0))
    with pytest.raises(SyncError):
        sync_pair(ref, _angle_from_scene(ref, 1.0, 3.0), FS)


# ---------------------------------------------------------------- real files


def _make_angle_mp4(out: str, seed: int, delay_ms: int, dur_s: int = 25) -> None:
    """Tiny real video with deterministic pink-noise audio, delayed by `delay_ms`."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x240:rate=10:duration={dur_s}",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=d={dur_s}:s=16000:seed={seed}:c=pink:a=0.6",
            "-filter_complex",
            f"[1:a]adelay={delay_ms}:all=1,apad[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-t",
            str(dur_s),
            out,
        ],
        check=True,
    )


def test_multicam_end_to_end_with_video_extraction(tmp_path):
    # adelay pushes content LATER in the file: the angle caught the scene N s in,
    # i.e. that device started recording N s BEFORE the reference -> offset < 0
    _make_angle_mp4(str(tmp_path / "a.mp4"), seed=77, delay_ms=0)
    _make_angle_mp4(str(tmp_path / "b.mp4"), seed=77, delay_ms=7000)
    _make_angle_mp4(str(tmp_path / "c.mp4"), seed=77, delay_ms=2100)

    sync = sync_multicam(
        {"a": tmp_path / "a.mp4", "b": tmp_path / "b.mp4", "c": tmp_path / "c.mp4"},
        reference_asset_id="a",
        workdir=tmp_path / "work",
    )
    assert sync.reference_asset_id == "a"
    ref = sync.for_asset("a")
    assert ref is not None and ref.offset_s == 0.0 and ref.drift_ppm == 0.0
    b = sync.for_asset("b")
    c = sync.for_asset("c")
    assert b is not None and abs(b.offset_s - (-7.0)) <= 0.02  # AAC codec delay slack
    assert c is not None and abs(c.offset_s - (-2.1)) <= 0.02
    # angle seek helper consistent with the measured offset
    assert abs(b.angle_source_time(20.0) - (20.0 - b.offset_s)) < 1e-9


def test_multicam_requires_reference_and_two_angles(tmp_path):
    wav = tmp_path / "x.wav"
    import wave

    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((np.zeros(FS, dtype=np.int16)).tobytes())
    with pytest.raises(SyncError):
        sync_multicam({"x": wav}, reference_asset_id="missing")
    with pytest.raises(SyncError):
        sync_multicam({"x": wav}, reference_asset_id="x")


def test_extract_and_load_roundtrip(tmp_path):
    _make_angle_mp4(str(tmp_path / "a.mp4"), seed=5, delay_ms=0, dur_s=8)
    wav = extract_audio(tmp_path / "a.mp4", tmp_path / "a.wav")
    samples = load_samples(wav)
    assert samples.dtype == np.float32
    assert abs(len(samples) / FS - 8.0) < 0.5
    assert float(np.max(np.abs(samples))) > 0.01
