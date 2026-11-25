"""QA acoustic-law clock alignment — raw->master speech-window remap.

Full-tape runway bug (2026-11-25, Eustis 2627s): the enhance receipt carries
speech windows on the RAW asset clock, but the QA audits probe the SHIPPED
master (a kept-band cut, 782s). Probing a 782s file with windows at t=2531s
reads silence/limiter noise as the "clean" floor and fails SNR/RMS on
perfectly-enhanced speech.
"""

from __future__ import annotations

from pathlib import Path

import director_os.audio_engine.snr as snr_mod
from director_os.core.qa_probe import AudioVerification, _audio_verification_checks
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.timeline import (
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from director_os.worker.pipeline import (
    master_clock_kept_windows,
    master_clock_speech_intervals,
)


def _timeline(ranges: list[tuple[float, float]], asset_id: str = "a1"):
    items = [
        ClipSpec(
            name=f"raw_{i}",
            source_start_s=s,
            dur_s=d,
            meta=RawClipMeta(asset_id=asset_id),
        )
        for i, (s, d) in enumerate(ranges)
    ]
    spec = TimelineSpec(
        name="t",
        fps=30,
        width=320,
        height=240,
        tracks=[TrackSpec(name="V1_master", items=items)],
    )
    return build_timeline(spec)


def test_remap_walks_kept_window_offsets():
    timeline = _timeline([(100.0, 10.0), (200.0, 5.0)])
    out = master_clock_speech_intervals([(102.0, 105.0)], timeline, "a1")
    assert out == [(1.85, 5.15)]  # the word-edge pad rides the mapping,


def test_remap_clips_interval_at_window_edges():
    timeline = _timeline([(100.0, 10.0)])
    out = master_clock_speech_intervals([(98.0, 103.0)], timeline, "a1")
    assert out == [(0.0, 3.15)]  # pad clamped at master 0, raw-side head dropped


def test_remap_drops_windows_outside_kept_ranges():
    timeline = _timeline([(100.0, 10.0), (200.0, 5.0)])
    out = master_clock_speech_intervals([(201.0, 204.0), (250.0, 260.0)], timeline, "a1")
    assert out == [(10.85, 14.15)]  # the tail window only; beyond-the-tape is dropped


def test_remap_keeps_pad_separated_spans():
    timeline = _timeline([(100.0, 10.0)])
    out = master_clock_speech_intervals([(100.2, 103.0), (103.5, 106.0)], timeline, "a1")
    assert [(round(a, 2), round(b, 2)) for a, b in out] == [
        (0.05, 3.15),
        (3.35, 6.15),
    ]  # the 0.15 pads keep the span gap unmerged


def test_remap_passthrough_without_timeline_or_asset():
    windows = [(10.0, 12.0)]
    assert master_clock_speech_intervals(windows, None, "a1") == windows
    timeline = _timeline([(100.0, 10.0)], asset_id="other")
    assert master_clock_speech_intervals(windows, timeline, "a1") == windows


def _audio_checks(path: Path, verification: AudioVerification, intervals):
    return _audio_verification_checks(path, verification, list(intervals))


def test_raw_reference_measured_on_raw_clock(monkeypatch, tmp_path):
    calls: list[tuple[str, list[tuple[float, float]]]] = []

    def _fake(path: str, intervals, restrict_windows=None):
        calls.append((str(path), list(intervals), restrict_windows))
        return {"snr_db": 30.0, "noise_floor_db": -60.0}

    monkeypatch.setattr(snr_mod, "measure_snr", _fake)
    monkeypatch.setattr(
        snr_mod,
        "speech_band_tilts",
        lambda path, intervals: {
            "tilt_presence_db": -14.0,
            "tilt_air_db": -19.0,
            "tilt_lowmid_db": 0.0,
        },
    )
    monkeypatch.setattr(snr_mod, "speech_crest_db", lambda path, intervals: 10.0)
    monkeypatch.setattr(snr_mod, "peak_true_dbfs", lambda path: -3.0)

    verification = AudioVerification(
        raw_audio_path=str(tmp_path / "raw_48k.wav"),
        enhancer_engine="deepfilternet3",
        raw_clock_speech_intervals=[(2500.0, 2510.0)],
    )
    master_intervals = [(0.5, 6.0)]
    checks = _audio_checks(tmp_path / "final.mp4", verification, master_intervals)
    raw_call, clean_call = calls[0], calls[1]
    assert raw_call[0].endswith("raw_48k.wav")
    assert raw_call[1] == [(2500.0, 2510.0)]
    assert raw_call[2] is None  # no kept windows -> whole-file measure (legacy 1:1)
    assert clean_call[1] == [(0.5, 6.0)]
    assert clean_call[2] is None
    names = {c.name for c in checks}
    assert {
        "snr_verification",
        "timbre_not_muffled",
        "natural_dynamics",
        "zero_clipping",
        "enhancer_engine_used",
    } <= names


def test_snr_check_reports_the_two_clocks(monkeypatch, tmp_path):
    """The failure detail must show raw and clean measured independently —
    the runway regression shipped 'raw_snr=6.8 clean_snr=8.8' on enhanced
    audio because both sides sampled the same wrong clock."""

    def _fake(path: str, intervals, restrict_windows=None):
        if "raw_48k" in path:
            return {"snr_db": 20.0, "noise_floor_db": -55.0}
        return {"snr_db": 28.0, "noise_floor_db": -63.0}

    monkeypatch.setattr(snr_mod, "measure_snr", _fake)
    monkeypatch.setattr(
        snr_mod,
        "speech_band_tilts",
        lambda path, intervals: {
            "tilt_presence_db": -14.0,
            "tilt_air_db": -19.0,
            "tilt_lowmid_db": 0.0,
        },
    )
    monkeypatch.setattr(snr_mod, "speech_crest_db", lambda path, intervals: 10.0)
    monkeypatch.setattr(snr_mod, "peak_true_dbfs", lambda path: -3.0)

    verification = AudioVerification(
        raw_audio_path=str(tmp_path / "raw_48k.wav"),
        enhancer_engine="deepfilternet3",
        raw_clock_speech_intervals=[(2600.0, 2610.0)],
    )
    checks = _audio_checks(tmp_path / "final.mp4", verification, [(1.0, 4.0)])
    snr = next(c for c in checks if c.name == "snr_verification")
    assert snr.ok
    assert "raw_snr=20.0dB" in snr.detail and "clean_snr=28.0dB" in snr.detail


def test_kept_windows_map_to_both_clocks():
    """The SNR region-quantization needs the kept band on BOTH clocks."""
    timeline = _timeline([(100.0, 10.0), (200.0, 5.0), (300.0, 2.0)])
    master_windows, raw_windows = master_clock_kept_windows(timeline, "a1")
    assert master_windows == [(0.0, 17.0)]  # adjacent maps merge in master clock
    assert raw_windows == [(100.0, 110.0), (200.0, 205.0), (300.0, 302.0)]


def test_verification_quantizes_snr_to_kept_windows(monkeypatch, tmp_path):
    """enhanced_windows/raw_keep_windows must reach measure_snr verbatim —
    the shipped floor can never beat the raw floor while gap ambience (raw
    by design) is allowed to meter as 'noise'."""
    seen: dict[str, object] = {}

    def _fake(path: str, intervals, restrict_windows=None):
        seen[path] = restrict_windows
        if "raw_48k" in path:
            return {"snr_db": 20.0, "noise_floor_db": -55.0}
        return {"snr_db": 30.0, "noise_floor_db": -60.0}

    monkeypatch.setattr(snr_mod, "measure_snr", _fake)
    monkeypatch.setattr(
        snr_mod,
        "speech_band_tilts",
        lambda path, intervals: {
            "tilt_presence_db": -14.0,
            "tilt_air_db": -19.0,
            "tilt_lowmid_db": 0.0,
        },
    )
    monkeypatch.setattr(snr_mod, "speech_crest_db", lambda path, intervals: 10.0)
    monkeypatch.setattr(snr_mod, "peak_true_dbfs", lambda path: -3.0)

    verification = AudioVerification(
        raw_audio_path=str(tmp_path / "raw_48k.wav"),
        enhancer_engine="deepfilternet3",
        raw_clock_speech_intervals=[(2500.0, 2510.0)],
        enhanced_windows=[(0.0, 17.0)],
        raw_keep_windows=[(100.0, 110.0), (200.0, 207.0)],
    )
    checks = _audio_checks(tmp_path / "final.mp4", verification, [(0.5, 6.0)])
    assert seen[str(tmp_path / "raw_48k.wav")] == [(100.0, 110.0), (200.0, 207.0)]
    assert seen[str(tmp_path / "final.mp4")] == [(0.0, 17.0)]
    assert all(c.ok for c in checks)
