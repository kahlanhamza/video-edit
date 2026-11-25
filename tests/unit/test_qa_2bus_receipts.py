"""The 2-bus receipt law for QA SNR + the enhance carry-through probe.

Full-tape Eustis bug (2026-11-25, window receipts vs aggregate): the
whole-mix SNR meter measures the between-word floor of the AGGREGATE —
which is the DUCKED bed (the -14 dB duck) — so a perfectly enhanced kept
band reads "raw_snr=6.8 clean_snr=12.1" while the per-window receipts prove
"snr_raw=7.3 -> snr_clean=34.8". The gate rides the receipts; the shipped
file gets the swap-in lie-detector instead (the raw mix handed back as
"enhanced" reads enhancement-neutral and dies there).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import director_os.audio_engine.snr as snr_mod
from director_os.core.qa_probe import AudioVerification, _audio_verification_checks


def _fake_tilts(path: str, intervals: Any) -> dict[str, float]:
    return {
        "tilt_presence_db": -14.0,
        "tilt_air_db": -19.0,
        "tilt_lowmid_db": 0.0,
    }


def _patch_audits(monkeypatch, ship: dict[str, float]) -> None:
    def _measure(path: str, intervals, restrict_windows=None):
        if "raw_48k" in str(path):
            raise AssertionError("the raw reference was re-measured in receipt mode")
        return ship

    monkeypatch.setattr(snr_mod, "measure_snr", _measure)
    monkeypatch.setattr(snr_mod, "speech_band_tilts", _fake_tilts)
    monkeypatch.setattr(snr_mod, "speech_crest_db", lambda path, intervals: 10.0)
    monkeypatch.setattr(snr_mod, "peak_true_dbfs", lambda path: -3.0)


def _verification(tmp_path: Path, **overrides: Any) -> AudioVerification:
    fields: dict[str, Any] = {
        "raw_audio_path": str(tmp_path / "raw_48k.wav"),
        "enhancer_engine": "deepfilternet3",
        "enhanced_windows": [(0.5, 2.5)],
        "raw_keep_windows": [(0.5, 2.5)],
    }
    fields.update(overrides)
    return AudioVerification(**fields)


_ENHANCED_SHIP = {"snr_db": 20.0, "noise_floor_db": -55.0}
_RAW_SHIP = {"snr_db": 6.8, "noise_floor_db": -14.7}


def test_receipts_drive_the_snr_gate_and_enhancement_carries_through(monkeypatch, tmp_path):
    _patch_audits(monkeypatch, ship=dict(_ENHANCED_SHIP))
    verification = _verification(
        tmp_path,
        receipt_raw_snr_db=6.7,
        receipt_clean_snr_db=107.5,
        receipt_delta_noise_db=-20.0,
    )
    checks = _audio_verification_checks(tmp_path / "final.mp4", verification, [(0.8, 2.0)])
    by_name = {c.name: c for c in checks}
    assert by_name["snr_verification"].ok is True
    assert by_name["snr_verification"].detail.startswith("2-bus receipts")
    assert by_name["enhance_carry_through"].ok is True
    assert "made it into the shipped bitstream" in by_name["enhance_carry_through"].detail


def test_receipt_gate_fails_a_shot_receipt_pair(monkeypatch, tmp_path):
    _patch_audits(monkeypatch, ship=dict(_ENHANCED_SHIP))
    verification = _verification(
        tmp_path,
        receipt_raw_snr_db=6.8,
        receipt_clean_snr_db=12.1,
        receipt_delta_noise_db=2.3,
    )
    checks = _audio_verification_checks(tmp_path / "final.mp4", verification, [(0.8, 2.0)])
    by_name = {c.name: c for c in checks}
    assert by_name["snr_verification"].ok is False
    assert "insufficient" in by_name["snr_verification"].detail


def test_carry_through_flags_the_raw_mix_shipped_in_place(monkeypatch, tmp_path):
    _patch_audits(monkeypatch, ship=dict(_RAW_SHIP))
    verification = _verification(
        tmp_path,
        receipt_raw_snr_db=6.7,
        receipt_clean_snr_db=107.5,
        receipt_delta_noise_db=-20.0,
    )
    checks = _audio_verification_checks(tmp_path / "final.mp4", verification, [(0.8, 2.0)])
    by_name = {c.name: c for c in checks}
    assert by_name["enhance_carry_through"].ok is False
    assert "ENHANCEMENT-NEUTRAL" in by_name["enhance_carry_through"].detail


def test_no_receipts_falls_back_to_the_classic_measure(monkeypatch, tmp_path):
    calls: list[str] = []

    def _record(path: str, intervals, restrict_windows=None):
        calls.append(str(path))
        return {"snr_db": 20.0, "noise_floor_db": -55.0}

    monkeypatch.setattr(snr_mod, "measure_snr", _record)
    monkeypatch.setattr(snr_mod, "speech_band_tilts", _fake_tilts)
    monkeypatch.setattr(snr_mod, "speech_crest_db", lambda path, intervals: 10.0)
    monkeypatch.setattr(snr_mod, "peak_true_dbfs", lambda path: -3.0)
    verification = _verification(
        tmp_path,
        raw_clock_speech_intervals=[(0.8, 2.0)],
        receipt_raw_snr_db=None,
    )
    checks = _audio_verification_checks(tmp_path / "final.mp4", verification, [(0.8, 2.0)])
    by_name = {c.name: c for c in checks}
    assert len(calls) == 2
    assert not by_name["snr_verification"].detail.startswith("2-bus receipts")
    assert "enhance_carry_through" not in by_name
