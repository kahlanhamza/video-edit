"""Bodycam overlay OCR — parser pins + backend routing (M4 job 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.perception.timestamp_ocr import (
    TimestampOcrError,
    extract_frame,
    parse_timestamp_text,
    read_bodycam_timestamp,
    read_timestamp_gemini,
)
from director_os.perception.vision import FakeVisionClient, SceneInsight

EUSTIS = "2022-07-04 18:11:55 -0400 / AXON BODY 3"


def test_parse_full_eustis_stamp():
    stamp = parse_timestamp_text(EUSTIS)
    assert stamp is not None
    assert stamp.iso == "2022-07-04T18:11:55-04:00"
    assert stamp.device == "AXON BODY 3"
    assert stamp.tz == "-04:00"


def test_parse_missing_seconds_defaults_to_zero():
    stamp = parse_timestamp_text("2022-07-04 18:11 AXON BODY 3")
    assert stamp is not None
    assert stamp.iso == "2022-07-04T18:11:00"
    assert stamp.device == "AXON BODY 3"


def test_parse_no_stamp_returns_none():
    assert parse_timestamp_text("nothing to see here") is None
    assert parse_timestamp_text("") is None


def test_parse_without_device_still_returns_stamp():
    stamp = parse_timestamp_text("2022-01-31 08:05:09")
    assert stamp is not None
    assert stamp.iso == "2022-01-31T08:05:09"
    assert stamp.device is None


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    import subprocess

    out = tmp_path / "tiny.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=2",
        "-frames:v", "20", "-y", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return out


def test_extract_frame_pulls_one_fullres_png(tiny_video, tmp_path):
    out = extract_frame(str(tiny_video), t_s=0.5, out_png=tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 1_000


def test_gemini_backend_parses_verbatim_transcription(tmp_path):
    frame = tmp_path / "f.png"
    frame.write_bytes(b"\x89PNG fake")
    fake = FakeVisionClient(
        [SceneInsight(timestamp_s=0.5, kind="observation", description=EUSTIS, confidence=0.95)]
    )
    stamp = read_timestamp_gemini(frame, fake, t_s=0.5)
    assert stamp is not None and stamp.device == "AXON BODY 3"


def test_auto_backend_without_any_backend_raises(tmp_path):
    with pytest.raises(TimestampOcrError, match="no OCR backend"):
        read_bodycam_timestamp(str(tmp_path / "missing.mp4"), backend="gemini", client=None)


def test_unknown_backend_raises(tmp_path):
    with pytest.raises(TimestampOcrError, match="unknown OCR backend"):
        read_bodycam_timestamp("x.mp4", backend="cloud-nine")
