"""Bodycam overlay OCR — parser pins + backend routing (M4 job 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.perception.timestamp_ocr import (
    TimestampOcrError,
    extract_frame,
    parse_overlay_lines,
    parse_timestamp_text,
    read_bodycam_timestamp,
    read_timestamp_gemini,
    read_timestamp_paddle,
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
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=10:duration=2",
        "-frames:v",
        "20",
        "-y",
        str(out),
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


# ---------------------------------------------------------------- paddle lane
def test_parse_overlay_lines_joins_device_and_location():
    """PaddleOCR returns reading-order blocks: date, badge and street sign on
    DIFFERENT lines — the parser must join them into one stamp."""
    stamp = parse_overlay_lines(
        [
            "2022-07-04 18:11:55 -0400",
            "AXON BODY 3",
            "321 N OAK ST",
            "OFFICER 12",
        ]
    )
    assert stamp is not None
    assert stamp.iso == "2022-07-04T18:11:55-04:00"
    assert stamp.device == "AXON BODY 3"
    assert stamp.location_text == "321 N OAK ST"


def test_parse_overlay_lines_route_shield_and_marker():
    assert parse_overlay_lines(["2022-01-31 08:05:09", "US-192 W"]).location_text == "US-192 W"
    assert (
        parse_overlay_lines(["2022-01-31 08:05:09", "MILE MARKER 205"]).location_text
        == "MILE MARKER 205"
    )
    assert parse_overlay_lines(["2022-01-31 08:05:09", "SR 434 E"]).location_text == "SR 434 E"


def test_parse_overlay_lines_noise_never_becomes_location():
    """OCR noise / dispatch chatter must not ship as a card fact."""
    stamp = parse_overlay_lines(["2022-01-31 08:05:09", "DISPATCH RESPONDING", "UNIT 12 ON SCENE"])
    assert stamp is not None
    assert stamp.location_text is None


def test_parse_overlay_lines_no_timestamp_returns_none():
    assert parse_overlay_lines(["N OAK ST", "AXON BODY 3"]) is None
    assert parse_overlay_lines([]) is None


def test_paddle_backend_via_injected_lines(tmp_path):
    """The paddle adapter is a thin mapping over a line provider — the port
    seam keeps the test hermetic (no paddleocr install needed)."""
    frame = tmp_path / "f.png"
    frame.write_bytes(b"\x89PNG fake")
    stamp = read_timestamp_paddle(frame, lines_fn=lambda _p: [EUSTIS, "321 N OAK ST"])
    assert stamp is not None
    assert stamp.iso == "2022-07-04T18:11:55-04:00"
    assert stamp.device == "AXON BODY 3"
    assert stamp.location_text == "321 N OAK ST"


def test_auto_prefers_local_paddle_lane(tmp_path, monkeypatch):
    """Cost law: auto uses the LOCAL free PaddleOCR when importable, before
    the Modal fleet or the keyed Gemini call."""
    import director_os.perception.timestamp_ocr as tmod

    monkeypatch.setattr(tmod, "paddleocr_available", lambda: True)
    monkeypatch.setattr(tmod, "extract_frame", lambda _v, t_s=0.5: tmp_path / "fake.png")
    monkeypatch.setattr(
        tmod,
        "_paddle_lines_fn",
        lambda _p: ["2022-07-04 18:11:55 -0400", "AXON BODY 3"],
    )
    stamp = read_bodycam_timestamp("whatever.mp4", backend="auto")
    assert stamp is not None and stamp.device == "AXON BODY 3"


def test_auto_without_paddle_and_no_key_raises(tmp_path, monkeypatch):
    import director_os.perception.timestamp_ocr as tmod

    monkeypatch.setattr(tmod, "paddleocr_available", lambda: False)
    with pytest.raises(TimestampOcrError, match="no OCR backend"):
        read_bodycam_timestamp(str(tmp_path / "missing.mp4"), backend="auto")
