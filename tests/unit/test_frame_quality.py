"""Frame quality pre-filter — darkness/blur scoring pins (M4 CPU tier)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from director_os.perception.frame_quality import (
    FrameQualityError,
    black_frame_probe,
    score_frame_quality,
)


def _ffmpeg_filter(filter_str: str, out: Path, dur: float = 2.0) -> Path:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", filter_str.replace("{DUR}", str(dur)),
        "-frames:v", str(int(dur * 10)),
        "-y", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return out


@pytest.fixture()
def black_video(tmp_path):
    return _ffmpeg_filter("color=c=black:s=320x240:rate=10:duration={DUR}", tmp_path / "black.mp4")


@pytest.fixture()
def sharp_video(tmp_path):
    return _ffmpeg_filter(
        "color=c=white:s=320x240:rate=10:duration={DUR},noise=alls=60:allf=t",
        tmp_path / "sharp.mp4",
    )


@pytest.fixture()
def blurred_video(tmp_path):
    return _ffmpeg_filter(
        "testsrc2=size=320x240:rate=10:duration={DUR},boxblur=luma_radius=25",
        tmp_path / "blurry.mp4",
    )


def test_black_frames_scored_dark_and_unusable(black_video):
    report = score_frame_quality(str(black_video), sample_every_s=0.5)
    assert report.frames
    assert all(not f.usable for f in report.frames)
    assert all("dark" in f.reason for f in report.frames)
    assert report.usable_ratio == 0.0


def test_black_video_yields_one_merged_unusable_span(black_video):
    report = score_frame_quality(str(black_video), sample_every_s=0.5)
    assert len(report.unusable_spans) == 1
    start, end = report.unusable_spans[0]
    assert start == 0.0
    assert end == pytest.approx(2.0, abs=0.3)


def test_noisy_white_frames_score_usable(sharp_video):
    report = score_frame_quality(str(sharp_video), sample_every_s=0.5)
    assert report.frames
    assert all(f.usable for f in report.frames)
    assert all(f.brightness > 100 for f in report.frames)
    assert all(f.sharpness > 100 for f in report.frames)  # noise = huge Laplacian var


def test_heavy_boxblur_scores_blurry(blurred_video):
    report = score_frame_quality(str(blurred_video), sample_every_s=0.5)
    assert report.frames
    blurry = [f for f in report.frames if not f.usable]
    assert len(blurry) >= len(report.frames) // 2
    assert all(f.sharpness < 35.0 for f in blurry)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FrameQualityError, match="cannot open"):
        score_frame_quality(str(tmp_path / "missing.mp4"))


def test_blackdetect_finds_whole_scene_blackout(black_video):
    spans = black_frame_probe(str(black_video))
    assert spans and spans[0][0] == pytest.approx(0.0, abs=0.1)
