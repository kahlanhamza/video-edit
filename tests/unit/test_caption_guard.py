"""Burned-caption guard — synthetic text in the bottom band must flag."""

from __future__ import annotations

import subprocess
from pathlib import Path

from director_os.vision_fx.caption_guard import (
    _sample_timestamps,
    bottom_band_text_score,
    scan_burned_captions,
)
from PIL import Image, ImageDraw

FONT = "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf"


def _frame_with_captions(dest: Path, w: int = 640, h: int = 360) -> Path:
    im = Image.new("RGB", (w, h), (24, 24, 28))
    d = ImageDraw.Draw(im)
    # footage texture: soft vertical bands in the LOWER third (ground/pavement)
    for x in range(0, w, 24):
        d.rectangle([x, int(h * 0.75), x + 24, h], fill=(40 + x % 9, 38, 36))
    # burned caption TEXT-STROKE pattern: tall thin white bars inside the
    # bottom quarter — deterministic on every host (no font dependency: the
    # CI runner lacks the Noto font and PIL's fallback font scores 0.009)
    y0, y1 = int(h * 0.84), int(h * 0.95)
    for i in range(18):
        x = 90 + i * 26
        d.rectangle([x, y0, x + 9, y1], fill=(255, 255, 255))
    im.save(dest)
    return dest


def _clean_frame(dest: Path, w: int = 640, h: int = 360) -> Path:
    im = Image.new("RGB", (w, h), (24, 24, 28))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 24):
        d.rectangle([x, int(h * 0.75), x + 24, h], fill=(40 + x % 9, 38, 36))
    im.save(dest)
    return dest


def test_score_high_for_burned_caption_text():
    a = bottom_band_text_score(_frame_with_captions(Path("/tmp/cap_text.png")))
    b = bottom_band_text_score(_clean_frame(Path("/tmp/cap_clean.png")))
    assert a > 0.012, f"caption text score too low: {a}"
    assert b < 0.005, f"clean pavement score too high: {b}"
    assert a - b > 0.01


def test_sample_timestamps_skip_intro_and_tail():
    ts = _sample_timestamps(933.0, 8)
    assert len(ts) == 8
    assert ts[0] >= 933.0 * 0.05 - 1.0
    assert ts[-1] <= 933.0 * 0.95 + 1.0
    assert ts == sorted(ts)


def test_scan_flags_drawtext_video(tmp_path: Path):
    """End-to-end: a 3s synthetic tape WITH drawtext captions flags."""
    font_arg = f"fontfile={FONT}" if Path(FONT).exists() else ""
    if not font_arg:
        return  # no system font in this environment — the PIL test covers the metric
    vf = (
        "drawtext=text='STEP BACK PLEASE':fontcolor=white:fontsize=28:"
        "x=(w-text_w)/2:y=h*0.85:" + font_arg
    )
    video = tmp_path / "capped.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=640x360:d=3:r=10",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    report = scan_burned_captions(video, 3.0, n_samples=4)
    assert report["flagged"], f"median {report['median_score']} must exceed threshold"
    assert report["median_score"] >= report["threshold"]


def test_scan_clean_video_not_flagged(tmp_path: Path):
    video = tmp_path / "clean.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "gradients=s=640x360:d=3:r=10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    report = scan_burned_captions(video, 3.0, n_samples=3)
    assert not report["flagged"], f"clean gradient flagged at {report['median_score']}"
