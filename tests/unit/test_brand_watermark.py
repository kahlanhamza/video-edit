"""LENS & LAW watermark — badge transparency, anchoring, and burn-in."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from director_os.vision_fx.brand_watermark import (
    VALID_CORNERS,
    apply_watermark,
    badge_overlay_command,
    corner_frame_path,
    full_frame_overlay_command,
)
from PIL import Image


# ------------------------------------------------------------ command shape
def test_corner_frame_path_maps_all_corners():
    pack = Path("/tmp/brand")
    for corner in VALID_CORNERS:
        p = corner_frame_path(pack, corner)
        assert f"1920x1080-{corner.replace('_', '-')}" in str(p)


def test_corner_frame_path_rejects_unknown_corner():
    with pytest.raises(ValueError):
        corner_frame_path(Path("/tmp"), "middle_center")


def test_badge_command_scales_and_anchors_bottom_right():
    argv = badge_overlay_command(
        Path("m.mp4"),
        Path("badge.png"),
        Path("o.mp4"),
        master_size=(1920, 1080),
        corner="bottom_right",
    )
    fc = argv[argv.index("-filter_complex") + 1]
    assert "scale=422:-1" in fc  # 1920 * 0.22
    assert "overlay=x=main_w-overlay_w-48" in fc  # 1920 * 0.025
    assert "y=main_h-overlay_h-48" in fc


def test_badge_command_anchors_top_left():
    argv = badge_overlay_command(
        Path("m.mp4"),
        Path("badge.png"),
        Path("o.mp4"),
        master_size=(1280, 720),
        corner="top_left",
    )
    fc = argv[argv.index("-filter_complex") + 1]
    assert "scale=282:-1" in fc  # 1280 * 0.22
    assert "overlay=x=32:y=32" in fc  # 1280 * 0.025, clamped >= 8


def test_badge_command_rejects_bad_geometry():
    with pytest.raises(ValueError):
        badge_overlay_command(
            Path("m.mp4"),
            Path("b.png"),
            Path("o.mp4"),
            master_size=(1920, 1080),
            width_frac=0.9,
        )
    with pytest.raises(ValueError):
        badge_overlay_command(
            Path("m.mp4"),
            Path("b.png"),
            Path("o.mp4"),
            master_size=(1920, 1080),
            corner="bottom_center",
        )


def test_full_frame_overlay_command_maps_both_streams():
    argv = full_frame_overlay_command(Path("m.mp4"), Path("ov.png"), Path("o.mp4"))
    fc = argv[argv.index("-filter_complex") + 1]
    assert "format=rgba[base]" in fc and "format=rgba[wm]" in fc
    assert "[base][wm]overlay=0:0" in fc
    assert argv.count("-i") == 2


# --------------------------------------------------------- real-pixel burn
def _make_master(dest: Path, size: str = "640x360") -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s={size}:d=1:r=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


def _make_badge(dest: Path) -> Path:
    """A transparent badge with an opaque white square core."""
    im = Image.new("RGBA", (120, 40), (0, 0, 0, 0))
    for x in range(40, 80):
        for y in range(10, 30):
            im.putpixel((x, y), (255, 255, 255, 255))
    im.save(dest)
    return dest


def _frame(video: Path) -> Image.Image:
    probe = Path(str(video) + ".png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-frames:v", "1", str(probe)],
        check=True,
        capture_output=True,
    )
    return Image.open(probe).convert("RGB")


def _assert_color(actual: tuple, expected: tuple, tol: int = 14) -> None:
    """yuv420 re-encode + full/limited range rounding shifts pure RGB slightly."""
    for got, want in zip(actual, expected, strict=False):
        assert abs(got - want) <= tol, f"color {actual} too far from {expected}"


def test_apply_watermark_badge_path_renders_visible_change(tmp_path: Path):
    master = _make_master(tmp_path / "m.mp4")
    badge = _make_badge(tmp_path / "badge.png")
    out = apply_watermark(master, badge, tmp_path / "wm.mp4")
    assert out.exists() and out.stat().st_size > 0
    im = _frame(out)
    # badge: 640*0.22=141px wide at x=640-141-16=483; core maps to ~(530..577, 309..332)
    _assert_color(im.getpixel((553, 320)), (255, 255, 255))
    _assert_color(im.getpixel((100, 100)), (255, 0, 0))


def test_apply_watermark_missing_badge_raises(tmp_path: Path):
    master = _make_master(tmp_path / "m.mp4")
    with pytest.raises(FileNotFoundError):
        apply_watermark(master, tmp_path / "missing.png", tmp_path / "o.mp4")


def test_apply_watermark_rejects_wrong_size_hint(tmp_path: Path):
    master = _make_master(tmp_path / "m.mp4")
    badge = _make_badge(tmp_path / "badge.png")
    with pytest.raises(ValueError):
        apply_watermark(master, badge, tmp_path / "o.mp4", master_size=(1280, 720))


def test_apply_watermark_1920_master_uses_pack_frame(tmp_path: Path):
    master = _make_master(tmp_path / "m.mp4", size="1920x1080")
    pack = tmp_path / "pack"
    pack.mkdir()
    # a pre-placed frame that is fully transparent EXCEPT an 8x8 green patch
    # (a single pixel would be blurred by yuv420 chroma subsampling)
    frame = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    for x in range(956, 964):
        for y in range(536, 544):
            frame.putpixel((x, y), (0, 255, 0, 255))
    frame.save(pack / "lens-and-law-overlay-1920x1080-bottom-right.png")
    out = apply_watermark(
        master,
        tmp_path / "badge.png",
        tmp_path / "o.mp4",
        master_size=(1920, 1080),
        pack_dir=pack,
    )
    assert out.exists()
    im = _frame(out)
    _assert_color(im.getpixel((960, 540)), (0, 255, 0))
    _assert_color(im.getpixel((100, 100)), (255, 0, 0))


def test_apply_watermark_missing_pack_frame_raises(tmp_path: Path):
    master = _make_master(tmp_path / "m.mp4", size="1920x1080")
    with pytest.raises(FileNotFoundError):
        apply_watermark(
            master,
            tmp_path / "b.png",
            tmp_path / "o.mp4",
            master_size=(1920, 1080),
            pack_dir=tmp_path,
        )
