"""LENS & LAW brand watermark — OUR badge on every published frame.

(APPLIES the channel badge; the SOURCE-watermark DETECTOR lives in
watermark.py — different law, different direction.)

The brand pack ships transparent RGBA overlays (four pre-placed 1920x1080
corner frames plus a standalone 1200x400 badge). Publishing law: every
master carries the badge; captions stay bottom-center, so the badge rides a
corner and never collides with the two-row caption block.

Two composition paths:

- SAME-SIZE overlay: master is 1920x1080 -> use the pre-placed corner frame
  directly (``overlay=0:0``), pixel-exact placement by the designer.
- ANY-SIZE overlay: scale the standalone badge to ``width_frac`` of the
  probed master width (absolute pixels, computed before the filter graph so
  the filter only sees numbers) and anchor it in the requested corner with
  proportional margins (shorts, 720p QA proxies, verticals).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CORNER_FRAMES: dict[str, str] = {
    "top_left": "top-left",
    "top_right": "top-right",
    "bottom_left": "bottom-left",
    "bottom_right": "bottom-right",
}

VALID_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")

# anchor expressions reference the SCALED watermark (overlay_w/overlay_h);
# margins arrive as absolute pixels computed from the probed master width
_ANCHOR = {
    "top_left": ("{m}", "{m}"),
    "top_right": ("main_w-overlay_w-{m}", "{m}"),
    "bottom_left": ("{m}", "main_h-overlay_h-{m}"),
    "bottom_right": ("main_w-overlay_w-{m}", "main_h-overlay_h-{m}"),
}


def corner_frame_path(pack_dir: Path, corner: str) -> Path:
    """The pre-placed 1920x1080 corner overlay for ``corner``."""
    if corner not in CORNER_FRAMES:
        raise ValueError(f"unknown corner {corner!r} — expected one of {VALID_CORNERS}")
    return pack_dir / f"lens-and-law-overlay-1920x1080-{CORNER_FRAMES[corner]}.png"


def _margin_expr(margin_frac: float, master_w: int) -> int:
    # proportional margin resolved to absolute pixels BEFORE the filter graph
    return max(8, round(master_w * margin_frac))


def badge_overlay_command(
    master: Path,
    badge: Path,
    out: Path,
    *,
    master_size: tuple[int, int],
    corner: str = "bottom_right",
    width_frac: float = 0.22,
    margin_frac: float = 0.025,
) -> list[str]:
    """ffmpeg argv that scales the badge and anchors it in ``corner``."""
    if corner not in VALID_CORNERS:
        raise ValueError(f"unknown corner {corner!r} — expected one of {VALID_CORNERS}")
    if not 0.05 <= width_frac <= 0.5:
        raise ValueError(f"width_frac {width_frac} outside the sane band 0.05..0.5")
    master_w = int(master_size[0])
    badge_w = max(24, round(master_w * width_frac))
    m = _margin_expr(margin_frac, master_w)
    ax, ay = _ANCHOR[corner]
    x = ax.format(m=m)
    y = ay.format(m=m)
    fc = f"[1:v]scale={badge_w}:-1[wm];[0:v][wm]overlay=x={x}:y={y}[v]"
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(master),
        "-i",
        str(badge),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(out),
    ]


def full_frame_overlay_command(
    master: Path,
    overlay: Path,
    out: Path,
) -> list[str]:
    """ffmpeg argv for a same-size pre-placed corner frame overlay.

    The composite happens in RGBA space (both inputs normalized to rgba)
    before the single yuv420p conversion — compositing directly in yuv
    shifts base-frame colors by the full/limited range delta (~9%).
    """
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(master),
        "-i",
        str(overlay),
        "-filter_complex",
        "[0:v]format=rgba[base];[1:v]format=rgba[wm];[base][wm]overlay=0:0,format=yuv420p[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(out),
    ]


def apply_watermark(
    master: Path,
    badge: Path,
    out: Path,
    *,
    master_size: tuple[int, int] | None = None,
    corner: str = "bottom_right",
    pack_dir: Path | None = None,
) -> Path:
    """Burn the badge into ``master`` -> ``out``; returns the output path.

    1920x1080 masters take the designer's pre-placed corner frame; every
    other geometry scales the standalone badge. The master size is probed
    from the file (the caller's hint is only a cross-check).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    from director_os.core.qa_probe import probe as ffprobe_video

    info = ffprobe_video(master)
    size = (int(info.width or 0), int(info.height or 0))
    if master_size is not None and tuple(master_size) != size:
        raise ValueError(f"master_size hint {master_size} != probed {size}")
    if size == (1920, 1080) and pack_dir is not None:
        frame = corner_frame_path(pack_dir, corner)
        if not frame.exists():
            raise FileNotFoundError(f"corner frame missing: {frame}")
        argv = full_frame_overlay_command(master, frame, out)
    else:
        if not badge.exists():
            raise FileNotFoundError(f"badge missing: {badge}")
        argv = badge_overlay_command(master, badge, out, master_size=size, corner=corner)
    subprocess.run(argv, check=True, capture_output=True)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"watermark overlay produced no output: {out}")
    return out
