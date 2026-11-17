"""Freeze-callout spotlight overlay — deterministic Pillow renderer.

`vision_fx.freeze_callout` (PLAN §3.3E): freezes a frame, dims the background
and highlights the evidence region while the narrator explains it. The dim +
spotlight look is one full-canvas RGBA PNG per callout (dark everywhere, a
transparent hole over the evidence box, an accent outline around it), overlaid
by the segment renderer for exactly the freeze window — the same mechanism the
graphic cards use, so the segment cache treats it identically (content digest
in the key, no path leakage).

Determinism matters: identical inputs must produce byte-identical PNGs, or a
recompile would churn cache keys. No fonts, no timestamps, no randomness —
pure geometry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from director_os.core.models import BBox
from PIL import Image, ImageDraw

ACCENT = (255, 64, 44, 255)
OUTLINE_W = 4


class SpotlightError(RuntimeError):
    pass


def clamp_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox | None:
    """Clip a bbox to the frame; returns None if it would degenerate (<4px)."""
    x = max(0, min(bbox.x, frame_w - 1))
    y = max(0, min(bbox.y, frame_h - 1))
    w = min(bbox.w, frame_w - x)
    h = min(bbox.h, frame_h - y)
    if w < 4 or h < 4:
        return None
    return BBox(x=x, y=y, w=w, h=h)


def render_spotlight_overlay(
    frame_w: int,
    frame_h: int,
    spotlight_bbox: BBox | None,
    dim_alpha: float,
    out_png: str | Path,
    outline_w: int = OUTLINE_W,
) -> Path:
    """Render the dim + spotlight overlay at exactly the render frame size.

    `dim_alpha` is how dark the background goes (0.0 = no dim, 0.9 = near
    black); the spotlight box area is fully transparent so the evidence shows
    at full brightness, ringed by the accent outline. A `None` bbox dims the
    whole frame (e.g. while a full-screen document is narrated).
    """
    if frame_w <= 0 or frame_h <= 0:
        raise SpotlightError(f"invalid frame size {frame_w}x{frame_h}")
    if not 0.0 <= dim_alpha <= 0.9:
        raise SpotlightError(f"dim_alpha out of range [0, 0.9]: {dim_alpha}")

    img = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, round(dim_alpha * 255)))
    draw = ImageDraw.Draw(img)

    if spotlight_bbox is not None:
        box = clamp_bbox(spotlight_bbox, frame_w, frame_h)
        if box is not None:
            x1 = box.x + box.w - 1
            y1 = box.y + box.h - 1
            # punch the hole: raw RGBA write clears the dim over the evidence
            draw.rectangle([box.x, box.y, x1, y1], fill=(0, 0, 0, 0))
            draw.rectangle(
                [box.x, box.y, x1, y1],
                outline=ACCENT,
                width=min(outline_w, max(1, min(box.w, box.h) // 2)),
            )

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG")
    return out


def content_digest(png_path: str | Path) -> str:
    """Content hash of the overlay — drives segment-cache invalidation."""
    return hashlib.sha256(Path(png_path).read_bytes()).hexdigest()[:24]
