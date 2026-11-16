"""Glass graphic cards — deterministic Pillow renderer + collision-free placement.

Templates (PLAN §3.3F `graphics.render_card`): `location_intro`, `case_outcome`,
`evidence_quote`, and `map_route` (M4 — Playwright HTML/CSS pipeline with a
deterministic Pillow fallback, see `vision_fx.map_route`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from director_os.core.fonts import pick_font
from PIL import Image, ImageDraw

CANVAS_W, CANVAS_H = 640, 180
CORNER_MARGIN = 24
ACCENT = (200, 30, 30, 255)
GLASS_BG = (16, 20, 28, 205)
TITLE_COLOR = (240, 240, 240, 255)
BODY_COLOR = (170, 176, 188, 255)

TEMPLATES = ("location_intro", "case_outcome", "evidence_quote", "map_route")


class CardError(RuntimeError):
    pass


def _rgba_canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font_path: Path, max_size: int, max_w: int):
    size = max_size
    while size > 14:
        font = _font(font_path, size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 2
    return _font(font_path, 14)


_FONT_CACHE: dict[tuple[str, int], object] = {}


def _font(path: Path, size: int):
    key = (str(path), size)
    if key not in _FONT_CACHE:
        from PIL import ImageFont

        _FONT_CACHE[key] = ImageFont.truetype(str(path), size)
    return _FONT_CACHE[key]


def render_card(
    template: str,
    fields: dict[str, Any],
    out_png: str | Path,
    size: tuple[int, int] = (CANVAS_W, CANVAS_H),
) -> Path:
    """Render an anti-aliased RGBA glass card; returns the PNG path."""
    if template == "map_route":
        from director_os.vision_fx.map_route import render_map_route

        return render_map_route(fields, out_png, size=size)
    if template not in TEMPLATES:
        raise CardError(f"unknown card template: {template} (known: {TEMPLATES})")

    w, h = size
    img, draw = _rgba_canvas(w, h)
    font_path = pick_font("DejaVu Sans")

    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=18, fill=GLASS_BG)
    draw.rounded_rectangle([0, 0, 10, h - 1], radius=4, fill=ACCENT)

    title = fields.get("title", "")
    body = fields.get("body", fields.get("subtitle", ""))
    badge = fields.get("badge", "")
    quote = fields.get("quote", "")
    attribution = fields.get("attribution", "")

    if template == "location_intro":
        f_title = _fit_text(draw, title, font_path, 44, w - 110)
        f_body = _fit_text(draw, body, font_path, 24, w - 110)
        draw.text((56, 38), title, font=f_title, fill=TITLE_COLOR)
        draw.text((56, 108), body, font=f_body, fill=BODY_COLOR)
    elif template == "case_outcome":
        f_badge = _font(font_path, 20)
        f_title = _fit_text(draw, title, font_path, 40, w - 230)
        f_body = _fit_text(draw, body, font_path, 22, w - 110)
        draw.rounded_rectangle([w - 200, 30, w - 36, 68], radius=14, fill=ACCENT)
        draw.text((w - 194, 38), badge or "OUTCOME", font=f_badge, fill=TITLE_COLOR)
        draw.text((56, 34), title, font=f_title, fill=TITLE_COLOR)
        draw.text((56, 106), body, font=f_body, fill=BODY_COLOR)
    else:  # evidence_quote
        f_quote = _fit_text(draw, f'"{quote}"', font_path, 36, w - 110)
        f_attr = _fit_text(draw, f"— {attribution}", font_path, 22, w - 110)
        draw.text((56, 34), f'"{quote}"', font=f_quote, fill=TITLE_COLOR)
        draw.text((56, 120), f"— {attribution}", font=f_attr, fill=BODY_COLOR)

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def content_digest(png_path: str | Path) -> str:
    """Content hash of a rendered card — drives segment-cache invalidation."""
    return hashlib.sha256(Path(png_path).read_bytes()).hexdigest()[:24]


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def pick_corner(
    canvas_w: int,
    canvas_h: int,
    box_w: int,
    box_h: int,
    occupied: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int]:
    """First collision-free corner (TL, TR, BL, BR); stacks downward when full."""
    occupied = occupied or []
    m = CORNER_MARGIN
    candidates = [
        (m, m),  # top-left
        (canvas_w - box_w - m, m),  # top-right
        (m, canvas_h - box_h - m),  # bottom-left
        (canvas_w - box_w - m, canvas_h - box_h - m),  # bottom-right
    ]
    for pos in candidates:
        if all(not _intersects((pos[0], pos[1], box_w, box_h), o) for o in occupied):
            return pos
    # corners exhausted: stack under the top-right column
    lowest = max((o[1] + o[3] for o in occupied), default=m)
    return (canvas_w - box_w - m, min(lowest + 12, canvas_h - box_h))
