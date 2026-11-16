"""map_route card — Playwright HTML/CSS -> PNG (PLAN §3.3F, M4).

The route card is data-driven: normalized route points render as an inline SVG
polyline inside a dark glass card, so any street geometry can drop in.

Two backends, same output contract:
- `playwright` (preferred): real HTML/CSS via headless Chromium — full design
  freedom, the M4 animation base. Optional dependency (`pip install playwright`
  + `playwright install chromium`); imported lazily so the engine never
  hard-requires it.
- `pillow` (fallback): deterministic schematic with identical layout — keeps
  the pipeline renderable everywhere (CI, air-gapped machines).

Both paths are pinned by tests; `content_digest` (cards.py) keys the segment
cache either way.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from director_os.core.fonts import pick_font
from director_os.vision_fx.cards import (
    ACCENT,
    BODY_COLOR,
    GLASS_BG,
    TITLE_COLOR,
    CardError,
    _fit_text,
)
from PIL import Image, ImageDraw

MAX_ROUTE_POINTS = 12


def render_map_route(
    fields: dict,
    out_png: str | Path,
    size: tuple[int, int] = (640, 180),
    backend: str = "auto",
) -> Path:
    """Render the map_route glass card; returns the PNG path.

    fields: title, subtitle, route_points [[x,y] normalized 0..1], optionally
    distance_label + duration_label.
    """
    points = _validate_route(fields.get("route_points"))
    if backend == "auto":
        backend = "playwright" if _playwright_available() else "pillow"
    if backend == "playwright":
        try:
            return _render_playwright(fields, points, out_png, size)
        except Exception as e:  # broken browser install -> deterministic fallback
            _ = e
    return _render_pillow(fields, points, out_png, size)


def _validate_route(raw) -> list[tuple[float, float]]:
    if not raw or not isinstance(raw, list) or len(raw) < 2:
        raise CardError("map_route needs route_points: >= 2 [x, y] pairs (normalized 0..1)")
    if len(raw) > MAX_ROUTE_POINTS:
        raise CardError(f"map_route caps route_points at {MAX_ROUTE_POINTS} (got {len(raw)})")
    pts: list[tuple[float, float]] = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise CardError(f"route point {p!r} is not [x, y]")
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise CardError(f"route point {p!r} outside normalized 0..1")
        pts.append((x, y))
    return pts


def _playwright_available() -> bool:
    return importlib.util.find_spec("playwright") is not None


# ------------------------------------------------------------------ html path
def _route_svg(points: list[tuple[float, float]], w: int, h: int) -> str:
    """Inline SVG: padded viewport, route polyline, start/end dots."""
    pad = 34
    inner_w, inner_h = w - 2 * pad - 120, h - 2 * pad  # leave room for labels
    pts_svg = " ".join(f"{pad + x * inner_w:.0f},{pad + y * inner_h:.0f}" for x, y in points)
    x0, y0 = points[0]
    xn, yn = points[-1]
    dot = lambda x, y, c: (  # noqa: E731
        f'<circle cx="{pad + x * inner_w:.0f}" cy="{pad + y * inner_h:.0f}" r="6" fill="{c}" '
        'stroke="#f5f5f5" stroke-width="2"/>'
    )
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{pts_svg}" fill="none" stroke="{_hex(ACCENT)}" '
        'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>'
        f"{dot(x0, y0, '#34d399')}{dot(xn, yn, _hex(ACCENT))}"
        "</svg>"
    )


def _hex(rgb: tuple[int, int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _html(fields: dict, points: list[tuple[float, float]], size: tuple[int, int]) -> str:
    w, h = size
    labels = " · ".join(
        v for v in (fields.get("distance_label"), fields.get("duration_label")) if v
    )
    return f"""<!doctype html><html><head><style>
      * {{ margin: 0; padding: 0; }}
      body {{ width: {w}px; height: {h}px; font-family: 'DejaVu Sans', sans-serif; }}
      .glass {{ width: 100%; height: 100%; background: rgba(16,20,28,0.80);
                border-radius: 18px; border-left: 12px solid {_hex(ACCENT)};
                display: flex; position: relative; overflow: hidden; }}
      .text {{ padding: 22px 0 0 56px; z-index: 2; }}
      .title {{ color: {_hex(TITLE_COLOR)}; font-size: 34px; font-weight: 700; }}
      .sub   {{ color: {_hex(BODY_COLOR)}; font-size: 18px; margin-top: 8px; }}
      .badge {{ color: {_hex(TITLE_COLOR)}; font-size: 15px; margin-top: 10px;
                opacity: 0.85; }}
      svg {{ position: absolute; inset: 0; z-index: 1; opacity: 0.95; }}
    </style></head><body>
      <div class="glass">
        <div class="text">
          <div class="title">{fields.get("title", "")}</div>
          <div class="sub">{fields.get("subtitle", "")}</div>
          <div class="badge">{labels}</div>
        </div>
        {_route_svg(points, w, h)}
      </div>
    </body></html>"""


def _render_playwright(fields, points, out_png, size) -> Path:
    from playwright.sync_api import sync_playwright

    w, h = size
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": w, "height": h})
        page.set_content(_html(fields, points, size))
        # transparent page background matches the Pillow path's alpha corners
        page.screenshot(path=str(out_png), omit_background=True)
        browser.close()
    return Path(out_png)


# ---------------------------------------------------------------- pillow path
def _render_pillow(fields, points, out_png, size) -> Path:
    w, h = size
    img, draw = Image.new("RGBA", (w, h), (0, 0, 0, 0)), None
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=18, fill=GLASS_BG)
    draw.rounded_rectangle([0, 0, 10, h - 1], radius=4, fill=ACCENT)

    font_path = pick_font("DejaVu Sans")
    title = fields.get("title", "")
    sub = fields.get("subtitle", "")
    labels = " · ".join(
        v for v in (fields.get("distance_label"), fields.get("duration_label")) if v
    )
    if title:
        draw.text(
            (56, 22), title, font=_fit_text(draw, title, font_path, 34, w - 130), fill=TITLE_COLOR
        )
    if sub:
        draw.text((56, 68), sub, font=_fit_text(draw, sub, font_path, 18, w - 130), fill=BODY_COLOR)
    if labels:
        draw.text(
            (56, 100), labels, font=_fit_text(draw, labels, font_path, 15, w - 130), fill=BODY_COLOR
        )

    # schematic route: pad + scale normalized points, start green / end red
    pad, label_w = 34, 120
    inner_w, inner_h = w - 2 * pad - label_w, h - 2 * pad
    scaled = [(pad + x * inner_w, pad + y * inner_h) for x, y in points]
    draw.line(scaled, fill=ACCENT, width=5, joint="curve")
    for (px, py), color in ((scaled[0], (52, 211, 153, 255)), (scaled[-1], ACCENT)):
        draw.ellipse(
            [px - 6, py - 6, px + 6, py + 6], fill=color, outline=(245, 245, 245, 255), width=2
        )

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
