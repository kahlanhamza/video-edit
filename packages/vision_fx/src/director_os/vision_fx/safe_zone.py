"""YouTube 16:9 safe-zone geometry — cards and subtitles must clear the player chrome.

YouTube's 16:9 player draws over the video: the title band fades in at the
top on pause/scrub, the progress bar + control gradient sit at the bottom,
and end-screen elements crowd the lower-right. A card that looks perfect in
the studio canvas gets its number buried under the progress bar on a phone.

`SAFE_*` fractions are the conservative editor rule-of-thumb band; every
card placement and subtitle margin runs through `check_safe_zone`, and
`qa.audit_layout_and_policy` gates it as `card_safe_zone` (issue #18).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SAFE_LEFT = 0.06  # clear of pillarbox/edge gradients
SAFE_RIGHT = 0.94
SAFE_TOP = 0.135  # below the title band
SAFE_BOTTOM = 0.88  # above progress bar + control gradient


class SafeZoneRect(BaseModel):
    """The safe area in PIXELS for one frame size."""

    frame_w: int = Field(gt=0)
    frame_h: int = Field(gt=0)
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int = Field(gt=0)
    y1: int = Field(gt=0)


def safe_zone_rect(frame_w: int, frame_h: int) -> SafeZoneRect:
    """Pixel safe rect for the given frame (fractions clamped to bounds)."""
    return SafeZoneRect(
        frame_w=frame_w,
        frame_h=frame_h,
        x0=max(0, int(frame_w * SAFE_LEFT)),
        y0=max(0, int(frame_h * SAFE_TOP)),
        x1=min(frame_w, int(frame_w * SAFE_RIGHT)),
        y1=min(frame_h, int(frame_h * SAFE_BOTTOM)),
    )


def check_safe_zone(
    bbox: tuple[int, int, int, int],
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> tuple[bool, str]:
    """(ok, detail) — is this card/subtitle box fully inside the safe area?"""
    x, y, w, h = bbox
    rect = safe_zone_rect(frame_w, frame_h)
    inside = x >= rect.x0 and y >= rect.y0 and x + w <= rect.x1 and y + h <= rect.y1
    if inside:
        return True, f"box {bbox} inside safe zone {rect.x0},{rect.y0}..{rect.x1},{rect.y1}"
    over = []
    if x < rect.x0:
        over.append(f"left {x - rect.x0}px")
    if y < rect.y0:
        over.append(f"top {y - rect.y0}px")
    if x + w > rect.x1:
        over.append(f"right {x + w - rect.x1}px")
    if y + h > rect.y1:
        over.append(f"bottom {y + h - rect.y1}px")
    return False, f"box {bbox} breaches YouTube chrome: {', '.join(over)}"
