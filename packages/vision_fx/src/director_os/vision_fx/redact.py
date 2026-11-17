"""Privacy redaction — blur filter chains for faces, plates and corner watermarks.

M1 ships deterministic static-region blur chains for the CPU renderer. Tracked
(YOLO + ByteTrack) regions arrive in M4 and feed the same chain builder with
per-frame boxes; the FFmpeg-side contract does not change.
"""

from __future__ import annotations

from director_os.core.models import BBox, BlurRegion


class RedactError(RuntimeError):
    pass


def _clamp_region(
    region: BlurRegion, frame_w: int, frame_h: int
) -> tuple[int, int, int, int] | None:
    """Clip a region to the frame; returns (x, y, w, h) or None if fully outside."""
    x = max(0, min(region.bbox.x, frame_w - 1))
    y = max(0, min(region.bbox.y, frame_h - 1))
    w = min(region.bbox.w, frame_w - x)
    h = min(region.bbox.h, frame_h - y)
    if w < 4 or h < 4:
        return None
    return x, y, w, h


def scale_region(
    region: BlurRegion, from_w: int, from_h: int, to_w: int, to_h: int
) -> BlurRegion:
    """Map a region from source pixel space to render pixel space (letterbox-aware).

    Studio canvas boxes and detector sweeps are stored in RAW-FOOTAGE pixels,
    but `build_blur_chain` runs AFTER `scale+pad` — i.e. in render space.
    Rendering a 1920x1080 source at 1280x720 without this mapping would place
    every raw-space box wrongly (or clamp it away entirely). Mapping matches
    the geometry chain exactly: `scale=force_original_aspect_ratio=decrease`
    then centered `pad`, so s = min(tw/fw, th/fh) and the remainder is
    letterbox. A no-op when the spaces already match.
    """
    if (from_w, from_h) == (to_w, to_h) or from_w <= 0 or from_h <= 0:
        return region
    s = min(to_w / from_w, to_h / from_h)
    pad_x = (to_w - from_w * s) / 2.0
    pad_y = (to_h - from_h * s) / 2.0
    x = max(0, int(round(region.bbox.x * s + pad_x)))
    y = max(0, int(round(region.bbox.y * s + pad_y)))
    w = max(4, int(round(region.bbox.w * s)))
    h = max(4, int(round(region.bbox.h * s)))
    w = min(w, to_w - x)
    h = min(h, to_h - y)
    if w < 4 or h < 4:
        # degenerate after scale — keep a minimal visible box so redaction intent survives
        w, h = 4, 4
    return region.model_copy(update={"bbox": BBox(x=x, y=y, w=w, h=h)})


def _safe_sigmas(sigma: int, box_w: int, box_h: int) -> tuple[int, int]:
    """boxblur radius must be < half the plane size — luma AND chroma planes differ
    on yuv420p (chroma is half-res), so clamp them independently."""
    luma = max(0, min(sigma, (min(box_w, box_h) - 1) // 2))
    chroma = max(0, min(sigma, (min(box_w // 2, box_h // 2) - 1) // 2))
    return luma, chroma


def build_blur_chain(
    in_label: str,
    out_label: str,
    regions: list[BlurRegion],
    frame_w: int,
    frame_h: int,
) -> str:
    """Build a filtergraph fragment: split -> crop+boxblur -> overlay per region.

    `in_label`/`out_label` include brackets, e.g. `"[v0]"`, `"[vblur]"`.
    """
    usable: list[tuple[int, int, int, int, int, int]] = []
    for r in regions:
        box = _clamp_region(r, frame_w, frame_h)
        if box is None:
            continue
        x, y, w, h = box
        luma, chroma = _safe_sigmas(r.sigma, w, h)
        usable.append((x, y, w, h, luma, chroma))
    if not usable:
        return f"{in_label}null{out_label}"

    n = len(usable)
    split_outs = "".join(f"[sp{i}]" for i in range(n + 1))
    parts = [f"{in_label}split={n + 1}{split_outs}"]
    cur = "[sp0]"
    for i, (x, y, w, h, luma, chroma) in enumerate(usable):
        parts.append(
            f"[sp{i + 1}]crop={w}:{h}:{x}:{y},"
            f"boxblur=luma_radius={luma}:luma_power=2:"
            f"chroma_radius={chroma}:chroma_power=2[rg{i}]"
        )
        out = out_label if i == n - 1 else f"[ov{i}]"
        parts.append(f"{cur}[rg{i}]overlay={x}:{y}{out}")
        cur = f"[ov{i}]"
    return ";".join(parts)


def validate_regions(regions: list[BlurRegion], frame_w: int, frame_h: int) -> int:
    """Return how many regions survive clamping — QA uses this to catch bad coords."""
    return sum(1 for r in regions if _clamp_region(r, frame_w, frame_h) is not None)
