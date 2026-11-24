"""Privacy redaction — blur filter chains for faces, plates and corner watermarks.

M1 shipped deterministic static-region blur chains. M4 closes the tracked loop:
Modal GPU (YOLO-face + plate + ByteTrack) produces regions with persistent track
IDs and SOURCE-footage time windows; `clip_region_time` maps those windows into
per-segment local time and `build_blur_chain` gates each overlay with an FFmpeg
`enable=between(t,...)` window — so blur boxes appear, follow and disappear with
their object instead of smearing the whole timeline. The FFmpeg-side contract
(crop+boxblur+overlay) does not change.
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


def scale_bbox(bbox: BBox, from_w: int, from_h: int, to_w: int, to_h: int) -> BBox:
    """Map one box from source pixel space to render pixel space (letterbox-aware).

    Studio canvas boxes and detector sweeps are stored in RAW-FOOTAGE pixels,
    but every renderer effect (blur chains, spotlight overlays) runs AFTER
    `scale+pad` — i.e. in render space. Mapping matches the geometry chain
    exactly: `scale=force_original_aspect_ratio=decrease` then centered `pad`,
    so s = min(tw/fw, th/fh) and the remainder is letterbox. A no-op when the
    spaces already match.
    """
    if (from_w, from_h) == (to_w, to_h) or from_w <= 0 or from_h <= 0:
        return bbox
    s = min(to_w / from_w, to_h / from_h)
    pad_x = (to_w - from_w * s) / 2.0
    pad_y = (to_h - from_h * s) / 2.0
    x = max(0, int(round(bbox.x * s + pad_x)))
    y = max(0, int(round(bbox.y * s + pad_y)))
    w = max(4, int(round(bbox.w * s)))
    h = max(4, int(round(bbox.h * s)))
    w = min(w, to_w - x)
    h = min(h, to_h - y)
    if w < 4 or h < 4:
        # degenerate after scale — keep a minimal visible box so redaction intent survives
        w, h = 4, 4
    return BBox(x=x, y=y, w=w, h=h)


def scale_region(region: BlurRegion, from_w: int, from_h: int, to_w: int, to_h: int) -> BlurRegion:
    """Map a blur region from source pixel space to render pixel space.

    A no-op that returns the ORIGINAL object when the spaces already match.
    """
    if (from_w, from_h) == (to_w, to_h) or from_w <= 0 or from_h <= 0:
        return region
    return region.model_copy(update={"bbox": scale_bbox(region.bbox, from_w, from_h, to_w, to_h)})


def clip_region_time(region: BlurRegion, seg_start_s: float, seg_dur_s: float) -> BlurRegion | None:
    """Translate a region's source-time window into one segment's LOCAL time.

    Region time bounds are absolute source-footage seconds; every segment renders
    a trim of that footage and the filtergraph `t` clock restarts at 0 per
    segment. Returns None when the window does not intersect the segment at all.
    Untimed (static) regions pass through unchanged.
    """
    if region.t_start_s is None and region.t_end_s is None:
        return region
    lo = 0.0 if region.t_start_s is None else region.t_start_s - seg_start_s
    hi = seg_dur_s if region.t_end_s is None else region.t_end_s - seg_start_s
    lo, hi = max(lo, 0.0), min(hi, seg_dur_s)
    if hi - lo < 0.05:  # sub-frame overlap — not worth an overlay node
        return None
    return region.model_copy(update={"t_start_s": round(lo, 3), "t_end_s": round(hi, 3)})


def _safe_sigmas(sigma: int, box_w: int, box_h: int) -> tuple[int, int]:
    """boxblur radius must be < half the plane size — luma AND chroma planes differ
    on yuv420p (chroma is half-res), so clamp them independently.

    The crop plane is EVEN on yuv420p: an odd crop size (43) rounds down to 42
    before the blur sees it, so the clamp derives from the even size — radius
    21 on a 43px region killed the segment render (tape-1 wire fact).
    """
    plane = min(box_w, box_h)
    plane -= plane % 2  # the crop's actual even plane size
    luma = max(0, min(sigma, (plane - 1) // 2))
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
    usable: list[tuple[int, int, int, int, int, int, str]] = []
    for r in regions:
        box = _clamp_region(r, frame_w, frame_h)
        if box is None:
            continue
        x, y, w, h = box
        luma, chroma = _safe_sigmas(r.sigma, w, h)
        enable = ""
        if r.t_start_s is not None and r.t_end_s is not None:
            enable = f":enable='between(t,{r.t_start_s:.2f},{r.t_end_s:.2f})'"
        usable.append((x, y, w, h, luma, chroma, enable))
    if not usable:
        return f"{in_label}null{out_label}"

    n = len(usable)
    split_outs = "".join(f"[sp{i}]" for i in range(n + 1))
    parts = [f"{in_label}split={n + 1}{split_outs}"]
    cur = "[sp0]"
    for i, (x, y, w, h, luma, chroma, enable) in enumerate(usable):
        parts.append(
            f"[sp{i + 1}]crop={w}:{h}:{x}:{y},"
            f"boxblur=luma_radius={luma}:luma_power=2:"
            f"chroma_radius={chroma}:chroma_power=2[rg{i}]"
        )
        out = out_label if i == n - 1 else f"[ov{i}]"
        parts.append(f"{cur}[rg{i}]overlay={x}:{y}{enable}{out}")
        cur = f"[ov{i}]"
    return ";".join(parts)


def validate_regions(regions: list[BlurRegion], frame_w: int, frame_h: int) -> int:
    """Return how many regions survive clamping — QA uses this to catch bad coords."""
    return sum(1 for r in regions if _clamp_region(r, frame_w, frame_h) is not None)
