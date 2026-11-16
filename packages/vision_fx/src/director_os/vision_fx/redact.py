"""Privacy redaction — blur filter chains for faces, plates and corner watermarks.

M1 ships deterministic static-region blur chains for the CPU renderer. Tracked
(YOLO + ByteTrack) regions arrive in M4 and feed the same chain builder with
per-frame boxes; the FFmpeg-side contract does not change.
"""

from __future__ import annotations

from director_os.core.models import BlurRegion


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
