"""Filter-complex composer — turns a SegmentRenderSpec into one FFmpeg filtergraph.

Composition order: geometry (scale/pad/sar) -> punch-in zoom -> custom video filters
-> blur redactions -> graphic cards (faded, enable-windowed) -> fps lock.

Blur chains come from vision_fx.redact; cards from vision_fx.cards.
"""

from __future__ import annotations

from director_os.otio_engine.segment_cache import SegmentRenderSpec
from director_os.vision_fx.redact import build_blur_chain


def base_geometry(w: int, h: int) -> list[str]:
    return [
        f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos",
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ]


def punch_filters(zoom: float, w: int, h: int) -> list[str]:
    if zoom <= 1.0:
        return []
    zw = (int(w * zoom) // 2) * 2
    zh = (int(h * zoom) // 2) * 2
    return [f"scale={zw}:{zh}:flags=lanczos", f"crop={w}:{h}"]


def plain_video_chain(spec: SegmentRenderSpec) -> str:
    return ",".join(
        base_geometry(spec.video.w, spec.video.h)
        + punch_filters(spec.zoom_factor, spec.video.w, spec.video.h)
        + list(spec.video_filters)
    )


def compose_filter_complex(spec: SegmentRenderSpec) -> tuple[str, bool]:
    """Return (filter_complex graph, uses_complex).

    The graph consumes `[0:v]` (master) plus `[1:v]..[n:v]` (card PNGs, in order)
    and yields `[vout]`. When no blur regions or cards exist, callers should use
    the plain `-vf` chain instead.
    """
    chain = base_geometry(spec.video.w, spec.video.h) + punch_filters(
        spec.zoom_factor, spec.video.w, spec.video.h
    )
    parts: list[str] = []
    cur = "v0"
    parts.append(f"[0:v]{','.join(chain)}[v0]")

    idx = 1
    if spec.blur_regions:
        frag = build_blur_chain(
            f"[{cur}]", "[vblur]", spec.blur_regions, spec.video.w, spec.video.h
        )
        parts.append(frag)
        cur = "vblur"

    for i, card in enumerate(spec.cards):
        parts.append(
            f"[{idx}:v]format=rgba,fade=t=in:st={card.t_start_s:.3f}:d={card.fade_s:.3f}:alpha=1"
            f"[card{i}]"
        )
        parts.append(
            f"[{cur}][card{i}]overlay={card.x}:{card.y}:"
            f"enable='between(t,{card.t_start_s:.3f},{card.t_end_s:.3f})'[vc{i}]"
        )
        cur = f"vc{i}"
        idx += 1

    parts.append(f"[{cur}]fps={spec.video.fps}[vout]")
    uses_complex = bool(spec.blur_regions or spec.cards)
    return ";".join(parts), uses_complex
