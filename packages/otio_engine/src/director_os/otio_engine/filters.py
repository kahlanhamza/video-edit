"""Filter-complex composer — turns a SegmentRenderSpec into one FFmpeg filtergraph.

Composition order (v2): geometry (scale/pad/sar) -> blur redactions -> zoom
(animated keyframed zoompan, or legacy static punch-in) -> freeze-callout ->
spotlight overlay -> graphic cards (faded, enable-windowed) -> fps lock.

Redaction runs BEFORE the zoom on purpose: blur boxes and spotlight boxes are
authored in render-space pixels of the *un-zoomed* frame, so zooming the
already-redacted picture keeps them glued to the evidence. (The v1 order
zoomed first, which misaligned every blur box on punch-in segments.)

Freeze-callouts need a split/overlay subgraph, so their segments always take
the filter_complex path — the plain `-vf` chain is only for simple segments.
"""

from __future__ import annotations

from director_os.core.models import ZoomKeyframe
from director_os.otio_engine.contract import FreezeCallout
from director_os.otio_engine.segment_cache import SegmentRenderSpec
from director_os.vision_fx.redact import build_blur_chain

ZOOM_PAN_MAX = 3.0


# narration b-roll law (Colab editorial rule): muted AND slightly darkened
NARRATION_DIM = "eq=brightness=-0.06:saturation=0.92"


def base_geometry(w: int, h: int) -> list[str]:
    return [
        f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos",
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ]


def punch_filters(zoom: float, w: int, h: int) -> list[str]:
    """Legacy static punch-in: pre-scale then center-crop (no animation)."""
    if zoom <= 1.0:
        return []
    zw = (int(w * zoom) // 2) * 2
    zh = (int(h * zoom) // 2) * 2
    return [f"scale={zw}:{zh}:flags=lanczos", f"crop={w}:{h}"]


def zoom_expression(points: list[tuple[float, float]], fps: int) -> str:
    """Piecewise-linear zoompan `z` expression over output frame index `on`.

    `points` are (t_seconds, zoom_factor) anchors; the first anchor is the zoom
    at t=0 (usually 1.0). Between anchors the factor interpolates linearly;
    past the last anchor it holds. Rendered as nested `if(lte(on,F),lerp,…)`.
    """
    if not points:
        return "1"
    frames: list[tuple[int, float]] = []
    for t_s, z in points:
        f = max(0, int(round(t_s * fps)))
        if frames and f <= frames[-1][0]:
            # non-advancing anchor — later value wins at the same frame
            frames[-1] = (frames[-1][0], z)
        else:
            frames.append((f, z))
    expr = f"{frames[-1][1]:.4f}"
    for i in range(len(frames) - 1, 0, -1):
        f_prev, z_prev = frames[i - 1]
        f_cur, z_cur = frames[i]
        if f_cur <= f_prev:
            lerp = f"{z_cur:.4f}"
        else:
            lerp = f"{z_prev:.4f}+({z_cur - z_prev:+.4f})*(on-{f_prev})/{f_cur - f_prev}"
        expr = f"if(lte(on,{f_cur}),{lerp},{expr})"
    return expr


def keyframe_points(keyframes: list[ZoomKeyframe]) -> list[tuple[float, float]]:
    """Zoom anchors from the contract: zoom starts at 1.0 and reaches each
    keyframe's factor at its own time, then holds the last value."""
    pts = [(0.0, 1.0)]
    for k in sorted(keyframes, key=lambda k: k.t_s):
        pts.append((k.t_s, k.zoom_factor))
    return pts


def animated_zoom_filters(keyframes: list[ZoomKeyframe], w: int, h: int, fps: int) -> list[str]:
    """Keyframed punch-in: normalize rate, pre-scale to max-zoom headroom, then
    zoompan with a per-frame zoom expression and a centered crop window.

    Pre-scaling to w*max_z keeps the crop window >= output size at every zoom
    level, so no zoom step re-upscales beyond the pre-scale (quality floor).
    """
    if not keyframes:
        return []
    max_z = min(max(k.zoom_factor for k in keyframes), ZOOM_PAN_MAX)
    pw = (int(w * max_z) // 2) * 2
    ph = (int(h * max_z) // 2) * 2
    expr = zoom_expression(keyframe_points(keyframes), fps)
    return [
        f"fps={fps}",
        f"scale={pw}:{ph}:flags=lanczos",
        (f"zoompan=z='{expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}"),
        "setsar=1",
    ]


def zoom_filters(spec: SegmentRenderSpec) -> list[str]:
    """Animated keyframed zoom when keyframes exist, else the static punch-in."""
    if spec.zoom_keyframes:
        return animated_zoom_filters(
            spec.zoom_keyframes, spec.video.w, spec.video.h, spec.video.fps
        )
    return punch_filters(spec.zoom_factor, spec.video.w, spec.video.h)


def freeze_fragment(in_label: str, out_label: str, fc: FreezeCallout, fps: int) -> str:
    """Filtergraph fragment: hold the frame at `freeze_at_s` for `hold_s`.

    Branch B grabs the frame at the freeze point (a one-frame trim window),
    re-bases it to t=0 and clones it far enough to cover the enable window on
    the main branch — the secondary stream carries frames from t=0, which keeps
    the overlay framesync from stalling on a late-starting branch. Audio is
    untouched: narration keeps running under the freeze.
    """
    a = fc.freeze_at_s
    eps = 1.05 / fps
    cover = a + fc.hold_s + 1.0  # clone tail covers the whole enable window
    return ";".join(
        [
            f"{in_label}split=2[fzm][fzs]",
            (
                f"[fzs]trim=start={a:.3f}:end={a + eps:.4f},setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={cover:.3f}[fzh]"
            ),
            (
                f"[fzm][fzh]overlay=eof_action=pass:"
                f"enable='between(t,{a:.3f},{a + fc.hold_s:.3f})'{out_label}"
            ),
        ]
    )


def spotlight_overlay_fragment(
    cur_label: str, out_label: str, input_idx: int, fc: FreezeCallout
) -> str:
    """Full-canvas dim + spotlight PNG overlaid for exactly the freeze window.

    `cur_label`/`out_label` include brackets, e.g. `"[vf]"`, `"[vsp]"`.
    """
    a = fc.freeze_at_s
    b = a + fc.hold_s
    return ";".join(
        [
            f"[{input_idx}:v]format=rgba[spot]",
            f"{cur_label}[spot]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})'{out_label}",
        ]
    )


def plain_video_chain(spec: SegmentRenderSpec) -> str:
    chain = (
        base_geometry(spec.video.w, spec.video.h) + zoom_filters(spec) + list(spec.video_filters)
    )
    if spec.kind == "narration":
        chain = [NARRATION_DIM] + chain  # muted, slightly darkened b-roll
    return ",".join(chain)


def compose_filter_complex(spec: SegmentRenderSpec) -> tuple[str, bool, int]:
    """Return (filter_complex graph, uses_complex, n_overlay_inputs).

    The graph consumes `[0:v]` (master) plus `[1:v]..[n:v]` overlays in order:
    spotlight PNG first when a freeze callout exists, then the card PNGs. It
    yields `[vout]`. When no blur regions, freeze or cards exist, callers
    should use the plain `-vf` chain instead.
    """
    n_overlay_inputs = len(spec.cards) + (1 if spec.spotlight_png_path else 0)

    parts: list[str] = []
    base = base_geometry(spec.video.w, spec.video.h)
    if spec.kind == "narration":
        base = [NARRATION_DIM] + base  # muted, slightly darkened b-roll
    parts.append(f"[0:v]{','.join(base)}[vg]")
    cur = "vg"

    if spec.blur_regions:
        frag = build_blur_chain(
            f"[{cur}]", "[vblur]", spec.blur_regions, spec.video.w, spec.video.h
        )
        parts.append(frag)
        cur = "vblur"

    zoom = zoom_filters(spec)
    if zoom:
        parts.append(f"[{cur}]{','.join(zoom)}[vz]")
        cur = "vz"

    fc = spec.freeze_callout
    if fc is not None:
        frag = freeze_fragment(f"[{cur}]", "[vf]", fc, spec.video.fps)
        parts.append(frag)
        cur = "vf"

    idx = 1
    if spec.spotlight_png_path and fc is not None:
        frag = spotlight_overlay_fragment(f"[{cur}]", "[vsp]", idx, fc)
        parts.append(frag)
        cur = "vsp"
        idx += 1

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
    uses_complex = bool(spec.blur_regions or spec.cards or fc is not None)
    return ";".join(parts), uses_complex, n_overlay_inputs
