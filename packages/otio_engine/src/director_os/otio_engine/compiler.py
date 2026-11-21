"""Timeline compiler — OTIO in, RenderDAG out (PLAN §3.3C/D + §7 Stage 9).

Walks `V1_master` clip-by-clip, honors the §5.2 metadata contract, maps overlay
cards onto the segments that contain them (collision-free corner placement),
enforces uniform codec/audio-law params across the DAG, and hashes every segment
for the cache.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from director_os.core.ffmpeg_args import loudnorm_chain
from director_os.core.models import BlurRegion, ZoomKeyframe
from director_os.otio_engine.contract import (
    CardMeta,
    FreezeCallout,
    NarratorClipMeta,
    RawClipMeta,
    parse_clip_meta,
)
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    CardOverlaySpec,
    SegmentCache,
    SegmentRenderSpec,
    VideoRenderSpec,
    segment_hash,
)
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    OVERLAY_TRACK,
    is_gap,
    item_duration_s,
    track_by_name,
)
from director_os.vision_fx.cards import pick_corner, render_card
from director_os.vision_fx.redact import clip_region_time, scale_bbox, scale_region
from director_os.vision_fx.spotlight import render_spotlight_overlay


class TimelineCompileError(RuntimeError):
    pass


def render_duration_s(dur_s: float, fps: float | int) -> float:
    """Frame-truth duration: what ffmpeg actually produces for a slice.

    ffmpeg emits frames while PTS < end, so a 7.9845 s slice @30 fps renders
    ceil(239.535) = 240 frames = 8.000 s. Expectations (cache admission,
    QA drift, DAG total) must ride the SAME quantization or a long concat
    accumulates half-frame rounding into multi-second drift (caught live on
    the 2627 s Eustis master: 4.92 s over 329 segments). Source slices stay
    exact — only the expectation is quantized to the render grid.
    """
    return math.ceil(round(dur_s * fps, 6)) / float(fps)


@dataclass
class SegmentRender:
    key: str
    spec: SegmentRenderSpec
    src_path: str | None
    out_path: Path
    dur_s: float


@dataclass
class RenderDAG:
    segments: list[SegmentRender]
    final_path: Path
    total_dur_s: float
    warnings: list[str] = field(default_factory=list)


def master_track(timeline):
    track = track_by_name(timeline, MASTER_TRACK) or next(
        (t for t in timeline.tracks if t.kind == "Video"), None
    )
    if track is None or not list(track):
        raise TimelineCompileError("timeline has no non-empty master video track")
    return track


def validate_uniformity(dag: RenderDAG) -> None:
    """Every segment must share identical codec/audio-law params for `-c copy` concat."""
    if not dag.segments:
        raise TimelineCompileError("empty DAG")
    ref = dag.segments[0].spec
    for seg in dag.segments[1:]:
        if seg.spec.video != ref.video or seg.spec.audio != ref.audio:
            raise TimelineCompileError(
                "non-uniform segment params would break `concat -c copy` "
                "(codec profile / fps / sample-rate mismatch)"
            )


def _resolve_freeze(
    fc: FreezeCallout,
    dur_s: float,
    raw_geo: tuple[int, int] | None,
    render_w: int,
    render_h: int,
) -> FreezeCallout | None:
    """Validate + adapt an authored freeze callout for one segment.

    Clamps the hold to the segment tail (dropping callouts that start beyond
    it) and maps the spotlight bbox from RAW-FOOTAGE pixels into render space
    with the same letterbox-aware geometry the blur regions use.
    """
    if fc.freeze_at_s >= dur_s - 0.1:
        return None
    hold = min(fc.hold_s, dur_s - fc.freeze_at_s)
    box = fc.spotlight_bbox
    if box is not None and raw_geo is not None and tuple(raw_geo) != (render_w, render_h):
        box = scale_bbox(box, raw_geo[0], raw_geo[1], render_w, render_h)
    return fc.model_copy(update={"hold_s": hold, "spotlight_bbox": box})


def compile_timeline(
    timeline,
    assets: Mapping[str, str | Path],
    cache: SegmentCache,
    video: VideoRenderSpec | None = None,
    audio: AudioRenderSpec | None = None,
    out_dir: str | Path = "./output/render",
    audio_target_lufs: float = -16.0,
    global_blur_regions: list[BlurRegion] | None = None,
    asset_geometry: Mapping[str, tuple[int, int]] | None = None,
    enhanced_audio: Mapping[str, str | Path] | None = None,
    clip_sources: Mapping[str, str | Path] | None = None,
    clip_source_digests: Mapping[str, str] | None = None,
    narration_audio: Mapping[str, str | Path] | None = None,
) -> RenderDAG:
    """Compile the master track into renderable segments.

    `enhanced_audio` maps asset_id -> the 2-bus master WAV produced by the
    audio-enhance pre-pass. Raw clips authored with `audio_mode="isolate_2bus"`
    REQUIRE their entry (fail loudly — a missing master WAV must never
    silently downgrade a broadcast mix back to noisy raw audio).

    `clip_sources` overrides the source file PER CLIP, keyed
    "asset_id@start+dur" (e.g. "a1@12.000+8.000") — this is how a tracked
    redaction plan is baked per KEPT clip (Pass 3) instead of over the whole
    raw file: cut-out footage is never re-encoded just to have its blur
    thrown away. `clip_source_digests` carries the derived source digest
    (plan digest + range) that rides the segment cache key.

    `narration_audio` maps a NarratorClipMeta script_text -> the persisted
    TTS clip (stage_narration, content-addressed) so the voiceover rides the
    render without re-synthesizing at render time.
    """
    video = video or VideoRenderSpec()
    audio = audio or AudioRenderSpec()
    out_dir = Path(out_dir).resolve()  # absolute: concat lists + PNG paths must resolve anywhere
    cards_dir = out_dir / "cards"

    master = master_track(timeline)
    overlay = track_by_name(timeline, OVERLAY_TRACK)
    # project-level redactions (Director Studio blur canvas) ride on every raw segment
    global_blur = list(global_blur_regions or [])

    segments: list[SegmentRender] = []
    t_master = 0.0
    for item in master:
        dur = item_duration_s(item)
        if dur <= 0:
            continue
        if is_gap(item):
            t_master += dur
            continue
        meta = parse_clip_meta(item)
        start_s = item.source_range.start_time.to_seconds() if item.source_range else 0.0

        if isinstance(meta, RawClipMeta):
            clip_key = f"{meta.asset_id}@{start_s:.3f}+{dur:.3f}"
            src = (clip_sources or {}).get(clip_key) or assets.get(meta.asset_id)
            if src is None:
                raise TimelineCompileError(f"unknown asset_id '{meta.asset_id}' in master track")
            zoom_keyframes: list[ZoomKeyframe] = list(meta.zoom_keyframes)
            zoom = max((k.zoom_factor for k in zoom_keyframes), default=1.0)
            enhanced_path: str | None = None
            enhanced_digest: str | None = None
            if meta.audio_mode == "isolate_2bus":
                wav = (enhanced_audio or {}).get(meta.asset_id)
                if wav is None:
                    raise TimelineCompileError(
                        f"asset '{meta.asset_id}' is audio_mode=isolate_2bus but no 2-bus "
                        "master WAV was provided (run stage_audio_enhance first — refusing "
                        "to silently render noisy raw audio)"
                    )
                enhanced_path = str(Path(wav).resolve())
                enhanced_digest = _file_digest_sha256(Path(wav))[:24]
            audio_filters = (
                [loudnorm_chain(audio.target_lufs)] if meta.audio_mode == "loudnorm_48k" else []
            )
            blur = [*global_blur, *meta.blur_regions]
            raw_geo = (asset_geometry or {}).get(meta.asset_id)
            if raw_geo is not None and tuple(raw_geo) != (video.w, video.h):
                # blur boxes are authored in RAW-FOOTAGE pixels (studio canvas,
                # detector sweep) — the renderer blurs AFTER scale+pad, so map
                # them into render space or they land in the wrong place
                blur = [scale_region(r, raw_geo[0], raw_geo[1], video.w, video.h) for r in blur]
            # tracked regions carry SOURCE-footage time windows: translate them
            # into this segment's local clock (or drop segments they miss) so
            # the blur only appears while the tracked object is actually on screen
            blur = [
                clipped
                for clipped in (clip_region_time(r, start_s, dur) for r in blur)
                if clipped is not None
            ]
            freeze = None
            spotlight_png = None
            if meta.freeze_callout is not None:
                freeze = _resolve_freeze(meta.freeze_callout, dur, raw_geo, video.w, video.h)
                if freeze is not None:
                    spotlight_png = cards_dir / f"spotlight_{int(t_master * 1000)}.png"
                    render_spotlight_overlay(
                        video.w, video.h, freeze.spotlight_bbox, freeze.dim_alpha, spotlight_png
                    )
            spec = SegmentRenderSpec(
                kind="raw",
                asset_id=meta.asset_id,
                source_start_s=start_s,
                source_dur_s=dur,
                video=video,
                audio=audio,
                audio_filters=audio_filters,
                blur_regions=blur,
                zoom_factor=zoom,
                zoom_keyframes=zoom_keyframes,
                freeze_callout=freeze,
                spotlight_png_path=str(spotlight_png) if spotlight_png else None,
                enhanced_audio_path=enhanced_path,
                enhanced_audio_digest=enhanced_digest,
                source_digest=(clip_source_digests or {}).get(clip_key),
                codec_profile=f"{video.codec}-{video.pix_fmt}-{video.fps}fps-{audio.sample_rate}hz",
            )
        elif isinstance(meta, NarratorClipMeta):
            src = assets.get(meta.broll_asset_id) if meta.broll_asset_id else None
            if meta.broll_asset_id and src is None:
                raise TimelineCompileError(f"unknown broll asset_id '{meta.broll_asset_id}'")
            narr_path: str | None = None
            narr_digest: str | None = None
            narr_src = (narration_audio or {}).get(meta.script_text)
            if narr_src is not None:
                narr_path = str(Path(narr_src).resolve())
                narr_digest = _file_digest_sha256(Path(narr_src))[:24]
            spec = SegmentRenderSpec(
                kind="narration",
                asset_id=meta.broll_asset_id or "",
                source_start_s=start_s,
                source_dur_s=dur,
                video=video,
                audio=audio,
                audio_filters=[],
                narration_text=meta.script_text,
                narration_audio_path=narr_path,
                narration_audio_digest=narr_digest,
                broll_windows=(
                    [(float(w0), float(w1)) for w0, w1 in meta.broll_windows]
                    if meta.broll_windows
                    else None
                ),
                codec_profile=f"{video.codec}-{video.pix_fmt}-{video.fps}fps-{audio.sample_rate}hz",
            )
        else:  # pragma: no cover — master track must be raw/narration only
            raise TimelineCompileError(
                f"master track cannot carry item kind '{type(meta).__name__}'"
            )

        key = segment_hash(spec)
        segments.append(
            SegmentRender(
                key=key,
                spec=spec,
                src_path=str(src) if src else None,
                out_path=cache.path_for(key),
                dur_s=render_duration_s(dur, video.fps),
            )
        )
        t_master += dur

    # ---- overlay cards -> absolute position -> owning segment
    t_overlay = 0.0
    if overlay is not None:
        for item in overlay:
            dur = item_duration_s(item)
            if dur <= 0:
                continue
            if is_gap(item):
                t_overlay += dur
                continue
            meta = parse_clip_meta(item)
            if isinstance(meta, CardMeta):
                seg = next(
                    (
                        s
                        for s in segments
                        if s.spec is not None
                        and t_overlay >= _seg_t0(segments, s) - 1e-6
                        and t_overlay < _seg_t0(segments, s) + s.dur_s
                    ),
                    None,
                )
                if seg is None:
                    # card falls outside the master cut — warn and drop (M2: extend cut)
                    continue
                seg_t0 = _seg_t0(segments, seg)
                local_start = max(0.0, t_overlay - seg_t0)
                local_end = min(t_overlay + dur, seg_t0 + seg.dur_s) - seg_t0
                if local_end - local_start < 0.1:
                    continue
                png = render_card(
                    meta.template,
                    {k: str(v) for k, v in meta.fields.items()},
                    cards_dir / f"{meta.template}_{int(t_overlay * 1000)}.png",
                    size=(min(640, int(seg.spec.video.w * 0.5)), 180),
                )
                occupied = [
                    (c.x, c.y, _png_size(Path(c.png_path))[0], _png_size(Path(c.png_path))[1])
                    for c in seg.spec.cards
                ]
                cw, ch = _png_size(png)
                x, y = pick_corner(seg.spec.video.w, seg.spec.video.h, cw, ch, occupied)
                seg.spec.cards.append(
                    CardOverlaySpec(
                        png_path=str(png),
                        x=x,
                        y=y,
                        t_start_s=round(local_start, 3),
                        t_end_s=round(local_end, 3),
                        fade_s=meta.fade_s,
                    )
                )
                # card placement + content changed the spec: re-hash
                seg.key = segment_hash(seg.spec)
                seg.out_path = cache.path_for(seg.key)
            t_overlay += dur

    final_path = Path(out_dir) / "final.mp4"
    dag = RenderDAG(
        segments=segments,
        final_path=final_path,
        total_dur_s=sum(s.dur_s for s in segments),
    )
    validate_uniformity(dag)
    return dag


def _seg_t0(segments: list[SegmentRender], target: SegmentRender) -> float:
    t = 0.0
    for s in segments:
        if s is target:
            return t
        t += s.dur_s
    return t


def _file_digest_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size
