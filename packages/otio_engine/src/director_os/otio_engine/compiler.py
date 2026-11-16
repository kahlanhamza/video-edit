"""Timeline compiler — OTIO in, RenderDAG out (PLAN §3.3C/D + §7 Stage 9).

Walks `V1_master` clip-by-clip, honors the §5.2 metadata contract, maps overlay
cards onto the segments that contain them (collision-free corner placement),
enforces uniform codec/audio-law params across the DAG, and hashes every segment
for the cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from director_os.core.ffmpeg_args import loudnorm_chain
from director_os.core.models import BlurRegion
from director_os.otio_engine.contract import (
    CardMeta,
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


class TimelineCompileError(RuntimeError):
    pass


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


def compile_timeline(
    timeline,
    assets: Mapping[str, str | Path],
    cache: SegmentCache,
    video: VideoRenderSpec | None = None,
    audio: AudioRenderSpec | None = None,
    out_dir: str | Path = "./output/render",
    audio_target_lufs: float = -16.0,
    global_blur_regions: list[BlurRegion] | None = None,
) -> RenderDAG:
    video = video or VideoRenderSpec()
    audio = audio or AudioRenderSpec()
    out_dir = Path(out_dir)
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
            src = assets.get(meta.asset_id)
            if src is None:
                raise TimelineCompileError(f"unknown asset_id '{meta.asset_id}' in master track")
            zoom = max((k.zoom_factor for k in meta.zoom_keyframes), default=1.0)
            audio_filters = (
                [loudnorm_chain(audio.target_lufs)] if meta.audio_mode == "loudnorm_48k" else []
            )
            spec = SegmentRenderSpec(
                kind="raw",
                asset_id=meta.asset_id,
                source_start_s=start_s,
                source_dur_s=dur,
                video=video,
                audio=audio,
                audio_filters=audio_filters,
                blur_regions=[*global_blur, *meta.blur_regions],
                zoom_factor=zoom,
                codec_profile=f"{video.codec}-{video.pix_fmt}-{video.fps}fps-{audio.sample_rate}hz",
            )
        elif isinstance(meta, NarratorClipMeta):
            src = assets.get(meta.broll_asset_id) if meta.broll_asset_id else None
            if meta.broll_asset_id and src is None:
                raise TimelineCompileError(f"unknown broll asset_id '{meta.broll_asset_id}'")
            spec = SegmentRenderSpec(
                kind="narration",
                asset_id=meta.broll_asset_id or "",
                source_start_s=start_s,
                source_dur_s=dur,
                video=video,
                audio=audio,
                audio_filters=[],
                narration_text=meta.script_text,
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
                dur_s=dur,
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


def _png_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size
