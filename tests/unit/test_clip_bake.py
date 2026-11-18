"""Per-clip baked redaction sources — compiler contract + clip-plan reindexing."""

from __future__ import annotations

import pytest
from director_os.core.models import BBox, BlurRegion
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import CardMeta, RawClipMeta
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    SegmentCache,
    SegmentRenderSpec,
    VideoRenderSpec,
    segment_hash,
)
from director_os.otio_engine.timeline import ClipSpec, TimelineSpec, TrackSpec, build_timeline
from director_os.vision_fx.trajectory import (
    PerFrameRedaction,
    RedactionTrajectory,
    TrackedBox,
)
from director_os.worker.pipeline import _clip_plan


def _plan(n_frames: int = 300, fps: int = 30, w: int = 640, h: int = 480) -> PerFrameRedaction:
    boxes = [
        TrackedBox(frame_idx=i, t_s=round(i / fps, 3), x=10, y=10, w=40, h=40)
        for i in range(60, 200)  # 2.0s .. 6.67s
    ]
    return PerFrameRedaction(
        fps=fps,
        n_frames=n_frames,
        frame_w=w,
        frame_h=h,
        trajectories=[RedactionTrajectory(track_id="person_1", boxes=boxes)],
    )


def test_clip_plan_reindexes_frames():
    plan = _plan()
    clip = _clip_plan(plan, 120 / 30, 60 / 30)  # 4.0s .. 6.0s
    assert clip.n_frames == 60
    assert clip.trajectories[0].track_id == "person_1"
    first = clip.trajectories[0].boxes[0]
    assert first.frame_idx == 0  # source frame 120 -> clip frame 0
    assert first.t_s == pytest.approx(0.0, abs=1e-3)
    assert clip.boxes_at(0) and clip.boxes_at(59)
    assert not clip.boxes_at(59 + 1)


def test_clip_plan_clips_outside_range_drop():
    plan = _plan()
    clip = _clip_plan(plan, 0.0, 1.0)  # frames 0..30 — before the track starts
    assert clip.trajectories == []


def _timeline(asset_id: str, ranges: list[tuple[float, float]]) -> object:
    items = [
        ClipSpec(
            name=f"raw_{i}",
            source_start_s=s,
            dur_s=d,
            meta=RawClipMeta(asset_id=asset_id),
        )
        for i, (s, d) in enumerate(ranges)
    ]
    spec = TimelineSpec(
        name="t",
        fps=30,
        width=320,
        height=240,
        tracks=[TrackSpec(name="V1_master", items=items)],
    )
    return build_timeline(spec)


def test_compile_uses_per_clip_sources(tmp_path):
    timeline = _timeline("a1", [(0.0, 2.0), (4.0, 2.0)])
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"not-a-real-video")
    baked_a = tmp_path / "baked_a.mp4"
    baked_b = tmp_path / "baked_b.mp4"
    baked_a.write_bytes(b"baked-a")
    baked_b.write_bytes(b"baked-b")
    cache = SegmentCache(tmp_path / "cache")
    dag = compile_timeline(
        timeline,
        assets={"a1": str(raw)},
        cache=cache,
        out_dir=tmp_path / "render",
        clip_sources={
            "a1@0.000+2.000": str(baked_a),
            "a1@4.000+2.000": str(baked_b),
        },
        clip_source_digests={
            "a1@0.000+2.000": "d1@0.000+2.000",
            "a1@4.000+2.000": "d1@4.000+2.000",
        },
    )
    assert [seg.src_path for seg in dag.segments] == [str(baked_a), str(baked_b)]
    assert [seg.spec.source_digest for seg in dag.segments] == ["d1@0.000+2.000", "d1@4.000+2.000"]


def test_compile_falls_back_to_asset_source(tmp_path):
    timeline = _timeline("a1", [(0.0, 2.0)])
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    cache = SegmentCache(tmp_path / "cache")
    dag = compile_timeline(
        timeline, assets={"a1": str(raw)}, cache=cache, out_dir=tmp_path / "render"
    )
    assert dag.segments[0].src_path == str(raw)
    assert dag.segments[0].spec.source_digest is None


def test_source_digest_rides_cache_key(tmp_path):
    def _spec(digest: str | None) -> SegmentRenderSpec:
        return SegmentRenderSpec(
            kind="raw",
            asset_id="a1",
            source_start_s=0.0,
            source_dur_s=2.0,
            video=VideoRenderSpec(w=64, h=64),
            audio=AudioRenderSpec(),
            source_digest=digest,
        )

    assert segment_hash(_spec(None)) != segment_hash(_spec("digest-1"))
    assert segment_hash(_spec("digest-1")) == segment_hash(_spec("digest-1"))
    assert segment_hash(_spec("digest-1")) != segment_hash(_spec("digest-2"))


def test_narration_audio_rides_spec(tmp_path):
    from director_os.otio_engine.contract import NarratorClipMeta

    clip = ClipSpec(
        name="vo_1",
        source_start_s=0.0,
        dur_s=3.0,
        meta=NarratorClipMeta(script_text="Bodycam footage shows the alley."),
    )
    spec = TimelineSpec(
        name="t",
        fps=30,
        width=320,
        height=240,
        tracks=[TrackSpec(name="V1_master", items=[clip])],
    )
    timeline = build_timeline(spec)
    wav = tmp_path / "vo.mp3"
    wav.write_bytes(b"audio")
    cache = SegmentCache(tmp_path / "cache")
    dag = compile_timeline(
        timeline,
        assets={},
        cache=cache,
        out_dir=tmp_path / "render",
        narration_audio={"Bodycam footage shows the alley.": str(wav)},
    )
    assert dag.segments[0].spec.narration_audio_path == str(wav)
    assert dag.segments[0].spec.narration_audio_digest
    # without narration audio: silent render path, no digest
    dag2 = compile_timeline(timeline, assets={}, cache=cache, out_dir=tmp_path / "render2")
    assert dag2.segments[0].spec.narration_audio_path is None


def test_blur_region_still_rides_global_filtergraph(tmp_path):
    timeline = _timeline("a1", [(0.0, 2.0)])
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    cache = SegmentCache(tmp_path / "cache")
    dag = compile_timeline(
        timeline,
        assets={"a1": str(raw)},
        cache=cache,
        out_dir=tmp_path / "render",
        global_blur_regions=[BlurRegion(scope="global", bbox=BBox(x=0, y=0, w=10, h=10), sigma=20)],
    )
    assert len(dag.segments[0].spec.blur_regions) == 1


def test_card_items_do_not_break_clip_sources(tmp_path):
    spec = TimelineSpec(
        name="t",
        fps=30,
        width=320,
        height=240,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw_1", source_start_s=0.0, dur_s=2.0, meta=RawClipMeta(asset_id="a1")
                    )
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    ClipSpec(
                        name="card",
                        source_start_s=0.0,
                        dur_s=2.0,
                        meta=CardMeta(template="location_intro", fields={"title": "CASE FILE"}),
                    )
                ],
            ),
        ],
    )
    timeline = build_timeline(spec)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    cache = SegmentCache(tmp_path / "cache")
    dag = compile_timeline(
        timeline,
        assets={"a1": str(raw)},
        cache=cache,
        out_dir=tmp_path / "render",
        clip_sources={"a1@0.000+2.000": str(raw)},
    )
    assert len(dag.segments) == 1
