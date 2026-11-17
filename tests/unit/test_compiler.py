"""Compiler: OTIO -> RenderDAG (segment hashing, card mapping, uniformity)."""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.otio_engine.compiler import (
    RenderDAG,
    TimelineCompileError,
    compile_timeline,
    validate_uniformity,
)
from director_os.otio_engine.contract import CardMeta, NarratorClipMeta, RawClipMeta
from director_os.otio_engine.segment_cache import SegmentCache
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)


def timeline_spec() -> TimelineSpec:
    return TimelineSpec(
        name="t",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw1", source_start_s=0.0, dur_s=6.0, meta=RawClipMeta(asset_id="a1")
                    ),
                    ClipSpec(
                        name="narr1",
                        source_start_s=0.0,
                        dur_s=4.0,
                        meta=NarratorClipMeta(script_text="He never saw it coming."),
                    ),
                    ClipSpec(
                        name="raw2", source_start_s=10.0, dur_s=5.0, meta=RawClipMeta(asset_id="a2")
                    ),
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    GapSpec(dur_s=7.0),
                    ClipSpec(
                        name="card",
                        source_start_s=0.0,
                        dur_s=3.0,
                        meta=CardMeta(
                            template="location_intro",
                            fields={"title": "JSO", "subtitle": "Jacksonville, FL"},
                        ),
                    ),
                ],
            ),
        ],
    )


@pytest.fixture()
def dag(tmp_path) -> RenderDAG:
    tl = build_timeline(timeline_spec())
    cache = SegmentCache(tmp_path / "cache")
    return compile_timeline(
        tl,
        assets={"a1": "/media/a1.mp4", "a2": "/media/a2.mp4"},
        cache=cache,
        out_dir=tmp_path / "render",
    )


def test_three_segments_in_order(dag: RenderDAG):
    assert [s.spec.kind for s in dag.segments] == ["raw", "narration", "raw"]
    assert [round(s.dur_s, 3) for s in dag.segments] == [6.0, 4.0, 5.0]
    assert dag.total_dur_s == pytest.approx(15.0)
    assert len({s.key for s in dag.segments}) == 3, "distinct specs -> distinct keys"


def test_narration_segment_is_silence_backed(dag: RenderDAG):
    narr = dag.segments[1]
    assert narr.spec.audio_filters == []
    assert narr.spec.narration_text == "He never saw it coming."
    assert narr.src_path is None  # no broll -> black background


def test_card_lands_on_owning_segment(dag: RenderDAG):
    seg = dag.segments[1]  # abs 7.0s falls inside narration seg (6.0-10.0)
    assert len(seg.spec.cards) == 1
    card = seg.spec.cards[0]
    assert card.t_start_s == pytest.approx(1.0)
    assert card.t_end_s == pytest.approx(4.0)
    assert Path(card.png_path).exists()
    assert 0 <= card.x < 640 and 0 <= card.y < 360
    # card re-hashed the segment spec
    from director_os.otio_engine.segment_cache import segment_hash

    assert seg.key == segment_hash(seg.spec)


def test_unknown_asset_raises(tmp_path):
    tl = build_timeline(timeline_spec())
    cache = SegmentCache(tmp_path / "cache")
    with pytest.raises(TimelineCompileError):
        compile_timeline(
            tl,
            assets={"a1": "/media/a1.mp4"},  # a2 missing
            cache=cache,
            out_dir=tmp_path / "r2",
        )


# --------------------------------------------- raw->render blur space mapping
def test_scale_region_same_space_is_noop():
    from director_os.core.models import BBox, BlurRegion
    from director_os.vision_fx.redact import scale_region

    r = BlurRegion(bbox=BBox(x=100, y=50, w=200, h=100), sigma=12)
    assert scale_region(r, 1920, 1080, 1920, 1080) is r


def test_scale_region_downscales_with_aspect():
    from director_os.core.models import BBox, BlurRegion
    from director_os.vision_fx.redact import scale_region

    r = BlurRegion(bbox=BBox(x=960, y=540, w=400, h=300))
    out = scale_region(r, 1920, 1080, 1280, 720)
    # s = 2/3, no letterbox on matched 16:9
    assert (out.bbox.x, out.bbox.y) == (640, 360)
    assert (out.bbox.w, out.bbox.h) == (267, 200)
    assert out.sigma == r.sigma and out.mode == r.mode and out.track_id == r.track_id


def test_scale_region_letterbox_offsets_y():
    from director_os.core.models import BBox, BlurRegion
    from director_os.vision_fx.redact import scale_region

    # 2.4:1 source into 16:9 render: s = 640/1920 = 1/3, source height 800*1/3=266.4
    r = BlurRegion(bbox=BBox(x=0, y=0, w=1920, h=800))
    out = scale_region(r, 1920, 800, 640, 360)
    pad_y = (360 - 800 / 3) / 2  # 46.8 -> 47
    assert out.bbox.y == int(round(pad_y))
    assert out.bbox.w == 640
    assert out.bbox.h == int(round(800 / 3))


def test_compile_maps_global_blur_from_raw_space(tmp_path):
    """Studio canvas boxes are raw-pixel coords; render at 720p must rescale them."""
    from director_os.core.models import BBox, BlurRegion
    from director_os.otio_engine.segment_cache import VideoRenderSpec

    tl = build_timeline(timeline_spec())
    cache = SegmentCache(tmp_path / "cache")
    dag = compile_timeline(
        tl,
        assets={"a1": "/media/a1.mp4", "a2": "/media/a2.mp4"},
        cache=cache,
        out_dir=tmp_path / "r3",
        video=VideoRenderSpec(w=1280, h=720),
        global_blur_regions=[BlurRegion(bbox=BBox(x=1600, y=60, w=280, h=160), sigma=18)],
        asset_geometry={"a1": (1920, 1080), "a2": (1920, 1080)},
    )
    raw_seg = dag.segments[0]
    assert len(raw_seg.spec.blur_regions) == 1
    box = raw_seg.spec.blur_regions[0].bbox
    # 1600*2/3=1067, 60*2/3=40, 280*2/3=187, 160*2/3=107
    assert (box.x, box.y, box.w, box.h) == (1067, 40, 187, 107)
    # narration segment never carries blur
    assert dag.segments[1].spec.blur_regions == []
    # second raw asset scales too
    box2 = dag.segments[2].spec.blur_regions[0].bbox
    assert (box2.x, box2.y, box2.w, box2.h) == (1067, 40, 187, 107)


def test_compile_without_geometry_keeps_coords(tmp_path):
    """Backward compat: no asset_geometry -> regions pass through untouched."""
    from director_os.core.models import BBox, BlurRegion

    tl = build_timeline(timeline_spec())
    cache = SegmentCache(tmp_path / "cache")
    dag = compile_timeline(
        tl,
        assets={"a1": "/media/a1.mp4", "a2": "/media/a2.mp4"},
        cache=cache,
        out_dir=tmp_path / "r4",
        global_blur_regions=[BlurRegion(bbox=BBox(x=1600, y=60, w=280, h=160))],
    )
    assert dag.segments[0].spec.blur_regions[0].bbox.x == 1600


def test_uniformity_guard(dag: RenderDAG):
    validate_uniformity(dag)  # must not raise
    broken = RenderDAG(
        segments=list(dag.segments), final_path=dag.final_path, total_dur_s=dag.total_dur_s
    )
    from director_os.otio_engine.segment_cache import VideoRenderSpec

    forged_spec = dag.segments[1].spec.model_copy(deep=True)
    forged_spec.video = VideoRenderSpec(w=1920, h=1080, fps=24)
    broken.segments[1] = type(dag.segments[1])(
        key="x", spec=forged_spec, src_path=None, out_path=Path("/tmp/x.mp4"), dur_s=4.0
    )
    with pytest.raises(TimelineCompileError):
        validate_uniformity(broken)


def test_loudnorm_attached_to_raw_segments(dag: RenderDAG):
    from director_os.core.ffmpeg_args import loudnorm_chain

    assert dag.segments[0].spec.audio_filters == [loudnorm_chain(-16.0)]
