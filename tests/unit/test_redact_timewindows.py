"""Tracked blur regions — time windows, FFmpeg enable gating, segment translation."""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.core.models import BBox, BlurRegion
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.segment_cache import SegmentCache
from director_os.otio_engine.timeline import ClipSpec, TimelineSpec, TrackSpec, build_timeline
from director_os.vision_fx.redact import build_blur_chain, clip_region_time


def _region(t0: float | None = None, t1: float | None = None, **kw) -> BlurRegion:
    return BlurRegion(
        bbox=BBox(x=100, y=50, w=80, h=60),
        mode="tracked" if (t0 is not None or t1 is not None) else "static",
        **kw,
        t_start_s=t0,
        t_end_s=t1,
    )


# ------------------------------------------------------------ clip_region_time
def test_static_region_passes_through_unchanged():
    r = _region()
    assert clip_region_time(r, seg_start_s=10.0, seg_dur_s=5.0) is r


def test_window_translates_to_segment_local_time():
    out = clip_region_time(_region(5.0, 9.0), seg_start_s=0.0, seg_dur_s=8.0)
    assert out is not None and (out.t_start_s, out.t_end_s) == (5.0, 8.0)


def test_window_open_start_clamps_to_zero():
    out = clip_region_time(_region(None, 12.0), seg_start_s=10.0, seg_dur_s=5.0)
    assert out is not None and (out.t_start_s, out.t_end_s) == (0.0, 2.0)


def test_window_outside_segment_is_dropped():
    assert clip_region_time(_region(20.0, 25.0), seg_start_s=0.0, seg_dur_s=8.0) is None
    assert clip_region_time(_region(5.0, 9.0), seg_start_s=10.0, seg_dur_s=5.0) is None


def test_subframe_overlap_is_dropped():
    assert clip_region_time(_region(7.98, 8.01), seg_start_s=0.0, seg_dur_s=8.0) is None


# ------------------------------------------------------------- blur chain gate
def test_blur_chain_emits_enable_window():
    chain = build_blur_chain("[v0]", "[vblur]", [_region(5.0, 8.0)], 640, 360)
    assert "enable='between(t,5.00,8.00)'" in chain
    assert "overlay=100:50:enable=" in chain


def test_blur_chain_static_has_no_enable():
    chain = build_blur_chain("[v0]", "[vblur]", [_region()], 640, 360)
    assert "enable=" not in chain
    assert "overlay=100:50[vblur]" in chain


def test_blur_chain_mixed_static_and_tracked():
    chain = build_blur_chain("[v0]", "[vblur]", [_region(), _region(1.0, 3.0)], 640, 360)
    assert chain.count("enable='between(t,1.00,3.00)'") == 1
    assert chain.count("overlay=") == 2


# ------------------------------------------------------------- compiler wiring
def _timeline_two_segments() -> TimelineSpec:
    return TimelineSpec(
        name="tracked",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw1",
                        source_start_s=0.0,
                        dur_s=8.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                    ClipSpec(
                        name="raw2",
                        source_start_s=8.0,
                        dur_s=8.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                ],
            )
        ],
    )


def test_compile_translates_tracked_windows_per_segment(tmp_path: Path):
    tl = build_timeline(_timeline_two_segments())
    dag = compile_timeline(
        tl,
        assets={"a1": "/media/a1.mp4"},
        cache=SegmentCache(tmp_path / "cache"),
        out_dir=tmp_path / "render",
        global_blur_regions=[_region(4.0, 12.0, sigma=24)],
    )
    bounds = [
        (s.spec.blur_regions[0].t_start_s, s.spec.blur_regions[0].t_end_s) for s in dag.segments
    ]
    assert bounds == [(4.0, 8.0), (0.0, 4.0)]


def test_compile_drops_regions_outside_all_segments(tmp_path: Path):
    tl = build_timeline(_timeline_two_segments())
    dag = compile_timeline(
        tl,
        assets={"a1": "/media/a1.mp4"},
        cache=SegmentCache(tmp_path / "cache"),
        out_dir=tmp_path / "render",
        global_blur_regions=[_region(20.0, 25.0)],
    )
    assert all(len(s.spec.blur_regions) == 0 for s in dag.segments)


def test_compile_static_region_untouched_by_time_translation(tmp_path: Path):
    tl = build_timeline(_timeline_two_segments())
    dag = compile_timeline(
        tl,
        assets={"a1": "/media/a1.mp4"},
        cache=SegmentCache(tmp_path / "cache"),
        out_dir=tmp_path / "render",
        global_blur_regions=[_region()],
    )
    for seg in dag.segments:
        r = seg.spec.blur_regions[0]
        assert (r.t_start_s, r.t_end_s) == (None, None)


# ------------------------------------------------- real ffmpeg render (smoke)
def test_render_tracked_region_end_to_end(tmp_path: Path, fixture_video: Path):
    """The enable-gated chain must survive a real encode, not just string checks."""
    from director_os.core.qa_probe import probe
    from director_os.otio_engine.renderer import render_dag
    from director_os.otio_engine.segment_cache import VideoRenderSpec

    video = VideoRenderSpec(w=320, h=240, fps=30)
    tl = build_timeline(
        TimelineSpec(
            name="tracked_smoke",
            fps=30,
            width=320,
            height=240,
            tracks=[
                TrackSpec(
                    name="V1_master",
                    items=[
                        ClipSpec(
                            name="raw1",
                            source_start_s=0.0,
                            dur_s=6.0,
                            meta=RawClipMeta(asset_id="fx"),
                        )
                    ],
                )
            ],
        )
    )
    dag = compile_timeline(
        tl,
        assets={"fx": str(fixture_video)},
        cache=SegmentCache(tmp_path / "cache"),
        video=video,
        out_dir=tmp_path / "render",
        global_blur_regions=[_region(1.0, 3.0)],
        asset_geometry={"fx": (320, 240)},  # spaces match: scale path is a no-op
    )
    report = render_dag(dag, SegmentCache(tmp_path / "cache"), log=lambda _m: None)
    assert report.final_path.exists()
    info = probe(str(report.final_path))
    assert info.duration_s == pytest.approx(6.0, abs=0.2)
    region = dag.segments[0].spec.blur_regions[0]
    assert (region.t_start_s, region.t_end_s) == (1.0, 3.0)
    chain = build_blur_chain("[v0]", "[vblur]", dag.segments[0].spec.blur_regions, 320, 240)
    assert "enable='between(t,1.00,3.00)'" in chain
