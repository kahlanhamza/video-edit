"""M4 effects: keyframed punch-in zoom + freeze-callout spotlight (PLAN §3.3E).

Covers the expression builder, the filtergraph composition order (blur BEFORE
zoom — boxes are authored on the un-zoomed frame), the freeze subgraph, the
deterministic spotlight PNG, and compiler wiring incl. raw->render bbox
mapping. A real-ffmpeg render smoke closes the loop end-to-end.
"""

from __future__ import annotations

import subprocess

import pytest
from director_os.core.models import BBox, ZoomKeyframe
from director_os.otio_engine.contract import FreezeCallout
from director_os.otio_engine.filters import (
    animated_zoom_filters,
    compose_filter_complex,
    keyframe_points,
    plain_video_chain,
    punch_filters,
    zoom_expression,
)
from director_os.otio_engine.segment_cache import SegmentRenderSpec, VideoRenderSpec
from director_os.vision_fx.spotlight import (
    SpotlightError,
    clamp_bbox,
    render_spotlight_overlay,
)
from PIL import Image


def _spec(**kw) -> SegmentRenderSpec:
    v = VideoRenderSpec(w=640, h=360, fps=30)
    base = dict(
        kind="raw",
        source_dur_s=10.0,
        video=v,
        codec_profile="libx264-yuv420p-30fps-48000hz",
    )
    base.update(kw)
    return SegmentRenderSpec(**base)


# ---------------------------------------------------------------- zoom expression


def test_zoom_expression_piecewise_linear():
    # points: zoom 1.0 at t=0 -> 1.3 at t=2s -> holds; 30fps -> F1=60
    expr = zoom_expression([(0.0, 1.0), (2.0, 1.3)], 30)
    assert expr.startswith("if(lte(on,60),")
    assert "1.0000+(+0.3000)*(on-0)/60" in expr
    assert expr.endswith(",1.3000)")  # holds 1.3 past the keyframe


def test_zoom_expression_multiple_keyframes_and_holds():
    expr = zoom_expression([(0.0, 1.0), (1.0, 1.2), (4.0, 2.0)], 30)
    assert "if(lte(on,30)" in expr and "if(lte(on,120)" in expr
    assert "1.2000+(+0.8000)*(on-30)/90" in expr
    assert expr.endswith("2.0000))")  # holds 2.0 past the last keyframe (nested if)


def test_zoom_expression_non_advancing_anchor_collapses():
    # two keyframes landing on the same frame — later value wins, no /0 lerp
    expr = zoom_expression([(0.0, 1.0), (1.0, 1.5), (1.0, 2.0)], 30)
    assert "/0" not in expr
    assert "2.0000" in expr


def test_keyframe_points_starts_at_identity():
    ks = [ZoomKeyframe(t_s=2.0, zoom_factor=1.4)]
    assert keyframe_points(ks) == [(0.0, 1.0), (2.0, 1.4)]


def test_animated_zoom_filters_prescale_and_zoompan():
    flt = animated_zoom_filters([ZoomKeyframe(t_s=1.0, zoom_factor=2.0)], 640, 360, 30)
    assert flt[0] == "fps=30"  # rate normalization -> on/fps == t
    assert "scale=1280:720" in flt[1]  # headroom at max zoom
    assert any("zoompan=" in f and "s=640x360" in f and "d=1" in f for f in flt)
    assert any("x='iw/2-(iw/zoom/2)'" in f for f in flt)
    assert flt[-1] == "setsar=1"


def test_static_punch_unchanged_when_no_keyframes():
    assert animated_zoom_filters([], 640, 360, 30) == []
    assert punch_filters(1.5, 640, 360) == ["scale=960:540:flags=lanczos", "crop=640:360"]


# ---------------------------------------------------------------- composition order


def test_blur_runs_before_zoom_in_graph():
    from director_os.core.models import BlurRegion

    spec = _spec(
        blur_regions=[BlurRegion(bbox=BBox(x=10, y=10, w=100, h=60))],
        zoom_keyframes=[ZoomKeyframe(t_s=2.0, zoom_factor=1.5)],
    )
    graph, uses_complex, n_extra = compose_filter_complex(spec)
    assert uses_complex and n_extra == 0
    assert graph.index("boxblur") < graph.index("zoompan")


def test_plain_chain_carries_static_zoom_and_no_complex():
    spec = _spec(zoom_factor=1.25)
    chain = plain_video_chain(spec)
    assert "scale=800:450" in chain and "crop=640:360" in chain
    _, uses_complex, _ = compose_filter_complex(spec)
    assert not uses_complex


def test_keyframed_zoom_uses_zoompan_in_plain_chain():
    spec = _spec(zoom_keyframes=[ZoomKeyframe(t_s=1.0, zoom_factor=1.5)])
    chain = plain_video_chain(spec)
    assert "zoompan=" in chain


# ---------------------------------------------------------------- freeze callout


def test_freeze_graph_split_trim_tpad_overlay():
    spec = _spec(
        freeze_callout=FreezeCallout(freeze_at_s=4.0, hold_s=3.0),
        spotlight_png_path=None,
    )
    graph, uses_complex, n_extra = compose_filter_complex(spec)
    assert uses_complex and n_extra == 0
    assert "split=2" in graph
    assert "trim=start=4.000:end=4.035" in graph  # 1.05-frame pick window
    assert "tpad=stop_mode=clone:stop_duration=8.000" in graph  # 4+3+1 cover
    assert "overlay=eof_action=pass:enable='between(t,4.000,7.000)'" in graph


def test_freeze_with_spotlight_takes_input_and_full_frame_overlay():
    spec = _spec(
        freeze_callout=FreezeCallout(freeze_at_s=2.0, hold_s=2.0),
        spotlight_png_path="/tmp/spot.png",
    )
    graph, uses_complex, n_extra = compose_filter_complex(spec)
    assert uses_complex and n_extra == 1
    assert "[1:v]format=rgba[spot]" in graph
    assert "overlay=0:0:enable='between(t,2.000,4.000)'" in graph


def test_freeze_after_zoom_in_graph():
    spec = _spec(
        zoom_keyframes=[ZoomKeyframe(t_s=1.0, zoom_factor=1.5)],
        freeze_callout=FreezeCallout(freeze_at_s=2.0, hold_s=2.0),
    )
    graph, _, _ = compose_filter_complex(spec)
    assert graph.index("zoompan") < graph.index("split=2")


# ---------------------------------------------------------------- spotlight PNG


def test_spotlight_png_deterministic_and_pixel_probes(tmp_path):
    bbox = BBox(x=100, y=80, w=200, h=120)
    out1 = render_spotlight_overlay(640, 360, bbox, 0.45, tmp_path / "s1.png")
    out2 = render_spotlight_overlay(640, 360, bbox, 0.45, tmp_path / "s2.png")
    assert out1.read_bytes() == out2.read_bytes()

    with Image.open(out1) as im:
        px = im.convert("RGBA").load()
        # outside the box: dim black at 45%
        assert px[10, 10] == (0, 0, 0, 115)
        # hole center: fully transparent
        assert px[200, 140] == (0, 0, 0, 0)
        # accent outline on the box edge
        assert px[100, 140][0] == 255 and px[100, 140][3] == 255


def test_spotlight_none_bbox_dims_everything(tmp_path):
    out = render_spotlight_overlay(320, 180, None, 0.45, tmp_path / "s.png")
    with Image.open(out) as im:
        px = im.convert("RGBA").load()
        assert px[160, 90] == (0, 0, 0, 115)


def test_spotlight_clamps_and_rejects(tmp_path):
    assert clamp_bbox(BBox(x=600, y=300, w=100, h=100), 640, 360) == BBox(x=600, y=300, w=40, h=60)
    with pytest.raises(SpotlightError):
        render_spotlight_overlay(0, 0, None, 0.45, tmp_path / "s.png")
    with pytest.raises(SpotlightError):
        render_spotlight_overlay(640, 360, None, 1.5, tmp_path / "s.png")


# ---------------------------------------------------------------- real ffmpeg smoke


def test_real_render_keyframed_zoom_and_freeze(tmp_path):
    """End-to-end segment render on synthetic footage: zoom ramp + freeze hold.

    Duration is untouched by the freeze (audio keeps running under it), which
    the cache duration check depends on.
    """
    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(src),
        ],
        check=True,
    )
    spec = _spec(
        zoom_keyframes=[ZoomKeyframe(t_s=2.0, zoom_factor=1.4)],
        freeze_callout=FreezeCallout(freeze_at_s=4.0, hold_s=2.0, dim_alpha=0.45),
    )
    spot = tmp_path / "spot.png"
    render_spotlight_overlay(640, 360, BBox(x=200, y=100, w=240, h=160), 0.45, spot)
    spec.spotlight_png_path = str(spot)

    out = tmp_path / "seg.mp4"
    from director_os.otio_engine.renderer import render_segment

    render_segment(spec, str(src), out)
    assert out.exists() and out.stat().st_size > 2048
    from director_os.core.qa_probe import probe

    info = probe(out)
    assert abs(info.duration_s - 10.0) <= 0.25
    assert info.has_audio and info.has_video
