"""Segment cache: content-hashed keys (ADR-0003)."""

from __future__ import annotations

import pytest
from director_os.core.models import BBox, BlurRegion, ZoomKeyframe
from director_os.otio_engine.segment_cache import (
    AudioRenderSpec,
    CardOverlaySpec,
    SegmentCache,
    SegmentRenderSpec,
    VideoRenderSpec,
    segment_hash,
)
from director_os.vision_fx.cards import render_card


def base_spec() -> SegmentRenderSpec:
    return SegmentRenderSpec(
        kind="raw",
        asset_id="a1",
        source_start_s=0.0,
        source_dur_s=6.0,
        video=VideoRenderSpec(w=640, h=360, fps=30),
        audio=AudioRenderSpec(),
        audio_filters=["loudnorm=I=-16:TP=-1.5:LRA=11"],
    )


def test_stable_hash_for_identical_spec():
    assert segment_hash(base_spec()) == segment_hash(base_spec())


def test_zoom_keyframe_change_invalidates():
    a = base_spec()
    b = base_spec()
    b.zoom_keyframes = [ZoomKeyframe(t_s=2.0, zoom_factor=1.4)]
    assert segment_hash(a) != segment_hash(b)


def test_freeze_callout_change_invalidates():
    from director_os.otio_engine.contract import FreezeCallout

    a = base_spec()
    b = base_spec()
    b.freeze_callout = FreezeCallout(freeze_at_s=3.0, hold_s=2.0)
    assert segment_hash(a) != segment_hash(b)


def test_relative_cache_root_resolves_absolute(tmp_path, monkeypatch):
    """ffmpeg's concat demuxer resolves `file` lines against the LIST file, not
    the CWD — so cache paths must be absolute even when the root is relative."""
    monkeypatch.chdir(tmp_path)
    cache = SegmentCache("./output/segments")
    assert cache.root.is_absolute()
    assert cache.path_for("abc").is_absolute()


def test_spotlight_png_content_change_invalidates(tmp_path):
    from director_os.vision_fx.spotlight import render_spotlight_overlay

    a = base_spec()
    b = base_spec()
    p1 = tmp_path / "s1.png"
    p2 = tmp_path / "s2.png"
    render_spotlight_overlay(640, 360, BBox(x=10, y=10, w=100, h=80), 0.45, p1)
    render_spotlight_overlay(640, 360, BBox(x=150, y=10, w=100, h=80), 0.45, p2)
    a.spotlight_png_path = str(p1)
    b.spotlight_png_path = str(p2)
    assert segment_hash(a) != segment_hash(b)
    # identical content at different paths stays the same key (no path leakage)
    p3 = tmp_path / "s3.png"
    render_spotlight_overlay(640, 360, BBox(x=10, y=10, w=100, h=80), 0.45, p3)
    c = base_spec()
    c.spotlight_png_path = str(p3)
    assert segment_hash(a) == segment_hash(c)


def test_blur_sigma_change_invalidates():
    a = base_spec()
    b = base_spec()
    b.blur_regions = [BlurRegion(bbox=BBox(x=0, y=0, w=50, h=50), sigma=30)]
    assert segment_hash(a) != segment_hash(b)


def test_narration_text_change_invalidates():
    a = base_spec()
    b = base_spec()
    b.kind = "narration"
    b.narration_text = "The stop begins quietly."
    assert segment_hash(a) != segment_hash(b)


def test_card_content_change_invalidates(tmp_path):
    png1 = render_card(
        "location_intro", {"title": "JSO", "subtitle": "Jacksonville"}, tmp_path / "c1.png"
    )
    png2 = render_card(
        "location_intro", {"title": "JSP", "subtitle": "Jacksonville"}, tmp_path / "c2.png"
    )
    a = base_spec()
    a.cards = [CardOverlaySpec(png_path=str(png1), x=24, y=24, t_start_s=0.0, t_end_s=2.0)]
    b = base_spec()
    b.cards = [CardOverlaySpec(png_path=str(png2), x=24, y=24, t_start_s=0.0, t_end_s=2.0)]
    assert segment_hash(a) != segment_hash(b)


def test_same_content_different_path_same_hash(tmp_path):
    png = render_card("location_intro", {"title": "JSO"}, tmp_path / "a.png")
    copy = png.with_name("b.png")
    copy.write_bytes(png.read_bytes())
    a = base_spec()
    a.cards = [CardOverlaySpec(png_path=str(png), x=0, y=0, t_start_s=0.0, t_end_s=1.0)]
    b = base_spec()
    b.cards = [CardOverlaySpec(png_path=str(copy), x=0, y=0, t_start_s=0.0, t_end_s=1.0)]
    assert segment_hash(a) == segment_hash(b)


def test_cache_store_roundtrip(tmp_path):
    cache = SegmentCache(tmp_path / "segments")
    key = segment_hash(base_spec())
    assert cache.get(key) is None
    fake = tmp_path / "seg.mp4"
    fake.write_bytes(b"x" * 60_000)
    # not a playable MP4 -> rejected, never enters the cache
    from director_os.otio_engine.segment_cache import SegmentCacheError

    with pytest.raises(SegmentCacheError, match="not a playable MP4"):
        cache.put(fake, key)
    assert cache.get(key) is None


def test_tiny_valid_segment_is_cached(fixture_video, tmp_path):
    """All-black narration chunks are legitimately small — byte floors must not
    bounce them out of the cache on every pass (found in the 1080p demo)."""
    import subprocess

    cache = SegmentCache(tmp_path / "segments")
    key = "black" + "0" * 36
    tmp = tmp_path / "black.tmp.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=320x240:rate=30",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "34",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(tmp),
        ],
        check=True,
        capture_output=True,
    )
    stored = cache.put(tmp, key, expect_dur_s=2.0)
    assert stored.exists()
    assert cache.get(key) == stored


def test_truncated_encode_rejected(tmp_path):
    import subprocess

    cache = SegmentCache(tmp_path / "segments")
    tmp = tmp_path / "trunc.tmp.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "34",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(tmp),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(Exception, match="truncated encode|duration"):
        cache.put(tmp, "k" * 40, expect_dur_s=6.0)
    assert not tmp.exists()
