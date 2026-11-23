"""Frame-grab service — the frame-by-frame law's shared primitive.

- one accurate-seek decode per (t, width); JPEG bytes out
- deterministic cache: same (t, width) is served with zero ffmpeg cost
- LRU eviction in tranches keeps the cache bounded
- past-EOF is a clear error, never a silently black frame
"""

from __future__ import annotations

import pytest
from director_os.core.frames import (
    FrameGrabError,
    clamp_width,
    frame_cache_path,
    grab_frame,
)

pytestmark = pytest.mark.usefixtures("fixture_video")


def test_grab_returns_jpeg_and_caches(fixture_video, tmp_path):
    cache = tmp_path / "frames"
    p1 = grab_frame(fixture_video, 1.0, cache_dir=cache, width=320)
    assert p1.exists()
    assert p1.read_bytes()[:3] == b"\xff\xd8\xff"  # JPEG magic
    assert p1.stat().st_size > 1000

    # second grab of the same (t, width): cache hit — same inode, no rewrite
    p2 = grab_frame(fixture_video, 1.0, cache_dir=cache, width=320)
    assert p2 == p1
    assert p2.stat().st_ino == p1.stat().st_ino


def test_cache_key_is_deterministic_per_time_and_width(tmp_path):
    a = frame_cache_path(tmp_path, 1.0, 320)
    b = frame_cache_path(tmp_path, 1.0, 320)
    c = frame_cache_path(tmp_path, 1.04, 320)
    d = frame_cache_path(tmp_path, 1.0, None)
    assert a == b
    assert a != c  # 1000ms vs 1040ms are different frames
    assert d.name.endswith("native.jpg")


def test_lru_eviction_keeps_cache_bounded(fixture_video, tmp_path):
    cache = tmp_path / "frames"
    for i in range(6):
        grab_frame(fixture_video, float(i), cache_dir=cache, width=160, max_files=3)
    leftovers = list(cache.glob("*.jpg"))
    assert len(leftovers) <= 3  # bounded — the machine never fills up scrubbing


def test_past_eof_is_a_clear_error(fixture_video, tmp_path):
    with pytest.raises(FrameGrabError, match="past end|no frame"):
        grab_frame(fixture_video, 10_000.0, cache_dir=tmp_path / "frames", width=320)


def test_width_clamps_into_range():
    assert clamp_width(10) == 64
    assert clamp_width(9999) == 1920
    assert clamp_width(None) is None
    assert clamp_width(640) == 640


def test_missing_media_raises(tmp_path):
    with pytest.raises(FrameGrabError, match="media missing"):
        grab_frame(tmp_path / "nope.mp4", 0.0, cache_dir=tmp_path, width=320)
