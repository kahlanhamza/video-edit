"""Modal GPU redaction — track payload conversion pins (hermetic, no fleet)."""

from __future__ import annotations

import pytest
from director_os.core.config import get_settings
from director_os.core.models import BBox, BlurRegion
from director_os.perception.modal_redact import (
    ModalRedactError,
    TrackedObject,
    TrackFrame,
    merge_with_static,
    modal_configured,
    parse_tracks_payload,
    redact_video_modal,
    tracks_to_regions,
)


def _payload() -> dict:
    return {
        "video": {"w": 1920, "h": 1080, "fps": 30.0, "duration_s": 10.0},
        "tracks": [
            {
                "id": 1,
                "cls": "face",
                "frames": [
                    {"t": 0.0, "bbox": [100, 50, 80, 60], "score": 0.9},
                    {"t": 0.5, "bbox": [102, 52, 82, 62], "score": 0.9},
                    {"t": 1.0, "bbox": [98, 48, 78, 58], "score": 0.9},
                    {"t": 1.5, "bbox": [100, 50, 80, 60], "score": 0.9},
                    {"t": 8.0, "bbox": [400, 300, 80, 60], "score": 0.8},  # 1 hit -> dropped
                ],
            },
            {
                "id": 2,
                "cls": "plate",
                "frames": [
                    {"t": 4.0, "bbox": [1100, 900, 200, 60], "score": 0.8},
                    {"t": 4.5, "bbox": [1100, 900, 200, 60], "score": 0.8},
                    {"t": 5.0, "bbox": [1100, 900, 200, 60], "score": 0.8},
                    {"t": 5.5, "bbox": [1100, 900, 200, 60], "score": 0.8},
                ],
            },
        ],
    }


def test_parse_tracks_payload_tolerant_reader():
    tracks = parse_tracks_payload(_payload())
    assert len(tracks) == 2
    assert tracks[0].track_id == 1 and tracks[0].kind == "face"
    assert len(tracks[0].frames) == 5


def test_parse_tracks_payload_accepts_alias_keys():
    payload = {
        "tracks": [
            {
                "track_id": 7,
                "class": "plate",
                "frames": [{"t_s": 1.0, "box": {"x": 1, "y": 2, "w": 3, "h": 4}}],
            }
        ]
    }
    tracks = parse_tracks_payload(payload)
    assert tracks[0].track_id == 7 and tracks[0].kind == "plate"
    assert tracks[0].frames[0].w == 3.0


def test_parse_tracks_payload_skips_junk_rows():
    payload = {"tracks": ["nope", {"no_id": True}, {"id": "x", "frames": "none"}]}
    assert parse_tracks_payload(payload) == []


def test_tracks_to_regions_bins_median_and_drops_sparse():
    regions = tracks_to_regions(parse_tracks_payload(_payload()), 1920, 1080)
    assert len(regions) == 2
    face = regions[0]
    assert face.track_id == "face_1" and face.mode == "tracked"
    assert (face.t_start_s, face.t_end_s) == (0.0, 2.0)
    # median of the four bin-0 observations: x=100, y=50, w=80, h=60, pad 12%
    assert (face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h) == (90, 43, 100, 74)
    plate = regions[1]
    assert plate.track_id == "plate_2"
    assert (plate.t_start_s, plate.t_end_s) == (4.0, 6.0)


def test_tracks_to_regions_clamps_padded_box_to_frame():
    obj = TrackedObject(
        track_id=3,
        kind="face",
        frames=[
            TrackFrame(t_s=0.0, x=1880, y=1020, w=100, h=100),
            TrackFrame(t_s=0.5, x=1880, y=1020, w=100, h=100),
            TrackFrame(t_s=1.0, x=1880, y=1020, w=100, h=100),
        ],
    )
    (r,) = tracks_to_regions([obj], 1920, 1080)
    assert r.bbox.x + r.bbox.w <= 1920
    assert r.bbox.y + r.bbox.h <= 1080


def test_tracks_to_regions_max_regions_keeps_top_by_hits():
    tracks = []
    for tid in range(6):
        tracks.append(
            TrackedObject(
                track_id=tid,
                kind="face",
                frames=[
                    TrackFrame(t_s=0.0 + i * 0.1, x=100, y=100, w=50, h=50) for i in range(9 - tid)
                ],
            )
        )
    regions = tracks_to_regions(tracks, 1920, 1080, max_regions=3)
    assert len(regions) == 3
    assert all(r.track_id == f"face_{i}" for i, r in enumerate(regions))


def test_merge_with_static_keeps_both_kinds_and_dedupes_tracked():
    tracked = tracks_to_regions(parse_tracks_payload(_payload()), 1920, 1080)
    static = BlurRegion(bbox=BBox(x=0, y=0, w=200, h=100))
    merged = merge_with_static(tracked, [static])
    assert len(merged) == 3
    assert merged[-1].mode == "static"
    assert len(merge_with_static(tracked, tracked)) == len(tracked)


def test_modal_not_configured_raises(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        assert modal_configured() is False
        with pytest.raises(ModalRedactError, match="MODAL_TOKEN"):
            redact_video_modal("/tmp/nope.mp4")
    finally:
        get_settings.cache_clear()
