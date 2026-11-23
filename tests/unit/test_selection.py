"""Selection service — promptable per-object tracking (roadmap step 3).

The honest-tracker contract: a selection follows the object the prompt named,
interpolates ONLY bounded miss gaps, and REFUSES when the object is lost —
a tracker that hallucinates boxes across occlusions is a privacy leak.
"""

from __future__ import annotations

import pytest
from director_os.vision_fx.compute.base import FaceTrack, TrackHit
from director_os.vision_fx.selection import (
    MAX_SPAN_S,
    Sam2SelectionBackend,
    SelectionError,
    SelectionNotAvailable,
    SelectionPrompt,
    TrackedSelection,
    available_selection_backends,
    probe_video,
    run_selection,
    select_from_tracks,
    track_selection_ncc,
)

FPS = 30.0


def _ground_truth(i: int) -> tuple[int, int, int, int]:
    """The box the conftest fixture drew on frame i."""
    x = 40 + 2 * i
    y = 60 + i
    return x, y, 28, 28


def _prompt_box(frame: int) -> SelectionPrompt:
    x, y, w, h = _ground_truth(frame)
    return SelectionPrompt(kind="box", x=x, y=y, w=w, h=h, anchor_frame=frame)


def test_probe_video_geometry(moving_box_video):
    dur, fps, w, h = probe_video(moving_box_video)
    assert 3.5 <= dur <= 4.5
    assert fps == pytest.approx(30.0, abs=0.01)
    assert (w, h) == (320, 240)


def test_ncc_tracks_the_box_across_the_span(moving_box_video):
    """Anchor at frame 30 (box has moved 6*30 px) -> track the whole span."""
    sel = track_selection_ncc(
        moving_box_video,
        _prompt_box(30),
        frame_start=0,
        frame_end=119,
        fps=FPS,
        frame_w=320,
        frame_h=240,
    )
    assert sel.backend == "ncc-cpu"
    assert len(sel.hits) >= 100
    # every REAL hit sits on the box (<= 8 px drift — encoder noise allowed)
    for hit in (h for h in sel.hits if not h.interpolated):
        i = int(round(hit.t_s * FPS))
        gx, gy, gw, gh = _ground_truth(i)
        assert abs(hit.x - gx) <= 8, f"frame {i}: x {hit.x} vs {gx}"
        assert abs(hit.y - gy) <= 8, f"frame {i}: y {hit.y} vs {gy}"
        assert abs(hit.w - gw) <= 8
    # interpolated hits stay ON the linear path too
    for hit in (h for h in sel.hits if h.interpolated):
        i = int(round(hit.t_s * FPS))
        gx, gy, _, _ = _ground_truth(i)
        assert abs(hit.x - gx) <= 10
    assert sel.mean_score >= 0.5


def test_ncc_anchors_late_and_walks_backwards(moving_box_video):
    """Anchor mid-span: backward tracking fills the frames BEFORE the anchor."""
    sel = track_selection_ncc(
        moving_box_video,
        _prompt_box(80),
        frame_start=40,
        frame_end=110,
        fps=FPS,
        frame_w=320,
        frame_h=240,
    )
    real_t = [h.t_s for h in sel.hits if not h.interpolated]
    assert min(real_t) <= 42.0 / FPS + 0.05  # walked back to the span start
    assert max(real_t) >= 108.0 / FPS  # and forward to the span end


def test_ncc_refuses_when_no_object_is_there(flat_video):
    """A prompt on a texture-free frame must FAIL, not fabricate a path."""
    with pytest.raises(SelectionError, match="texture-free"):
        track_selection_ncc(
            flat_video,
            SelectionPrompt(kind="box", x=150, y=100, w=24, h=24, anchor_frame=5),
            frame_start=0,
            frame_end=29,
            fps=FPS,
            frame_w=320,
            frame_h=240,
        )


def test_ncc_point_prompt_defaults_a_patch(moving_box_video):
    """A bare click centers a default patch — no box needed."""
    i = 40
    gx, gy, _, _ = _ground_truth(i)
    sel = track_selection_ncc(
        moving_box_video,
        SelectionPrompt(kind="point", x=gx + 14, y=gy + 14, anchor_frame=i),
        frame_start=10,
        frame_end=70,
        fps=FPS,
        frame_w=320,
        frame_h=240,
    )
    assert len(sel.hits) >= 40
    anchor_hit = min(sel.hits, key=lambda h: abs(h.t_s - i / FPS))
    # the default patch (40x30) OVERLAPS the box: its center sits on the box center
    cx, cy = anchor_hit.x + anchor_hit.w / 2, anchor_hit.y + anchor_hit.h / 2
    assert abs(cx - (gx + 14)) <= 10, f"center {cx} vs {gx + 14}"
    assert abs(cy - (gy + 14)) <= 10


def test_ncc_rejects_bad_windows(moving_box_video):
    with pytest.raises(SelectionError, match="frame_end"):
        track_selection_ncc(
            moving_box_video,
            _prompt_box(5),
            frame_start=40,
            frame_end=10,
            fps=FPS,
            frame_w=320,
            frame_h=240,
        )
    with pytest.raises(SelectionError, match="anchor"):
        track_selection_ncc(
            moving_box_video,
            _prompt_box(5),
            frame_start=60,
            frame_end=100,
            fps=FPS,
            frame_w=320,
            frame_h=240,
        )


def _face_track(track_id: str, frames: list[int]) -> FaceTrack:
    return FaceTrack(
        track_id=track_id,
        hits=[
            TrackHit(
                t_s=i / FPS,
                x=100 + 4 * i,
                y=50,
                w=30,
                h=30,
                score=0.9,
                interpolated=False,
            )
            for i in frames
        ],
    )


def test_detector_assist_picks_iou_best_track():
    tr_a = _face_track("person_1", [0, 10, 20, 30, 40])
    # person_2 overlaps the prompt exactly at the anchor frame 20
    tr_b = FaceTrack(
        track_id="person_2",
        hits=[
            TrackHit(t_s=i / FPS, x=180 + 2 * i, y=50, w=30, h=30, score=0.8)
            for i in (0, 10, 20, 30, 40)
        ],
    )
    anchor = 20
    sel = select_from_tracks(
        [tr_a, tr_b],
        SelectionPrompt(kind="box", x=180 + 2 * anchor, y=50, w=30, h=30, anchor_frame=anchor),
        frame_start=0,
        frame_end=40,
        fps=FPS,
        frame_w=320,
        frame_h=240,
    )
    assert sel.backend == "detector-assist:person_2"
    assert len(sel.hits) == 5


def test_detector_assist_refuses_unmatched_prompt():
    tr = _face_track("person_1", [0, 10, 20])
    with pytest.raises(SelectionError, match="no detector track"):
        select_from_tracks(
            [tr],
            SelectionPrompt(kind="box", x=290, y=200, w=20, h=20, anchor_frame=10),
            frame_start=0,
            frame_end=20,
            fps=FPS,
            frame_w=320,
            frame_h=240,
        )


def test_chooser_falls_back_to_ncc_when_no_track_matches(moving_box_video):
    """detector-assist refusal on a non-face object -> template tracker owns it."""
    tr = _face_track("person_1", [0, 10, 20])
    i = 40
    gx, gy, gw, gh = _ground_truth(i)
    sel = run_selection(
        str(moving_box_video),
        SelectionPrompt(kind="box", x=gx, y=gy, w=gw, h=gh, anchor_frame=i),
        frame_start=20,
        frame_end=80,
        fps=FPS,
        frame_w=320,
        frame_h=240,
        detector_tracks=[tr],
    )
    assert sel.backend == "ncc-cpu"


def test_chooser_explicit_sam2_is_loud_not_silent():
    """SELECTION_BACKEND=sam2 without weights: LOUD error, never a silent swap."""
    with pytest.raises((SelectionNotAvailable, SelectionError)):
        run_selection(
            "whatever.mp4",
            SelectionPrompt(kind="box", x=1, y=1, w=4, h=4, anchor_frame=0),
            frame_start=0,
            frame_end=10,
            fps=FPS,
            frame_w=320,
            frame_h=240,
            backend="sam2",
        )


def test_sam2_unavailable_message_names_the_flip():
    backend = Sam2SelectionBackend()
    if not backend.available():  # CI: no ultralytics extra installed
        with pytest.raises(SelectionNotAvailable, match="SELECTION_BACKEND=sam2"):
            backend.track("x.mp4", _prompt_box(0))
    assert available_selection_backends()["ncc-cpu"] is True


def test_union_box_covers_the_whole_path():
    hits = [
        TrackHit(t_s=i / FPS, x=40 + 2 * i, y=60 + i, w=28, h=28, score=0.9) for i in range(0, 120)
    ]
    sel = TrackedSelection(
        backend="ncc-cpu",
        anchor_frame=0,
        fps=FPS,
        frame_start=0,
        frame_end=119,
        frame_w=320,
        frame_h=240,
        hits=hits,
    )
    box = sel.union_box()
    # first box fully inside the union; last box edge is clamped to the frame
    assert box["x"] <= 40
    assert box["y"] <= 60
    right_last = min(320, 40 + 2 * 119 + 28)  # 306+28 stays inside 320
    bottom_last = min(240, 60 + 119 + 28)  # 207 stays inside 240
    assert box["x"] + box["w"] >= right_last
    assert box["y"] + box["h"] >= bottom_last
    assert box["x"] + box["w"] <= 320 and box["y"] + box["h"] <= 240


def test_to_params_carries_the_tracked_path():
    hits = [TrackHit(t_s=i / FPS, x=10 * i, y=5 * i, w=28, h=28, score=0.9) for i in range(12)]
    sel = TrackedSelection(
        backend="ncc-cpu",
        anchor_frame=6,
        fps=FPS,
        frame_start=0,
        frame_end=11,
        frame_w=320,
        frame_h=240,
        hits=hits,
        mean_score=0.82,
    )
    params = sel.to_params()
    assert params["mode"] == "tracked"
    assert params["n_tracked"] == 12
    assert len(params["tracked"]) == 12
    assert params["tracked"][6]["x"] == 60
    assert params["tracked_backend"] == "ncc-cpu"


def test_span_cap_constant_is_sane():
    assert 10 <= MAX_SPAN_S <= 120
