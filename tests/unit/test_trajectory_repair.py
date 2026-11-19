"""Pass 2 trajectory repair — bridging, lead-in/out, envelope, zero-leak contract."""

import pytest
from director_os.core.models import BBox
from director_os.vision_fx.compute.base import FaceTrack, TrackHit
from director_os.vision_fx.trajectory import (
    DEFAULT_LEAD_FRAMES,
    DEFAULT_SCALE_H,
    DEFAULT_SCALE_W,
    PerFrameRedaction,
    repair_trajectories,
)

FPS = 30.0


def _track(
    frames: list[int], x0: float = 500.0, vx: float = 0.0, w: int = 40, h: int = 40
) -> FaceTrack:
    """Real detections (stride 3) on `frames` (real frame indices)."""
    hits = [
        TrackHit(
            t_s=round(f / FPS, 3),
            x=int(x0 + vx * f),
            y=100,
            w=w,
            h=h,
            score=0.9,
        )
        for f in frames
    ]
    return FaceTrack(track_id="person_1", hits=hits)


def _hits_by_frame(traj_id: str, plan: PerFrameRedaction):
    for t in plan.trajectories:
        if t.track_id == traj_id:
            return {b.frame_idx: b for b in t.boxes}
    raise AssertionError(f"{traj_id} missing")


def test_dense_coverage_every_frame_of_active_span():
    plan = repair_trajectories([_track([3, 6, 9, 12])], fps=FPS, frame_w=1920, frame_h=1080)
    idx = {b.frame_idx for b in plan.trajectories[0].boxes}
    # lead-in clamps at frame 0 when the track starts that early
    assert idx == set(range(max(0, 3 - DEFAULT_LEAD_FRAMES), 12 + DEFAULT_LEAD_FRAMES + 1))


def test_gap_within_25_bridged_with_inflation():
    """A 12-frame dropout (<=25) is bridged; bridged boxes are bigger (+15%)."""
    plan = repair_trajectories([_track([9, 21])], fps=FPS, frame_w=1920, frame_h=1080)
    idx = _hits_by_frame("person_1", plan)
    for f in range(9, 22):
        assert f in idx, f"frame {f} not covered"
    bridged = idx[15]
    assert bridged.bridged
    assert bridged.w > 40 * 1.1, "bridged box must be inflated"


def test_gap_over_25_not_bridged():
    """A 40-frame hole splits coverage: no stale box held across its MIDDLE.

    Windows now carry their own lead-in/out (pre/post roll) — the true
    absence is only the gap's interior, clear of both windows' rolls."""
    plan = repair_trajectories([_track([9, 49])], fps=FPS, frame_w=1920, frame_h=1080)
    idx = _hits_by_frame("person_1", plan)
    covered = set(idx)
    hole = [f for f in range(10 + DEFAULT_LEAD_FRAMES, 49 - DEFAULT_LEAD_FRAMES) if f in covered]
    assert not hole, f"stale boxes held across unbridgeable gap: {hole[:5]}"
    assert plan.trajectories[0].n_windows == 2
    # both windows get their own pre/post roll now
    assert 9 - DEFAULT_LEAD_FRAMES in covered
    assert 49 + DEFAULT_LEAD_FRAMES in covered


def test_declared_stride_bridges_sampled_cadence():
    """THE flicker fix (owner finding 2026-11-20): a 1.0 s sweep at 30 fps
    strides 30 frames between real hits — more than the fixed 25 bridge, so
    every gap used to shatter into 1-frame windows (blur applied/removed
    within milliseconds). Declaring the stride scales the bridge budget and
    the plan comes out DENSE across the whole active span."""
    plan = repair_trajectories(
        [_track([9, 39, 69])],
        fps=FPS,
        frame_w=1920,
        frame_h=1080,
        sample_stride_frames=30,
    )
    idx = {b.frame_idx for b in plan.trajectories[0].boxes}
    assert idx == set(range(9 - DEFAULT_LEAD_FRAMES, 69 + DEFAULT_LEAD_FRAMES + 1))


def test_short_interpolated_fills_are_kept():
    """Missed-sample fills from the tracker (interpolated=True) join the plan
    when the real-to-real gap they fill is <= interp_gap_max_s."""
    hits = [TrackHit(t_s=9 / FPS, x=500, y=100, w=40, h=40, score=0.9)]
    for f in range(10, 15):  # one missed sample at stride 5 (0.2 s gap)
        hits.append(
            TrackHit(
                t_s=round(f / FPS, 3),
                x=500,
                y=100,
                w=40,
                h=40,
                score=0.9,
                interpolated=True,
            )
        )
    hits.append(TrackHit(t_s=15 / FPS, x=500, y=100, w=40, h=40, score=0.9))
    plan = repair_trajectories(
        [FaceTrack(track_id="person_1", hits=hits)], fps=FPS, frame_w=1920, frame_h=1080
    )
    idx = {b.frame_idx for b in plan.trajectories[0].boxes}
    assert set(range(4, 21)).issubset(idx)


def test_long_interpolated_fills_dropped_no_stale_box():
    """Fills spanning a true absence (> interp_gap_max_s) are discarded — the
    window must split instead of holding a stale box through an occlusion."""
    hits = [TrackHit(t_s=9 / FPS, x=500, y=100, w=40, h=40, score=0.9)]
    for f in range(10, 100):  # 3 s of fill — a real absence
        hits.append(
            TrackHit(
                t_s=round(f / FPS, 3),
                x=500,
                y=100,
                w=40,
                h=40,
                score=0.9,
                interpolated=True,
            )
        )
    hits.append(TrackHit(t_s=100 / FPS, x=500, y=100, w=40, h=40, score=0.9))
    plan = repair_trajectories(
        [FaceTrack(track_id="person_1", hits=hits)], fps=FPS, frame_w=1920, frame_h=1080
    )
    idx = {b.frame_idx for b in plan.trajectories[0].boxes}
    assert 40 not in idx and 70 not in idx, "stale fill held across an absence"


def test_isolated_real_hits_survive_min_window():
    """LEAK LAW (owner finding 2026-11-20): with lead_frames=0, isolated
    1-frame hits still form their windows — a real detection may NEVER be
    dropped by the noise-window filter; only bridged/interp artifacts can."""
    plan = repair_trajectories(
        [_track([100, 140])],
        fps=FPS,
        frame_w=1920,
        frame_h=1080,
        lead_frames=0,
        min_window_frames=3,
    )
    idx = _hits_by_frame("person_1", plan)
    assert 100 in idx and 140 in idx, "isolated real detection dropped (leak)"


def test_lead_in_and_out_exact():
    plan = repair_trajectories([_track([30, 60])], fps=FPS, frame_w=1920, frame_h=1080)
    idx = _hits_by_frame("person_1", plan)
    assert min(idx) == 30 - DEFAULT_LEAD_FRAMES
    assert max(idx) == 60 + DEFAULT_LEAD_FRAMES
    assert idx[min(idx)].lead and idx[max(idx)].lead


def test_envelope_covers_every_raw_box():
    """THE contract: padded box contains raw detection on every active frame."""
    raw = _track([3, 6, 9, 12, 15, 18, 21], x0=500, vx=8.0)
    plan = repair_trajectories([raw], fps=FPS, frame_w=1920, frame_h=1080)
    idx = _hits_by_frame("person_1", plan)
    for h in raw.hits:
        fi = round(h.t_s * FPS)
        box = idx[fi]
        assert box.x <= h.x and box.y <= h.y
        assert box.x + box.w >= h.x + h.w and box.y + box.h >= h.y + h.h


def test_envelope_min_scale_when_smooth_hugs():
    """Even a zero-motion track gets at least the 1.35/1.40 safety scale."""
    plan = repair_trajectories([_track([3, 6, 9])], fps=FPS, frame_w=1920, frame_h=1080)
    box = _hits_by_frame("person_1", plan)[6]
    assert box.w >= 40 * DEFAULT_SCALE_W * 0.95
    assert box.h >= 40 * DEFAULT_SCALE_H * 0.95


def test_boxes_at_binary_search():
    plan = repair_trajectories([_track([3, 6, 9, 12])], fps=FPS, frame_w=1920, frame_h=1080)
    assert len(plan.boxes_at(6)) == 1
    assert len(plan.boxes_at(5)) == 1  # bridged
    assert plan.boxes_at(100) == []


def test_two_tracks_stay_separate():
    a = FaceTrack(track_id="person_1", hits=[TrackHit(t_s=0.1, x=10, y=10, w=40, h=40, score=0.9)])
    b = FaceTrack(
        track_id="person_2", hits=[TrackHit(t_s=0.2, x=900, y=300, w=50, h=50, score=0.8)]
    )
    plan = repair_trajectories([a, b], fps=FPS, frame_w=1920, frame_h=1080)
    assert {t.track_id for t in plan.trajectories} == {"person_1", "person_2"}


def test_clamped_to_frame_bounds():
    a = FaceTrack(
        track_id="person_edge",
        hits=[
            TrackHit(t_s=0.1, x=1890, y=1050, w=50, h=50, score=0.9),
            TrackHit(t_s=0.2, x=1895, y=1055, w=50, h=50, score=0.9),
        ],
    )
    plan = repair_trajectories([a], fps=FPS, frame_w=1920, frame_h=1080)
    for b in plan.trajectories[0].boxes:
        assert b.x >= 0 and b.y >= 0
        assert b.x + b.w <= 1920 and b.y + b.h <= 1080


def test_rejects_bad_fps():
    with pytest.raises(ValueError):
        repair_trajectories([_track([3])], fps=0, frame_w=1920, frame_h=1080)


def test_static_union_path_still_available():
    """Backward-compat: tracks_to_regions still emits static padded regions."""
    from director_os.vision_fx.compute.tracker import tracks_to_regions

    regions = tracks_to_regions([_track([3, 6, 9])], frame_w=1920, frame_h=1080)
    assert len(regions) == 1
    assert regions[0].track_id == "person_1"
    assert isinstance(regions[0].bbox, BBox)
