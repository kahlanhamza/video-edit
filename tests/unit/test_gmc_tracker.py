"""Pass 1 tracker — GMC keeps identities through whip-pans (issue #14).

The old failure mode: chest-camera pans fast, the face's absolute position
jumps by tens of pixels per sampled frame, bare-IoU prediction misses it, the
track dies and a NEW id is born — the first identity's boxes stop following
and the face leaks. Global motion compensation shifts predictions by the
camera's own displacement so matching survives.
"""

from director_os.vision_fx.compute.tracker import track_and_interpolate

FPS = 30.0
STRIDE = 3  # sampled every 3rd real frame


def _sweep(frames: list[int], x0: float = 400.0, w: int = 40) -> tuple[list, list, list]:
    """A face at (x0 + vx*f, 100) sampled on `frames`; camera pans `pan` px/frame."""
    dets, ts, motion = [], [], []
    for f in frames:
        dets.append([(int(x0 + 8.0 * f), 100, w, 40, 0.9)])
        ts.append(round(f / FPS, 3))
        motion.append((0.0, 0.0))
    return dets, ts, motion


def test_slow_motion_single_identity():
    dets, ts, _ = _sweep([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
    tracks = track_and_interpolate(dets, ts, min_hits=3)
    assert len(tracks) == 1
    assert tracks[0].n_real == 11


def test_whip_pan_with_gmc_keeps_identity():
    """30 px/frame camera pan + slow-moving face: GMC holds ONE track.

    Without compensation the prediction sits ~90px off after 3 sampled
    frames — IoU 0 vs the true box -> identity dies every few frames.
    """
    pan = 30.0  # px per REAL frame -> 90 px between sampled frames
    dets, ts, motion = [], [], []
    for f in range(0, 31, STRIDE):
        # screen position = world drift (2 px/frame) + camera pan (30 px/frame)
        dets.append([(int(400 + 32.0 * f), 100, 40, 40, 0.9)])
        ts.append(round(f / FPS, 3))
        # no motion BEFORE the first sampled frame — content shifts start at k=1
        motion.append((pan * STRIDE if f > 0 else 0.0, 0.0))
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=motion, track_buffer=60)
    assert len(tracks) == 1, f"GMC failed: {len(tracks)} identities on one face"
    assert tracks[0].n_real == 11


def test_whip_pan_without_gmc_shreds_identity():
    """The control: same sweep, no GMC — bare IoU drops the track (the bug)."""
    pan = 30.0
    dets, ts, _motion = [], [], []
    for f in range(0, 16, STRIDE):
        dets.append([(int(400 + 0.0 * f), 100, 40, 40, 0.9)])  # face static in WORLD
        ts.append(round(f / FPS, 3))
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=None)
    # static face, no camera motion -> tracker holds it (sanity of the fixture)
    assert len(tracks) == 1
    # now WITH camera motion but WITHOUT compensation: box positions on screen
    # jump by 90px/sample; predictions do not -> identity shredded (0-2 tracks,
    # never the single continuous identity GMC provides)
    dets2 = []
    for f in range(0, 16, STRIDE):
        dets2.append([(int(400 + pan * f), 100, 40, 40, 0.9)])
    tracks2 = track_and_interpolate(dets2, ts, min_hits=3)
    assert len(tracks2) != 1, "fixture broken: pan should shred bare-IoU tracking"


def _blackout_sweep(x0: float) -> tuple[list, list, list]:
    """Detector blackout: sampled frames 9..27 find NOTHING (empty entries).

    The face is static on screen — only the buffer (not velocity, not GMC)
    decides whether the identity survives the hole.
    """
    dets, ts, motion = [], [], []
    for f in range(0, 37, STRIDE):
        if 9 <= f <= 27:
            dets.append([])  # detector blackout — entry exists, no detections
        else:
            dets.append([(int(x0), 100, 40, 40, 0.9)])
        ts.append(round(f / FPS, 3))
        motion.append((0.0, 0.0))
    return dets, ts, motion


def test_track_buffer_60_survives_dropout():
    """7-sampled-frame detector blackout: track_buffer=60 re-associates the face."""
    dets, ts, motion = _blackout_sweep(400.0)
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=motion, track_buffer=60)
    assert len(tracks) == 1, "dropout with track_buffer=60 must not orphan identity"


def test_short_buffer_still_splits_legacy():
    """Legacy default (max_age=6) unchanged: the same hole splits."""
    dets, ts, motion = _blackout_sweep(400.0)
    tracks = track_and_interpolate(dets, ts, min_hits=3, gmc_vectors=motion)
    assert len(tracks) == 2, "legacy max_age=6 semantics must remain for compat"


def test_no_ema_lag_on_emitted_boxes():
    """Emitted hit boxes are RAW detector boxes — zero lag at Pass 1 output."""
    dets, ts, _ = _sweep([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30], 400.0)
    # rebuild with per-frame velocity: x = 400 + 8*f handled by fixture
    tracks = track_and_interpolate(dets, ts, min_hits=3)
    tr = tracks[0]
    last = tr.hits[-1]
    expected_x = 400 + 8 * 30
    assert abs(last.x - expected_x) <= 2, f"emitted box lags: {last.x} vs {expected_x}"
