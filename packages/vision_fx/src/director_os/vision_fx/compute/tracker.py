"""IoU tracker with stride sampling + linear interpolation — persistent IDs.

BoT-SORT-style behavior without the dependency: greedy IoU matching on sampled
frames, **global motion compensation** (sparse-optical-flow camera motion is
subtracted from predictions so chest-camera whip-pans keep Track IDs intact —
the exact failure mode of bare IoU on bodycams), `min_hits` confirmation
(one-shot false positives never earn a track), `track_buffer` tolerance
(BoT-SORT default 60 sampled frames), then per-track linear interpolation
across the stride gaps. This is the host-side half of BOTH local and cloud
backends: Kaggle/Modal kernels dump raw per-frame detections and re-attach
them here, keeping tracker logic in exactly one place.

Smoothing policy for redaction: NO EMA on emitted boxes (a smoothed box lags
fast motion — a privacy leak). Pass 2 (`trajectory.py`) owns smoothing via a
One-Euro filter + safety envelope; this module emits raw matched boxes.
"""

from __future__ import annotations

from director_os.core.models import BBox
from director_os.vision_fx.compute.base import FaceTrack, TrackHit


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _TrackState:
    __slots__ = (
        "track_id",
        "xyxy",
        "score",
        "hits",
        "missed",
        "confirmed",
        "last_t",
        "vel",
        "gmc",
    )

    def __init__(
        self,
        track_id: int,
        xyxy: tuple[float, float, float, float],
        score: float,
        t_s: float,
        gmc: tuple[float, float] = (0.0, 0.0),
    ):
        self.track_id = track_id
        self.xyxy = xyxy
        self.score = score
        self.hits: list[TrackHit] = [
            TrackHit(
                t_s=t_s,
                x=int(xyxy[0]),
                y=int(xyxy[1]),
                w=int(xyxy[2] - xyxy[0]),
                h=int(xyxy[3] - xyxy[1]),
                score=score,
            )
        ]
        self.missed = 0
        self.confirmed = False
        self.last_t = t_s
        self.vel = (0.0, 0.0)  # motion-compensated velocity, px/s
        self.gmc = gmc  # accumulator value at birth — future deltas stay step-local

    def predict(
        self,
        t_s: float,
        gmc_delta: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float, float, float]:
        """Box extrapolated to `t_s` — camera motion shift + constant velocity.

        `gmc_delta` is the accumulated global motion since this track's last
        update: the camera moved, so EVERY static-scene box moved with it.
        Matching against the un-shifted prediction IoU-collapses on pans.
        """
        dt = t_s - self.last_t
        cx = (self.xyxy[0] + self.xyxy[2]) / 2 + self.vel[0] * dt + gmc_delta[0]
        cy = (self.xyxy[1] + self.xyxy[3]) / 2 + self.vel[1] * dt + gmc_delta[1]
        w, h = self.xyxy[2] - self.xyxy[0], self.xyxy[3] - self.xyxy[1]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def update(
        self,
        xyxy: tuple[float, float, float, float],
        score: float,
        t_s: float,
        gmc_acc: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        dt = t_s - self.last_t
        if dt > 0:
            gmc_delta = (gmc_acc[0] - self.gmc[0], gmc_acc[1] - self.gmc[1])
            # previous center carried forward by CAMERA motion (not object
            # velocity) — velocity estimated on motion-compensated positions
            # stays local to the object instead of absorbing the pan
            pcx = (self.xyxy[0] + self.xyxy[2]) / 2 + gmc_delta[0]
            pcy = (self.xyxy[1] + self.xyxy[3]) / 2 + gmc_delta[1]
            ncx = (xyxy[0] + xyxy[2]) / 2
            ncy = (xyxy[1] + xyxy[3]) / 2
            self.vel = ((ncx - pcx) / dt, (ncy - pcy) / dt)
        self.xyxy = xyxy
        self.last_t = t_s
        self.score = score
        self.missed = 0
        self.gmc = gmc_acc

    def _emit(self, t_s: float, score: float) -> TrackHit:
        """Emit the RAW matched box — Pass 2 owns smoothing (One-Euro + envelope)."""
        return TrackHit(
            t_s=t_s,
            x=int(self.xyxy[0]),
            y=int(self.xyxy[1]),
            w=int(self.xyxy[2] - self.xyxy[0]),
            h=int(self.xyxy[3] - self.xyxy[1]),
            score=score,
        )


def track_and_interpolate(
    detections_per_frame: list[list[tuple[int, int, int, int, float]]],
    t_per_frame: list[float],
    *,
    sample_step_s: float | None = None,
    iou_threshold: float = 0.2,
    min_hits: int = 3,
    max_age: int = 6,
    ema_alpha: float = 1.0,
    gmc_vectors: list[tuple[float, float]] | None = None,
    track_buffer: int | None = None,
) -> list[FaceTrack]:
    """Greedy-IoU tracking over sampled frames -> interpolated FaceTracks.

    `detections_per_frame[k]` holds [(x, y, w, h, score)] at time
    `t_per_frame[k]`; frames must be ordered. `sample_step_s` is the sampling
    cadence (defaults to the median positive diff of `t_per_frame`) — stride
    gaps larger than one step are filled by linear interpolation. Tracks
    below `min_hits` real detections are dropped.

    `gmc_vectors[k]` = (dx, dy) scene displacement between sampled frames
    k-1 and k (sparse optical flow): camera motion is accumulated and every
    prediction is shifted by it, so whip-pans do not shred identities.
    `track_buffer` (BoT-SORT default 60) overrides `max_age` when given —
    how long a track stays matchable after its last detection, in sampled
    frames. Redaction sweeps use the long buffer: re-associating a face is
    always safer than orphaning it into a second identity.

    `ema_alpha` is retained for call compatibility but ignored: emitted boxes
    are ALWAYS raw detector matches. A smoothed box lags fast motion — a
    redaction box trailing the face it must hide is a privacy leak. Smoothing
    belongs to Pass 2 (`trajectory.py`: One-Euro + safety envelope), which
    sees the whole history before any rendering.
    """
    del ema_alpha  # smoothing moved to Pass 2 — raw emission is the contract
    tracks: list[_TrackState] = []
    finished: list[_TrackState] = []
    next_id = 1
    tolerance = track_buffer if track_buffer is not None else max_age
    gmc_acc = (0.0, 0.0)

    for k, dets in enumerate(detections_per_frame):
        if gmc_vectors is not None and k < len(gmc_vectors):
            mv = gmc_vectors[k]
            gmc_acc = (gmc_acc[0] + mv[0], gmc_acc[1] + mv[1])
        t_s = t_per_frame[k]
        # candidates: live tracks matched by IoU, best first
        by_score = sorted(range(len(dets)), key=lambda i: dets[i][4], reverse=True)
        used: set[int] = set()
        for tr in tracks:
            best_i, best_iou = -1, iou_threshold
            delta = (gmc_acc[0] - tr.gmc[0], gmc_acc[1] - tr.gmc[1])
            pred = tr.predict(t_s, delta)
            for i in by_score:
                if i in used:
                    continue
                x, y, w, h, _ = dets[i]
                box = (float(x), float(y), float(x + w), float(y + h))
                iou = _iou_xyxy(pred, box)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0:
                used.add(best_i)
                x, y, w, h, score = dets[best_i]
                raw = (float(x), float(y), float(x + w), float(y + h))
                tr.update(raw, score, t_s, gmc_acc)
                tr.hits.append(tr._emit(t_s, score))
                if len(tr.hits) >= min_hits:
                    tr.confirmed = True
            else:
                tr.missed += 1

        # retire: missed longer than the buffer OR (unconfirmed and dead) —
        # keep confirmed tracks that still have a pulse in `tracks`
        still: list[_TrackState] = []
        for tr in tracks:
            if tr.missed > tolerance:
                finished.append(tr)
            else:
                still.append(tr)
        tracks = still

        for i in range(len(dets)):
            if i in used:
                continue
            x, y, w, h, score = dets[i]
            tr = _TrackState(
                next_id,
                (float(x), float(y), float(x + w), float(y + h)),
                score,
                t_s,
                gmc_acc,
            )
            next_id += 1
            tracks.append(tr)

    finished.extend(tracks)
    out: list[FaceTrack] = []
    for tr in finished:
        real_hits = tr.hits
        if not tr.confirmed or len(real_hits) < min_hits:
            continue
        out.append(FaceTrack(track_id=f"person_{tr.track_id}", hits=real_hits))
    # interpolation pass: fill stride gaps between consecutive real hits
    filled: list[FaceTrack] = []
    step = sample_step_s if sample_step_s else _median_t_step(t_per_frame)
    for ft in out:
        hits = ft.hits
        dense: list[TrackHit] = [hits[0]]
        for prev, cur in zip(hits, hits[1:], strict=False):
            dt = cur.t_s - prev.t_s
            steps = max(1, round(dt / step)) if step > 0 else 1
            for s in range(1, steps):
                frac = s / steps
                dense.append(
                    TrackHit(
                        t_s=round(prev.t_s + dt * frac, 3),
                        x=int(prev.x + (cur.x - prev.x) * frac),
                        y=int(prev.y + (cur.y - prev.y) * frac),
                        w=int(prev.w + (cur.w - prev.w) * frac),
                        h=int(prev.h + (cur.h - prev.h) * frac),
                        score=prev.score,
                        interpolated=True,
                    )
                )
            dense.append(cur)
        filled.append(FaceTrack(track_id=ft.track_id, hits=dense))
    return filled


def _median_t_step(t_per_frame: list[float]) -> float:
    """Median positive gap between sampled frame times (the sampling cadence)."""
    steps = [b - a for a, b in zip(t_per_frame, t_per_frame[1:], strict=False) if b > a]
    if not steps:
        return 1.0
    steps.sort()
    return steps[len(steps) // 2]


class EmptySweepError(RuntimeError):
    """Both face AND plate sweeps came back empty — the plan would be vacuous.

    A sweep that sees nothing is a broken sweep, not an empty scene (issue #14
    follow-up: the Eustis bake shipped ZERO redaction while the audit passed
    with 0 detections = 0 leaks). Privacy is not soft: refuse.
    """


def merge_tracks(*groups: list[FaceTrack]) -> list[FaceTrack]:
    """Concatenate detector groups (faces, plates, ...) into one id-unique list.

    Each sweep's tracker emits `person_{n}` ids independently, so a bare
    concatenation collides (`person_1` from faces AND plates). Only colliding
    ids are renamed (suffix `_2`, `_3`, ...); domain semantics — `plate_N` vs
    `person_N` — come from `merge_redaction_tracks`, which pre-prefixes plate
    ids so the two sweeps NEVER alias each other's objects. Order is preserved
    (faces first, plates second); the redaction plan, Studio exclusion toggles
    and the debug overlay all key on these ids.
    """
    merged: list[FaceTrack] = []
    seen: set[str] = set()
    for group in groups:
        for tr in group:
            tid = tr.track_id
            if tid in seen:
                n = 1
                while f"{tid}_{n}" in seen:
                    n += 1
                tid = f"{tid}_{n}"
            seen.add(tid)
            merged.append(FaceTrack(track_id=tid, hits=tr.hits))
    return merged


def merge_redaction_tracks(
    face_tracks: list[FaceTrack], plate_tracks: list[FaceTrack]
) -> list[FaceTrack]:
    """Face + plate tracks -> one redaction track list; EMPTY is an error.

    Plates get readable ids (`person_N` -> `plate_N`) so the Studio UI and the
    debug overlay distinguish a blurred face from a blurred plate. An empty
    MERGED list raises `EmptySweepError` — a face-less source is legitimate
    (plates may still be present), but nothing-detectable-everywhere is the
    broken-sweep signature we refuse to bake (issue #14 follow-up law).
    """
    prefixed_plates: list[FaceTrack] = []
    for tr in plate_tracks:
        tid = tr.track_id
        if tid.startswith("person_"):
            tid = f"plate_{tid[len('person_') :]}"
        prefixed_plates.append(FaceTrack(track_id=tid, hits=tr.hits))
    merged = merge_tracks(face_tracks, prefixed_plates)
    if not merged:
        raise EmptySweepError(
            "tracked redaction sweep found ZERO faces and ZERO plates — refusing to persist an "
            "empty plan (broken detector/model or genuinely empty scene; verify with "
            "scripts/fetch_yolov8n_face.py + scripts/fetch_yolov8n_plate.py "
            "+ a debug-overlay sweep)"
        )
    return merged


def tracks_to_regions(
    tracks: list[FaceTrack],
    *,
    pad_frac: float = 0.10,
    sigma: int = 24,
    frame_w: int = 1920,
    frame_h: int = 1080,
):
    """Tracked faces -> padded static BlurRegions (backward-compatible path).

    The existing renderer consumes static regions; each track's active span
    becomes one padded union box (track id preserved on `BlurRegion.track_id`
    so the tracked-mode renderer upgrade can key off it later).
    """
    from director_os.core.models import BlurRegion

    regions = []
    for tr in tracks:
        if not tr.hits:
            continue
        x0 = min(h.x for h in tr.hits)
        y0 = min(h.y for h in tr.hits)
        x1 = max(h.x + h.w for h in tr.hits)
        y1 = max(h.y + h.h for h in tr.hits)
        pad_w, pad_h = int((x1 - x0) * pad_frac), int((y1 - y0) * pad_frac)
        x0, y0 = max(0, x0 - pad_w), max(0, y0 - pad_h)
        x1, y1 = min(frame_w, x1 + pad_w), min(frame_h, y1 + pad_h)
        regions.append(
            BlurRegion(
                scope="global",
                bbox=BBox(x=x0, y=y0, w=max(8, x1 - x0), h=max(8, y1 - y0)),
                mode="static",
                sigma=sigma,
                track_id=tr.track_id,
            )
        )
    return regions
