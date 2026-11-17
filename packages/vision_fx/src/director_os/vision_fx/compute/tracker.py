"""IoU tracker with stride-3 sampling + linear interpolation — persistent IDs.

ByteTrack-class behavior without the dependency: greedy IoU matching on
sampled frames, `min_hits` confirmation (one-shot false positives never earn
a track), `max_age` tolerance, then per-track linear interpolation across
the stride gaps + light EMA smoothing, so the redaction box glides instead
of strobing. This is the host-side half of BOTH local and cloud backends:
Kaggle/Modal kernels dump raw per-frame detections and re-attach them here,
keeping tracker logic in exactly one place.
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
        "smooth",
        "score",
        "hits",
        "missed",
        "confirmed",
        "last_t",
        "vel",
    )

    def __init__(
        self, track_id: int, xyxy: tuple[float, float, float, float], score: float, t_s: float
    ):
        self.track_id = track_id
        self.xyxy = xyxy
        # emitted coordinates ride an EMA so the redaction box glides instead
        # of strobing — but the MATCHING state stays raw (an EMA'd tracker box
        # lags fast motion and IoU-collapses on the very next frame)
        self.smooth = xyxy
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
        self.vel = (0.0, 0.0)  # constant-velocity model, px/s (Kalman-lite)

    def predict(self, t_s: float) -> tuple[float, float, float, float]:
        """Box extrapolated to `t_s` by the constant-velocity model — what IoU
        matching sees. Bridges detector dropouts across stride gaps."""
        dt = t_s - self.last_t
        cx = (self.xyxy[0] + self.xyxy[2]) / 2 + self.vel[0] * dt
        cy = (self.xyxy[1] + self.xyxy[3]) / 2 + self.vel[1] * dt
        w, h = self.xyxy[2] - self.xyxy[0], self.xyxy[3] - self.xyxy[1]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def update(self, xyxy: tuple[float, float, float, float], score: float, t_s: float) -> None:
        dt = t_s - self.last_t
        if dt > 0:
            pcx = (self.xyxy[0] + self.xyxy[2]) / 2
            pcy = (self.xyxy[1] + self.xyxy[3]) / 2
            ncx = (xyxy[0] + xyxy[2]) / 2
            ncy = (xyxy[1] + xyxy[3]) / 2
            self.vel = ((ncx - pcx) / dt, (ncy - pcy) / dt)
        self.xyxy = xyxy
        self.last_t = t_s
        self.score = score
        self.missed = 0

    def _emit(self, t_s: float, score: float, ema_alpha: float) -> TrackHit:
        sm: tuple[float, float, float, float] = (
            self.smooth[0] * (1 - ema_alpha) + self.xyxy[0] * ema_alpha,
            self.smooth[1] * (1 - ema_alpha) + self.xyxy[1] * ema_alpha,
            self.smooth[2] * (1 - ema_alpha) + self.xyxy[2] * ema_alpha,
            self.smooth[3] * (1 - ema_alpha) + self.xyxy[3] * ema_alpha,
        )
        self.smooth = sm
        return TrackHit(
            t_s=t_s,
            x=int(sm[0]),
            y=int(sm[1]),
            w=int(sm[2] - sm[0]),
            h=int(sm[3] - sm[1]),
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
) -> list[FaceTrack]:
    """Greedy-IoU tracking over sampled frames -> interpolated FaceTracks.

    `detections_per_frame[k]` holds [(x, y, w, h, score)] at time
    `t_per_frame[k]`; frames must be ordered. `sample_step_s` is the sampling
    cadence (defaults to the median positive diff of `t_per_frame`) — stride
    gaps larger than one step are filled by linear interpolation. Tracks
    below `min_hits` real detections are dropped.

    `ema_alpha` defaults to 1.0 (emit raw detector boxes): interpolated fills
    already glide, and a smoothed box LAGS fast motion — a redaction box
    trailing the face it must hide is a privacy leak. Raise smoothing only
    for cosmetic overlays, never for redaction.
    """
    tracks: list[_TrackState] = []
    finished: list[_TrackState] = []
    next_id = 1

    for k, dets in enumerate(detections_per_frame):
        t_s = t_per_frame[k]
        # candidates: live tracks matched by IoU, best first
        by_score = sorted(range(len(dets)), key=lambda i: dets[i][4], reverse=True)
        used: set[int] = set()
        for tr in tracks:
            best_i, best_iou = -1, iou_threshold
            pred = tr.predict(t_s)
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
                tr.update(raw, score, t_s)
                tr.hits.append(tr._emit(t_s, score, ema_alpha))
                if len(tr.hits) >= min_hits:
                    tr.confirmed = True
            else:
                tr.missed += 1

        # retire: missed too long OR (unconfirmed and dead) — keep confirmed
        # tracks that still have a pulse in `tracks`, retire the rest
        still: list[_TrackState] = []
        for tr in tracks:
            if tr.missed > max_age:
                finished.append(tr)
            else:
                still.append(tr)
        tracks = still

        for i in range(len(dets)):
            if i in used:
                continue
            x, y, w, h, score = dets[i]
            tr = _TrackState(next_id, (float(x), float(y), float(x + w), float(y + h)), score, t_s)
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
