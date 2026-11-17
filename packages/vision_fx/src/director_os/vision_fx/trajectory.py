"""Pass 2 — trajectory repair & zero-lag enveloping.

Never blur directly from raw per-frame detections. This pass consumes the
tracker's FaceTracks (sampled frames + interpolated stride fills) and emits a
DENSE per-frame redaction plan for the whole video, processed with full track
history BEFORE any rendering:

1. Convert to exact frame indices (CFR source => frame_idx = round(t_s * fps)).
2. Bridge gaps <= `max_bridge_frames` with linear interpolation; bridged
   boxes carry +15% size inflation (a bridged face may turn mid-gap).
3. Gaps > `max_bridge_frames` SPLIT the track's coverage into windows —
   never hold a stale box across an unbridgeable hole.
4. 5-frame lead-in before the first real detection and 5-frame lead-out
   after the last (detection fires late on enters / early on exits).
5. One-Euro filter per channel (center x/y, w, h) — velocity-aware, no EMA lag.
6. Envelope every box: scale_w=1.35 / scale_h=1.40 around the smoothed center,
   UNION the raw detection box so `padded >= raw` holds by construction.
"""

from __future__ import annotations

import math

from director_os.vision_fx.compute.base import FaceTrack
from director_os.vision_fx.one_euro import OneEuroFilter
from pydantic import BaseModel, Field

DEFAULT_MAX_BRIDGE_FRAMES = 25
DEFAULT_LEAD_FRAMES = 5
DEFAULT_SCALE_W = 1.35
DEFAULT_SCALE_H = 1.40
DEFAULT_GAP_INFLATE = 1.15

# One-Euro: heavy damping when slow, rides the signal when fast (beta).
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.05


class TrackedBox(BaseModel):
    """One frame's redaction box (padded envelope, RAW source pixels)."""

    frame_idx: int = Field(ge=0)
    t_s: float = Field(ge=0.0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    bridged: bool = False  # linear-interpolated across a <=max-bridge gap
    lead: bool = False  # lead-in / lead-out frame


class RedactionTrajectory(BaseModel):
    """One identity's dense per-frame boxes over its active window(s)."""

    track_id: str
    boxes: list[TrackedBox] = Field(default_factory=list)
    n_windows: int = 1

    @property
    def frame_start(self) -> int:
        return self.boxes[0].frame_idx if self.boxes else 0

    @property
    def frame_end(self) -> int:
        return self.boxes[-1].frame_idx if self.boxes else -1


class PerFrameRedaction(BaseModel):
    """The complete Pass-2 plan: every frame -> boxes to blur."""

    fps: int = Field(gt=0)
    n_frames: int = Field(ge=0)
    frame_w: int = Field(gt=0)
    frame_h: int = Field(gt=0)
    trajectories: list[RedactionTrajectory] = Field(default_factory=list)

    def boxes_at(self, frame_idx: int) -> list[TrackedBox]:
        out: list[TrackedBox] = []
        for tr in self.trajectories:
            if tr.frame_start <= frame_idx <= tr.frame_end:
                lo, hi = 0, len(tr.boxes) - 1
                # boxes are dense + sorted: binary search the exact frame
                while lo < hi:
                    mid = (lo + hi) // 2
                    if tr.boxes[mid].frame_idx < frame_idx:
                        lo = mid + 1
                    else:
                        hi = mid
                if tr.boxes[lo].frame_idx == frame_idx:
                    out.append(tr.boxes[lo])
        return out


def repair_trajectories(
    tracks: list[FaceTrack],
    *,
    fps: float,
    frame_w: int,
    frame_h: int,
    n_frames: int | None = None,
    max_bridge_frames: int = DEFAULT_MAX_BRIDGE_FRAMES,
    lead_frames: int = DEFAULT_LEAD_FRAMES,
    scale_w: float = DEFAULT_SCALE_W,
    scale_h: float = DEFAULT_SCALE_H,
    gap_inflate: float = DEFAULT_GAP_INFLATE,
    min_cutoff: float = ONE_EURO_MIN_CUTOFF,
    beta: float = ONE_EURO_BETA,
) -> PerFrameRedaction:
    """FaceTracks -> dense per-frame padded redaction plan (Pass 2)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    trajectories: list[RedactionTrajectory] = []
    for track in tracks:
        real = _real_hits_by_frame(track, fps)
        if not real:
            continue
        dense = _dense_frames(real, max_bridge_frames, gap_inflate, lead_frames)
        smoothed = _one_euro_smooth(dense, min_cutoff, beta)
        padded = [
            _envelope(b, raw, scale_w, scale_h, frame_w, frame_h)
            for b, raw in zip(smoothed, dense, strict=True)
        ]
        boxes = [
            TrackedBox(
                frame_idx=f["frame_idx"],
                t_s=round(f["frame_idx"] / fps, 3),
                x=p[0],
                y=p[1],
                w=p[2],
                h=p[3],
                bridged=bool(f["bridged"]),
                lead=bool(f["lead"]),
            )
            for f, p in zip(dense, padded, strict=True)
        ]
        n_windows = _count_windows(real, max_bridge_frames, lead_frames)
        trajectories.append(
            RedactionTrajectory(track_id=track.track_id, boxes=boxes, n_windows=n_windows)
        )
    trajectories.sort(key=lambda t: (t.frame_start, t.track_id))
    total = (
        n_frames
        if n_frames is not None
        else max((t.frame_end for t in trajectories), default=-1) + 1
    )
    return PerFrameRedaction(
        fps=int(round(fps)),
        n_frames=max(0, total),
        frame_w=frame_w,
        frame_h=frame_h,
        trajectories=trajectories,
    )


def tracks_to_per_frame(
    tracks: list[FaceTrack],
    *,
    fps: float,
    frame_w: int,
    frame_h: int,
    n_frames: int,
) -> PerFrameRedaction:
    """Backward-compat alias with explicit duration (worker seam calls this)."""
    return repair_trajectories(tracks, fps=fps, frame_w=frame_w, frame_h=frame_h, n_frames=n_frames)


# --------------------------------------------------------------------- steps


def _real_hits_by_frame(track: FaceTrack, fps: float) -> list[dict]:
    """Real detections -> [{frame_idx, x, y, w, h}] sorted by frame."""
    seen: dict[int, dict] = {}
    for h in track.hits:
        if h.interpolated:
            continue
        fi = int(round(h.t_s * fps))
        seen.setdefault(fi, {"frame_idx": fi, "x": h.x, "y": h.y, "w": h.w, "h": h.h})
    return [seen[k] for k in sorted(seen)]


def _dense_frames(
    real: list[dict],
    max_bridge: int,
    gap_inflate: float,
    lead: int,
) -> list[dict]:
    """Real detections + bridged gaps + lead-in/out -> dense per-frame dicts."""
    out: list[dict] = []
    for prev, cur in zip(real, real[1:], strict=False):
        out.append({**prev, "bridged": False, "lead": False})
        gap = cur["frame_idx"] - prev["frame_idx"]
        if 1 < gap <= max_bridge:
            for s in range(1, gap):
                frac = s / gap
                inflate = gap_inflate if gap_inflate > 1.0 else 1.0
                out.append(_interp_frame(prev, cur, frac, bridged=True, inflate=inflate))
    out.append({**real[-1], "bridged": False, "lead": False})
    if lead > 0:
        first = real[0]
        last_fi = -1
        for i in range(lead, 0, -1):
            fi = max(0, first["frame_idx"] - i)
            if fi >= first["frame_idx"] or fi <= last_fi:
                continue  # clamped onto the first real frame (or an earlier lead) — skip
            out.insert(0, {**first, "frame_idx": fi, "bridged": False, "lead": True})
            last_fi = fi
        last = out[-1]
        for i in range(1, lead + 1):
            out.append({**last, "frame_idx": last["frame_idx"] + i, "bridged": False, "lead": True})
    return out


def _interp_frame(a: dict, b: dict, frac: float, *, bridged: bool, inflate: float = 1.0) -> dict:
    """Linear box interpolation a->b at `frac`, optionally size-inflated."""
    cx = (a["x"] + a["w"] / 2) + ((b["x"] + b["w"] / 2) - (a["x"] + a["w"] / 2)) * frac
    cy = (a["y"] + a["h"] / 2) + ((b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)) * frac
    w = (a["w"] + (b["w"] - a["w"]) * frac) * inflate
    h = (a["h"] + (b["h"] - a["h"]) * frac) * inflate
    return {
        "frame_idx": round(a["frame_idx"] + (b["frame_idx"] - a["frame_idx"]) * frac),
        "x": cx - w / 2,
        "y": cy - h / 2,
        "w": w,
        "h": h,
        "bridged": bridged,
        "lead": False,
    }


def _count_windows(real: list[dict], max_bridge: int, lead: int) -> int:
    """Coverage windows = real-detection runs separated by >max_bridge gaps."""
    windows = 1
    for prev, cur in zip(real, real[1:], strict=False):
        if cur["frame_idx"] - prev["frame_idx"] > max_bridge:
            windows += 1
    return windows


def _one_euro_smooth(dense: list[dict], min_cutoff: float, beta: float) -> list[dict]:
    """One-Euro each channel independently; window starts reset the filter."""
    fx = OneEuroFilter(freq=30.0, min_cutoff=min_cutoff, beta=beta)
    fy = OneEuroFilter(freq=30.0, min_cutoff=min_cutoff, beta=beta)
    fw = OneEuroFilter(freq=30.0, min_cutoff=min_cutoff, beta=beta)
    fh = OneEuroFilter(freq=30.0, min_cutoff=min_cutoff, beta=beta)
    out: list[dict] = []
    prev_fi: int | None = None
    for f in dense:
        if prev_fi is not None and f["frame_idx"] != prev_fi + 1:
            # window split (unbridgeable gap or lead jump): re-prime the filters
            fx.reset()
            fy.reset()
            fw.reset()
            fh.reset()
        sx, sy = fx(f["x"]), fy(f["y"])
        sw, sh = fw(f["w"]), fh(f["h"])
        out.append({**f, "x": sx, "y": sy, "w": max(4.0, sw), "h": max(4.0, sh)})
        prev_fi = f["frame_idx"]
    return out


def _envelope(
    f: dict,
    raw: dict,
    scale_w: float,
    scale_h: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Padded box: scale around smoothed center, UNION raw box, clamp to frame.

    The union guarantees the padded box always covers the raw detection — the
    One-Euro output may hug the signal tightly, but redaction must never sit
    inside the truth.
    """
    sx, sy, sw, sh = f["x"], f["y"], f["w"], f["h"]
    px, py = sw * scale_w, sh * scale_h
    p_x0, p_y0 = sx - (px - sw) / 2.0, sy - (py - sh) / 2.0
    p_x1, p_y1 = p_x0 + px, p_y0 + py
    r_x1 = raw["x"] + raw["w"]
    r_y1 = raw["y"] + raw["h"]
    x0 = min(p_x0, float(raw["x"]))
    y0 = min(p_y0, float(raw["y"]))
    x1 = max(p_x1, float(r_x1))
    y1 = max(p_y1, float(r_y1))
    x0i = max(0, math.floor(x0))
    y0i = max(0, math.floor(y0))
    x1i = min(frame_w, math.ceil(x1))
    y1i = min(frame_h, math.ceil(y1))
    if x1i - x0i < 4 or y1i - y0i < 4:
        x1i, y1i = min(frame_w, x0i + 8), min(frame_h, y0i + 8)
    return x0i, y0i, x1i - x0i, y1i - y0i
