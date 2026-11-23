"""Promptable selection service — box/click on frame X -> tracked selection.

Agent-frame-editor roadmap step 3 (docs/research/agent-frame-editor.md §4):
the editor — human OR agent — selects an object ON a frame (a drawn box or a
click) and the service follows that object across a frame window, emitting
per-frame boxes the blur op consumes (mode="tracked"). SAM 2 / EdgeTAM is the
planned heavy backend (Meta's promptable video segmentation: a click, box or
mask prompt selects an object and a memory bank tracks it through occlusion);
until weights ship locally, the CPU default is:

- ``ncc-cpu``        zero-download template tracker — normalized
                     cross-correlation (cv2.matchTemplate TM_CCOEFF_NORMED,
                     pure-numpy ZNCC fallback) around the previous position,
                     forward AND backward from the anchor, template blending
                     for appearance drift, miss -> linear interpolation up to
                     a bounded gap, then the selection honestly ends (a
                     tracker that hallucinates boxes across occlusions leaks
                     privacy).
- ``detector-assist`` when detector sweeps already ran (the pipeline's
                     FaceTracks), pick the IoU-best track at the anchor —
                     detector boxes beat template drift wherever they exist.

Coordinates: prompts and hits are RAW SOURCE pixels — the same canvas the
renderer, Studio monitor and FaceTracks already speak (one canvas law).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from director_os.vision_fx.compute.base import FaceTrack, TrackHit
from pydantic import BaseModel, Field

#: decode width for template tracking (objects, not portraits — 480px keeps
#: ZNCC well under a millisecond per frame on CPU while retaining detail)
TRACK_WIDTH = 480
#: a template with less gray-level variation than this is UNTRACKABLE —
#: NCC of a zero-variance patch is degenerate (cv2 reports 1.0 anywhere)
TEMPLATE_STD_MIN = 2.0
#: peak NCC below this is a miss (0.25 — slow drift tolerated, wrong object refused)
NCC_MIN_SCORE = 0.25
#: template refresh blend (new patch weight when the match is strong)
TEMPLATE_BLEND = 0.15
#: template refresh only above this score (refreshing on a bad match poisons it)
TEMPLATE_REFRESH_MIN = 0.6
#: search radius around the previous position, px at TRACK_WIDTH
SEARCH_RADIUS_PX = 48
#: max consecutive miss frames bridged by interpolation (bounds hallucination)
MAX_INTERP_GAP_FRAMES = 12
#: a selection shorter than this many real matches is not a selection
MIN_REAL_HITS = 2
#: API guard — selection is interactive; a 60 s span decode is a job, not a click
MAX_SPAN_S = 60.0


class SelectionNotAvailable(RuntimeError):
    """A requested backend cannot run here — message says which flip enables it."""


class SelectionError(RuntimeError):
    """Selection failed on valid input (no object there, media unreadable)."""


class SelectionPrompt(BaseModel):
    """What the editor pointed at: a box (drawn) or a point (clicked)."""

    kind: str = Field(pattern="^(box|point)$")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)
    anchor_frame: int = Field(ge=0)

    @property
    def is_box(self) -> bool:
        return self.kind == "box"


class TrackedSelection(BaseModel):
    """The follow: per-frame boxes in RAW source pixels + provenance."""

    asset_id: str = ""
    backend: str
    prompt: dict[str, Any] = Field(default_factory=dict)
    anchor_frame: int = 0
    fps: float = 30.0
    frame_start: int = 0
    frame_end: int = 0
    frame_w: int = 0
    frame_h: int = 0
    hits: list[TrackHit] = Field(default_factory=list)
    n_interpolated: int = 0
    mean_score: float = 0.0

    @property
    def span_s(self) -> float:
        if not self.hits:
            return 0.0
        return max(0.0, self.hits[-1].t_s - self.hits[0].t_s)

    def union_box(self) -> dict[str, int]:
        """Padded full-span union box — the backward-compatible static region.

        The shipped render path consumes static regions; a tracked selection
        mirrors its union (10% pad, same law as tracks_to_regions) so privacy
        coverage exists TODAY, with per-frame baking landing in step 4.
        """
        if not self.hits:
            raise SelectionError("empty selection has no union box")
        x0 = min(h.x for h in self.hits)
        y0 = min(h.y for h in self.hits)
        x1 = max(h.x + h.w for h in self.hits)
        y1 = max(h.y + h.h for h in self.hits)
        pad_w, pad_h = int((x1 - x0) * 0.10), int((y1 - y0) * 0.10)
        x0, y0 = max(0, x0 - pad_w), max(0, y0 - pad_h)
        x1, y1 = min(self.frame_w, x1 + pad_w), min(self.frame_h, y1 + pad_h)
        return {"x": int(x0), "y": int(y0), "w": max(8, int(x1 - x0)), "h": max(8, int(y1 - y0))}

    def to_params(self) -> dict[str, Any]:
        """Blur-op params payload — tracked hits ride the op (step 4 reads them)."""
        return {
            "mode": "tracked",
            "x": self.hits[len(self.hits) // 2].x if self.hits else 0,
            "y": self.hits[len(self.hits) // 2].y if self.hits else 0,
            "w": self.hits[len(self.hits) // 2].w if self.hits else 0,
            "h": self.hits[len(self.hits) // 2].h if self.hits else 0,
            "sigma": 24,
            "tracked": [h.model_dump() for h in self.hits],
            "tracked_backend": self.backend,
            "anchor_frame": self.anchor_frame,
            "n_tracked": len(self.hits),
            "n_interpolated": self.n_interpolated,
            "mean_score": round(self.mean_score, 3),
        }


def probe_video(path: str | Path) -> tuple[float, float, int, int]:
    """(duration_s, fps, width, height) — ffprobe, guarded."""
    import json as _json

    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )  # noqa: S603
    try:
        d = _json.loads(r.stdout or "{}")
        fmt = float(d["format"]["duration"])
        st = d["streams"][0]
        rate = st.get("r_frame_rate", "30/1")
        num, den = rate.split("/", 1) if "/" in rate else (rate, "1")
        fps = float(num) / float(den) if float(den) else 30.0
        return fmt, fps, int(st["width"]), int(st["height"])
    except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as e:
        raise SelectionError(f"cannot probe video {path}: {e}") from e


def _decode_span_gray(
    path: str | Path,
    t_start: float,
    duration_s: float,
    width: int,
    height: int,
) -> bytes:
    """Decode [t_start, t_start+duration) as grayscale frames (raw bytes).

    One sequential decode (cheap: short span, low width). Dimensions are
    EXPLICIT (`scale=WxH`) — raw gray video has no container, so the geometry
    must never be inferred: an off-by-two guess reshapes garbage.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t_start:.3f}",
        "-i",
        str(path),
        "-t",
        f"{duration_s:.3f}",
        "-vf",
        f"scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)  # noqa: S603
    except subprocess.TimeoutExpired as e:
        raise SelectionError(f"span decode timed out ({duration_s:.1f}s span)") from e
    if r.returncode != 0 or not r.stdout:
        raise SelectionError(f"span decode failed: {r.stderr[-200:].decode(errors='replace')}")
    return r.stdout


# ------------------------------------------------------------------ NCC core


def _zncc_numpy(img: Any, tmpl: Any) -> tuple[float, int, int]:
    """Pure-numpy zero-mean NCC — fallback when cv2 is absent (CI-safe)."""
    import numpy as np

    th, tw = tmpl.shape
    t = tmpl.astype("float32")
    t_mean = float(t.mean())
    t_z = t - t_mean
    t_norm = float(np.sqrt((t_z * t_z).sum()))
    if t_norm < 1e-6:
        return 0.0, 0, 0
    ih, iw = img.shape
    if ih < th or iw < tw:
        return 0.0, 0, 0
    # integral images for local sums (fast sliding normalization)
    i = img.astype("float32")
    ii = np.cumsum(np.cumsum(i, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    win_sum = ii[th:, tw:] - ii[:-th, tw:] - ii[th:, :-tw] + ii[:-th, :-tw]
    win_mean = win_sum / (th * tw)
    best, bx, by = -1.0, 0, 0
    # stride-2 scan then refine stride-1 around the winner (2x faster, same result)
    for stride, refine in ((2, True), (1, False)):
        ys = range(0, ih - th + 1, stride)
        xs = range(0, iw - tw + 1, stride)
        for yy in ys:
            for xx in xs:
                patch = i[yy : yy + th, xx : xx + tw]
                p_z = patch - win_mean[yy, xx]
                denom = float(np.sqrt((p_z * p_z).sum()) * t_norm)
                if denom < 1e-6:
                    continue
                score = float((p_z * t_z).sum() / denom)
                if score > best:
                    best, bx, by = score, xx, yy
        if not refine:
            break
        # refine pass: dense around winner
        y0, y1 = max(0, by - 2), min(ih - th, by + 2) + 1
        x0, x1 = max(0, bx - 2), min(iw - tw, bx + 2) + 1
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                patch = i[yy : yy + th, xx : xx + tw]
                p_z = patch - win_mean[yy, xx]
                denom = float(np.sqrt((p_z * p_z).sum()) * t_norm)
                if denom < 1e-6:
                    continue
                score = float((p_z * t_z).sum() / denom)
                if score > best:
                    best, bx, by = score, xx, yy
    return best, bx, by


def _ncc_score(img: Any, tmpl: Any) -> tuple[float, int, int]:
    """Peak NCC + top-left position via cv2 when available, numpy otherwise."""
    try:
        import cv2
    except ImportError:
        return _zncc_numpy(img, tmpl)
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    _min_v, _max_v, _min_l, max_l = cv2.minMaxLoc(res)
    return float(_max_v), int(max_l[0]), int(max_l[1])


def _clip_box(
    x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    x = max(0, min(frame_w - 1, x))
    y = max(0, min(frame_h - 1, y))
    w = max(4, min(frame_w - x, w))
    h = max(4, min(frame_h - y, h))
    return x, y, w, h


class _NccTracker:
    """Forward/backward template tracker around an anchor frame."""

    def __init__(
        self,
        frames: Any,
        times: list[float],
        *,
        search_radius: int = SEARCH_RADIUS_PX,
    ) -> None:
        self.frames = frames
        self.times = times
        self.search_radius = search_radius
        self.n = len(frames)

    def track(self, anchor_idx: int, box: tuple[int, int, int, int]) -> list[TrackHit]:

        fx, fy, fw, fh = box
        tmpl = self.frames[anchor_idx][fy : fy + fh, fx : fx + fw].copy()
        if float(tmpl.std()) < TEMPLATE_STD_MIN:
            raise SelectionError(
                "prompt sits on a texture-free region — normalized correlation has "
                "nothing to lock onto; aim at an object with visible structure"
            )
        hits: dict[int, tuple[int, int, int, int, float]] = {anchor_idx: (fx, fy, fw, fh, 1.0)}

        def walk(start: int, stop: int, step: int) -> None:
            tmpl_local = tmpl.copy()
            px, py = fx, fy
            miss_run = 0
            for k in range(start, stop, step):
                if miss_run > MAX_INTERP_GAP_FRAMES:
                    break
                frame = self.frames[k]
                fh_, fw_ = tmpl_local.shape
                x0 = max(0, px - self.search_radius)
                y0 = max(0, py - self.search_radius)
                x1 = min(frame.shape[1], px + fw_ + self.search_radius)
                y1 = min(frame.shape[0], py + fh_ + self.search_radius)
                region = frame[y0:y1, x0:x1]
                if region.shape[0] < fh_ or region.shape[1] < fw_:
                    miss_run += 1
                    continue
                score, rx, ry = _ncc_score(region, tmpl_local)
                if score < NCC_MIN_SCORE:
                    miss_run += 1
                    continue
                miss_run = 0
                nx, ny = x0 + rx, y0 + ry
                hits[k] = (nx, ny, fw_, fh_, max(0.0, min(1.0, score)))
                if score > TEMPLATE_REFRESH_MIN:
                    patch = frame[ny : ny + fh_, nx : nx + fw_].astype("float32")
                    blended = (1.0 - TEMPLATE_BLEND) * tmpl_local.astype(
                        "float32"
                    ) + TEMPLATE_BLEND * patch
                    tmpl_local = blended.astype("uint8")
                px, py = nx, ny

        walk(anchor_idx + 1, self.n, 1)
        walk(anchor_idx - 1, -1, -1)
        # assemble ordered hits with bounded-gap interpolation
        ordered: list[TrackHit] = []
        keys = sorted(hits)
        real_scores = [hits[k][4] for k in keys if k != anchor_idx]
        for k in keys:
            hx, hy, hw, hh, sc = hits[k]
            ordered.append(
                TrackHit(
                    t_s=round(self.times[k], 3),
                    x=int(hx),
                    y=int(hy),
                    w=int(hw),
                    h=int(hh),
                    score=round(float(sc), 3),
                    interpolated=False,
                )
            )
        # fill gaps (miss_run gaps) between consecutive real hits
        filled: list[TrackHit] = []
        for a, b in zip(ordered, ordered[1:], strict=False):
            filled.append(a)
            ka = self._frame_of(a.t_s)
            kb = self._frame_of(b.t_s)
            gap = kb - ka
            if 1 < gap <= MAX_INTERP_GAP_FRAMES + 1:
                for s in range(1, gap):
                    frac = s / gap
                    filled.append(
                        TrackHit(
                            t_s=round(self.times[ka + s], 3),
                            x=int(a.x + (b.x - a.x) * frac),
                            y=int(a.y + (b.y - a.y) * frac),
                            w=int(a.w + (b.w - a.w) * frac),
                            h=int(a.h + (b.h - a.h) * frac),
                            score=a.score,
                            interpolated=True,
                        )
                    )
        if ordered:
            filled.append(ordered[-1])
        if real_scores:
            self.last_mean_score = sum(real_scores) / len(real_scores)
        else:
            self.last_mean_score = 0.0
        return filled

    def _frame_of(self, t_s: float) -> int:
        best_i, best_d = 0, 1e9
        for i, t in enumerate(self.times):
            d = abs(t - t_s)
            if d < best_d:
                best_i, best_d = i, d
        return best_i


def track_selection_ncc(
    video_path: str | Path,
    prompt: SelectionPrompt,
    *,
    frame_start: int,
    frame_end: int,
    fps: float,
    frame_w: int,
    frame_h: int,
    track_width: int = TRACK_WIDTH,
) -> TrackedSelection:
    """CPU template tracker: prompt box on the anchor frame -> span-wide boxes.

    Decodes ONLY the span once at `track_width`, tracks forward+backward from
    the anchor, interpolates bounded miss gaps, and reports honestly: fewer
    than MIN_REAL_HITS real matches raises (no object there — refuse, never
    fabricate). Boxes return in RAW SOURCE pixels (scaled from the decode).
    """
    import numpy as np

    if frame_end < frame_start:
        raise SelectionError(f"frame_end {frame_end} < frame_start {frame_start}")
    if fps <= 0:
        raise SelectionError(f"invalid fps {fps}")
    t0 = frame_start / fps
    t1 = (frame_end + 1) / fps
    span_s = min(t1 - t0, MAX_SPAN_S)
    anchor_t = prompt.anchor_frame / fps
    if not (t0 - 1e-6 <= anchor_t <= t0 + span_s + 1e-6):
        raise SelectionError("anchor frame outside the requested span")

    # decode geometry EXPLICIT (even dims — yuv420 encoders require it)
    dec_w = min(track_width, frame_w) or track_width
    dec_h = max(2, int(round(frame_h * dec_w / max(1, frame_w) / 2)) * 2)
    raw = _decode_span_gray(video_path, t0, span_s, dec_w, dec_h)
    per_frame = dec_w * dec_h
    if len(raw) < per_frame:
        raise SelectionError("span decode produced zero frames")
    n = len(raw) // per_frame
    frames = np.frombuffer(raw[: n * per_frame], dtype=np.uint8).reshape(n, dec_h, dec_w)
    times = [t0 + i / fps for i in range(n)]
    anchor_idx = min(n - 1, max(0, prompt.anchor_frame - frame_start))

    # decode-scale -> source-scale factor
    sx, sy = frame_w / dec_w, frame_h / dec_h

    if prompt.is_box:
        if prompt.w <= 0 or prompt.h <= 0:
            raise SelectionError("box prompt needs w and h")
        bx = int(prompt.x / sx)
        by = int(prompt.y / sy)
        bw = max(4, int(prompt.w / sx))
        bh = max(4, int(prompt.h / sy))
    else:
        # point prompt: default patch centered on the click (1/8th of frame)
        bw = max(8, dec_w // 8)
        bh = max(8, dec_h // 8)
        bx = int(prompt.x / sx - bw / 2)
        by = int(prompt.y / sy - bh / 2)
    box = _clip_box(bx, by, bw, bh, dec_w, dec_h)

    tracker = _NccTracker(frames, times)
    hits_dec = tracker.track(anchor_idx, box)
    real = [h for h in hits_dec if not h.interpolated]
    if len(real) < MIN_REAL_HITS:
        raise SelectionError(
            f"selection lost after {len(real)} real frame(s) — no stable object at "
            f"frame {prompt.anchor_frame}; adjust the box or the span"
        )
    scaled = [
        TrackHit(
            t_s=h.t_s,
            x=int(h.x * sx),
            y=int(h.y * sy),
            w=max(4, int(h.w * sx)),
            h=max(4, int(h.h * sy)),
            score=h.score,
            interpolated=h.interpolated,
        )
        for h in hits_dec
    ]
    sel = TrackedSelection(
        backend="ncc-cpu",
        prompt=prompt.model_dump(),
        anchor_frame=prompt.anchor_frame,
        fps=fps,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_w=frame_w,
        frame_h=frame_h,
        hits=scaled,
        n_interpolated=sum(1 for h in scaled if h.interpolated),
        mean_score=round(tracker.last_mean_score, 3),
    )
    return sel


def select_from_tracks(
    tracks: list[FaceTrack],
    prompt: SelectionPrompt,
    *,
    frame_start: int,
    frame_end: int,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> TrackedSelection:
    """Detector-assist selection: IoU-best track at the anchor wins.

    Detector boxes beat template drift wherever a sweep already ran — this is
    the same canvas the pipeline's redaction tracks live on, so picking a
    track is exact (no re-decode at all). Raises when no track overlaps the
    prompt at the anchor (the honest refusal: wrong click, wrong object).
    """
    from director_os.vision_fx.compute.tracker import _iou_xyxy

    if not tracks:
        raise SelectionError("no detector tracks available for detector-assist selection")
    anchor_t = prompt.anchor_frame / fps
    if prompt.is_box:
        pbox = (
            float(prompt.x),
            float(prompt.y),
            float(prompt.x + prompt.w),
            float(prompt.y + prompt.h),
        )
    else:
        pbox = (float(prompt.x - 8), float(prompt.y - 8), float(prompt.x + 8), float(prompt.y + 8))
    best_tr, best_iou = None, 0.10  # minimum meaningful overlap
    for tr in tracks:
        nearest = min(tr.hits, key=lambda h: abs(h.t_s - anchor_t))
        if abs(nearest.t_s - anchor_t) > max(1.0, 2.0 / fps) * 2:
            continue
        hbox = (
            float(nearest.x),
            float(nearest.y),
            float(nearest.x + nearest.w),
            float(nearest.y + nearest.h),
        )
        iou = _iou_xyxy(pbox, hbox)
        if iou > best_iou:
            best_tr, best_iou = tr, iou
    if best_tr is None:
        raise SelectionError(
            "no detector track overlaps the prompt at the anchor frame — "
            "the clicked object is not a tracked face/plate; falling back is the caller's call"
        )
    lo, hi = frame_start / fps, (frame_end + 1) / fps
    hits = [h for h in best_tr.hits if lo - 1e-6 <= h.t_s < hi + 1e-6]
    if len(hits) < MIN_REAL_HITS:
        raise SelectionError(f"track {best_tr.track_id} has only {len(hits)} hits inside the span")
    return TrackedSelection(
        backend=f"detector-assist:{best_tr.track_id}",
        prompt=prompt.model_dump(),
        anchor_frame=prompt.anchor_frame,
        fps=fps,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_w=frame_w,
        frame_h=frame_h,
        hits=hits,
        n_interpolated=sum(1 for h in hits if h.interpolated),
        mean_score=(sum(h.score for h in hits if not h.interpolated) / max(1, len(hits)))
        if hits
        else 0.0,
    )


# ------------------------------------------------------------------ backends


class Sam2ServiceBackend:
    """SAM 2 over the local weight service (gpu_services/sam2, env-gated).

    The service boundary keeps torch OUT of the default path: the weight
    service container owns the checkpoint (named volume, never re-downloaded)
    and this port is a plain httpx call. `SAM2_SERVICE_URL=http://sam2:8300`
    switches it on; the response maps 1:1 onto TrackedSelection hits (same
    RAW-pixel boxes, same interpolation flags), so downstream blur/redaction
    code cannot tell which backend produced the follow.
    """

    name = "sam2"

    def __init__(self, base_url: str | None = None, transport: Any = None):
        self.base_url = (base_url or os.getenv("SAM2_SERVICE_URL") or "").rstrip("/")
        self._transport = transport  # tests inject httpx.MockTransport here

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("SAM2_SERVICE_URL", ""))

    def track(
        self,
        video_path: str | Path,
        prompt: SelectionPrompt,
        *,
        frame_start: int,
        frame_end: int,
        fps: float,
        frame_w: int,
        frame_h: int,
        **kw: Any,
    ) -> TrackedSelection:
        if not self.base_url:
            raise SelectionNotAvailable(
                "sam2 service backend unavailable: set SAM2_SERVICE_URL "
                "(docker compose --profile gpu up sam2) — ncc-cpu stays the default"
            )
        import httpx

        payload = {
            "video_path": str(video_path),
            "prompt": prompt.model_dump(),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "fps": fps,
            "frame_w": frame_w,
            "frame_h": frame_h,
        }
        try:
            if self._transport is not None:  # tests: httpx.MockTransport injection
                with httpx.Client(transport=self._transport, timeout=600.0) as client:
                    r = client.post(f"{self.base_url}/track", json=payload)
            else:
                r = httpx.post(f"{self.base_url}/track", json=payload, timeout=600.0)
        except httpx.HTTPError as e:
            raise SelectionNotAvailable(f"sam2 service unreachable: {e}") from e
        if r.status_code == 503:
            raise SelectionNotAvailable(f"sam2 service not ready: {r.json().get('detail')}")
        if r.status_code == 404:
            raise SelectionError(f"sam2 service cannot see the video: {r.json().get('detail')}")
        if r.status_code != 200:
            raise SelectionError(f"sam2 service failed ({r.status_code}): {r.text[:300]}")
        data = r.json()
        return TrackedSelection(
            backend="sam2",
            prompt=prompt.model_dump(),
            anchor_frame=prompt.anchor_frame,
            fps=fps,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_w=frame_w,
            frame_h=frame_h,
            hits=[TrackHit(**h) for h in data.get("hits", [])],
            n_interpolated=int(data.get("n_interpolated", 0)),
            mean_score=float(data.get("mean_score", 0.0)),
        )


class Sam2SelectionBackend:
    """SAM 2 in-process (OPTIONAL, env-gated): needs the ultralytics extra
    AND the checkpoint file. Prefer Sam2ServiceBackend in deployments —
    this path exists for a beefy workstation that runs everything local.
    """

    name = "sam2"

    @staticmethod
    def available() -> bool:
        import importlib.util

        return importlib.util.find_spec("ultralytics") is not None

    def track(self, video_path: str | Path, prompt: SelectionPrompt, **kw: Any) -> TrackedSelection:
        if not self.available():
            raise SelectionNotAvailable(
                "sam2 backend unavailable: install the optional extra "
                "(uv sync --extra sam2) and set SELECTION_BACKEND=sam2; "
                "the ncc-cpu backend is the always-available default"
            )
        weights = os.getenv("SAM2_WEIGHTS", "")
        if not weights or not Path(weights).exists():
            raise SelectionNotAvailable(
                "sam2 weights not configured locally (download sam2.1_hiera_tiny.pt and set "
                "SAM2_WEIGHTS — or set SAM2_SERVICE_URL to use the weight service); "
                "ncc-cpu covers selection until then"
            )
        raise SelectionNotAvailable(
            "in-process SAM2 tracking lands with the ultralytics video-predictor "
            "wiring (roadmap M4); the weight service backend (SAM2_SERVICE_URL) "
            "is the supported path today"
        )


def available_selection_backends() -> dict[str, bool]:
    return {
        "ncc-cpu": True,
        "detector-assist": True,
        "sam2": Sam2ServiceBackend.available() or Sam2SelectionBackend.available(),
    }


def run_selection(
    video_path: str | Path,
    prompt: SelectionPrompt,
    *,
    frame_start: int,
    frame_end: int,
    fps: float,
    frame_w: int,
    frame_h: int,
    detector_tracks: list[FaceTrack] | None = None,
    backend: str = "",
) -> TrackedSelection:
    """Backend chooser — the one entry point callers use.

    Priority: explicit `backend` request (sam2 unavailable = loud error, never
    a silent swap) > detector-assist when a track matches the prompt > ncc-cpu
    (always available). A detector-assist REFUSAL falls back to ncc-cpu (the
    click was on a non-detected object — the template tracker still owns it).
    """
    if backend == "sam2":
        chosen: Any = (
            Sam2ServiceBackend() if Sam2ServiceBackend.available() else Sam2SelectionBackend()
        )
        return chosen.track(
            video_path,
            prompt,
            frame_start=frame_start,
            frame_end=frame_end,
            fps=fps,
            frame_w=frame_w,
            frame_h=frame_h,
        )
    if backend not in ("", "ncc-cpu", "detector-assist"):
        raise SelectionError(f"unknown selection backend '{backend}'")
    if detector_tracks and backend in ("", "detector-assist"):
        try:
            return select_from_tracks(
                detector_tracks,
                prompt,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=fps,
                frame_w=frame_w,
                frame_h=frame_h,
            )
        except SelectionError:
            if backend == "detector-assist":
                raise  # explicit ask: the refusal IS the answer
    return track_selection_ncc(
        video_path,
        prompt,
        frame_start=frame_start,
        frame_end=frame_end,
        fps=fps,
        frame_w=frame_w,
        frame_h=frame_h,
    )
