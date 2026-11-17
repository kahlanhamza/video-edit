"""Modal GPU redaction client (PLAN §8, M4) — YOLOv8-face + plate + ByteTrack.

Job 3 of the vision stack. A slow VLM never tracks faces 30x/second — the
serverless GPU fleet does (deploy/modal/redact_gpu.py): YOLOv8-face +
license-plate models + ByteTrack persistent IDs on a T4, 60-120 FPS, pennies
out of the free credits. This module is the CPU-side contract:

- `parse_tracks_payload` / `tracks_to_regions` — PURE conversion of the fleet's
  track JSON into time-binned `BlurRegion(mode="tracked")` specs in RAW pixel
  space. Everything downstream (compile_timeline, build_blur_chain) already
  consumes them, so every behavior is testable hermetically without Modal.
- `redact_video_modal` — the actual remote call; activates when MODAL_TOKEN_ID/
  MODAL_TOKEN_SECRET exist, raises ModalRedactError otherwise (the pipeline
  falls back to the YuNet CPU sweep).

Bodycam-smart workflow: tracks arrive labeled `face_<id>` / `plate_<id>`, so
the studio can keep the suspect (Person #1) unblurred while bystanders, minors
and every plate are redacted; static FFmpeg boxes still cover corner watermarks.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from director_os.core.models import BBox, BlurRegion


class ModalRedactError(RuntimeError):
    pass


DEFAULT_BIN_S = 2.0
DEFAULT_MIN_HITS = 3
DEFAULT_PAD_FRAC = 0.12
DEFAULT_SIGMA = 32
DEFAULT_MAX_REGIONS = 24


@dataclass(frozen=True)
class TrackFrame:
    t_s: float
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0


@dataclass
class TrackedObject:
    """One persistent ByteTrack identity: ordered frame observations."""

    track_id: int
    kind: str  # face | plate
    frames: list[TrackFrame] = field(default_factory=list)

    @property
    def t_start_s(self) -> float:
        return min(f.t_s for f in self.frames)

    @property
    def t_end_s(self) -> float:
        return max(f.t_s for f in self.frames)


def parse_tracks_payload(payload: dict) -> list[TrackedObject]:
    """Tolerant reader for the fleet JSON:
    {"tracks": [{"id": 1, "cls": "face", "frames": [{"t":..., "bbox":[x,y,w,h], "score":...}]}]}
    Accepts "track_id"/"id", "cls"/"class"/"kind", bbox as list or dict; skips junk.
    """
    out: list[TrackedObject] = []
    for tr in payload.get("tracks", []):
        if not isinstance(tr, dict):
            continue
        tid = tr.get("id", tr.get("track_id"))
        kind = str(tr.get("cls", tr.get("class", tr.get("kind", "face")))).lower()
        if tid is None:
            continue
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        obj = TrackedObject(track_id=tid, kind=kind)
        for fr in tr.get("frames", []):
            try:
                bbox = fr.get("bbox", fr.get("box"))
                if isinstance(bbox, dict):
                    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                else:
                    x, y, w, h = bbox
                obj.frames.append(
                    TrackFrame(
                        t_s=float(fr.get("t", fr.get("t_s", 0.0))),
                        x=float(x),
                        y=float(y),
                        w=float(w),
                        h=float(h),
                        score=float(fr.get("score", 1.0)),
                    )
                )
            except (TypeError, ValueError, KeyError, IndexError):
                continue
        if obj.frames:
            obj.frames.sort(key=lambda f: f.t_s)
            out.append(obj)
    return out


def _pad_clip(x: float, y: float, w: float, h: float, pad_frac: float, fw: int, fh: int) -> BBox:
    pw, ph = w * pad_frac, h * pad_frac
    x0 = max(0, int(round(x - pw)))
    y0 = max(0, int(round(y - ph)))
    x1 = min(fw, int(round(x + w + pw)))
    y1 = min(fh, int(round(y + h + ph)))
    if x1 - x0 < 4:
        x1 = min(fw, x0 + 4)
    if y1 - y0 < 4:
        y1 = min(fh, y0 + 4)
    return BBox(x=x0, y=y0, w=max(4, x1 - x0), h=max(4, y1 - y0))


def tracks_to_regions(
    tracks: list[TrackedObject],
    frame_w: int,
    frame_h: int,
    bin_s: float = DEFAULT_BIN_S,
    min_hits: int = DEFAULT_MIN_HITS,
    pad_frac: float = DEFAULT_PAD_FRAC,
    sigma: int = DEFAULT_SIGMA,
    max_regions: int = DEFAULT_MAX_REGIONS,
) -> list[BlurRegion]:
    """Fleet tracks -> time-binned blur regions (RAW pixel space).

    Per track, frames are bucketed into `bin_s` windows; each surviving bucket
    gets the MEDIAN box (ByteTrack already smoothed coordinates — the median
    kills residual jitter so boxes glide without flickering). Buckets with
    fewer than `min_hits` observations are dropped (single-frame false
    positives never become blur). Regions are ranked by total hits so the
    `max_regions` budget goes to the objects that matter most.
    """
    candidates: list[tuple[int, BlurRegion]] = []
    for obj in tracks:
        if not obj.frames:
            continue
        bins: dict[int, list[TrackFrame]] = {}
        for f in obj.frames:
            key = int(f.t_s // bin_s) if bin_s > 0 else 0
            bins.setdefault(key, []).append(f)
        for key, frames in bins.items():
            if len(frames) < min_hits:
                continue
            mx = statistics.median(f.x for f in frames)
            my = statistics.median(f.y for f in frames)
            mw = statistics.median(f.w for f in frames)
            mh = statistics.median(f.h for f in frames)
            if mw <= 1 or mh <= 1:
                continue
            bbox = _pad_clip(mx, my, mw, mh, pad_frac, frame_w, frame_h)
            t0 = round(key * bin_s, 3)
            t1 = round(min((key + 1) * bin_s, obj.t_end_s + bin_s), 3)
            region = BlurRegion(
                scope="global",
                bbox=bbox,
                mode="tracked",
                sigma=sigma,
                track_id=f"{obj.kind}_{obj.track_id}",
                t_start_s=t0,
                t_end_s=max(t1, t0 + 0.5),
            )
            candidates.append((len(frames), region))

    candidates.sort(key=lambda c: -c[0])
    return [r for _, r in candidates[:max_regions]]


def merge_with_static(tracked: list[BlurRegion], static: list[BlurRegion]) -> list[BlurRegion]:
    """Tracked + studio/detector static boxes, no duplicates by (track_id, bin)."""
    seen: set[tuple[str | None, float, float]] = set()
    out: list[BlurRegion] = []
    for r in [*tracked, *static]:
        key = (r.track_id, r.t_start_s or 0.0, r.t_end_s or 0.0)
        if r.mode == "tracked" and key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def modal_configured() -> bool:
    from director_os.core.config import get_settings

    s = get_settings()
    return bool(s.modal_token_id and s.modal_token_secret)


def redact_video_modal(
    video_path: str,
    sample_fps: float = 5.0,
    conf: float = 0.25,
    remote_url: str | None = None,
) -> dict:
    """Call the deployed fleet function; returns the raw track payload.

    Either pass a local `video_path` (uploaded as bytes — fine for segments up
    to ~70MB) or a `remote_url` the container fetches itself (the full 448MB
    webm case). Requires `modal deploy deploy/modal/redact_gpu.py` once.
    """
    if not modal_configured():
        raise ModalRedactError(
            "MODAL_TOKEN_ID/MODAL_TOKEN_SECRET not configured — falling back to CPU YuNet sweep"
        )
    try:
        import base64

        import modal

        fn = modal.Function.from_name("directoros-vision", "redact_video")
        if remote_url:
            return fn.remote(url=remote_url, sample_fps=sample_fps, conf=conf)
        data = base64.b64encode(Path(video_path).read_bytes()).decode()
        return fn.remote(video_b64=data, sample_fps=sample_fps, conf=conf)
    except ModalRedactError:
        raise
    except Exception as e:  # fleet down / function not deployed — degrade loudly
        raise ModalRedactError(f"Modal redaction call failed: {e}") from e
