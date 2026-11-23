"""SAM 2 local weight service — promptable video segmentation on YOUR machine.

The law of this repo: no torch in the default path, no silent model
downloads, CI never needs a GPU. So SAM 2 lives behind a service boundary:

- this FastAPI app is the ONLY component allowed to carry torch/ultralytics;
  it runs in its own container (docker-compose gpu profile), holding the
  weights on a named volume so a restart never re-downloads
- the Studio's selection port (`vision_fx.selection.Sam2ServiceBackend`)
  talks to it over httpx — a plain HTTP call, injectable, mockable
- with the service down, `run_selection` falls back to detector-assist /
  ncc-cpu exactly as before: nothing else changes

Endpoints:
    GET  /health           weights present? which model? device, tracker state
    POST /weights/ensure   download the checkpoint if missing (atomic, hashed)
    POST /track            prompt (box/point) on one frame -> per-frame boxes

The /track response matches `TrackedSelection`'s hit shape (t_s, x, y, w, h,
score, interpolated) in RAW source pixels — the API side fills asset_id and
provenance, so the service stays a pure perception worker.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="DirectorOS SAM2 local weight service", version="1.0.0")

# Meta's official sam2.1 Hiera-Tiny — the smallest checkpoint (fast on CPU)
DEFAULT_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)

_weights_lock = threading.Lock()
_predictor: Any = None  # SAM2VideoPredictor, loaded once on first /track


def weights_dir() -> Path:
    return Path(os.getenv("SAM2_WEIGHTS_DIR", str(Path(__file__).parent / "weights")))


def weights_path() -> Path:
    return Path(os.getenv("SAM2_WEIGHTS", str(weights_dir() / "sam2.1_hiera_tiny.pt")))


def weights_ready() -> bool:
    p = weights_path()
    return p.exists() and p.stat().st_size > 1_000_000  # real checkpoints are ~40-900MB


class HealthOut(BaseModel):
    service: str = "sam2-weights"
    weights_ready: bool
    weights_path: str
    weights_bytes: int
    weights_url: str
    device: str
    tracker_loaded: bool
    sha256: str | None = None


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — torch absent in tests/CI: still a valid answer
        return "cpu(unavailable)"


@app.get("/health")
def health() -> HealthOut:
    p = weights_path()
    ready = weights_ready()
    sha = None
    if ready:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return HealthOut(
        weights_ready=ready,
        weights_path=str(p),
        weights_bytes=p.stat().st_size if p.exists() else 0,
        weights_url=os.getenv("SAM2_WEIGHTS_URL", DEFAULT_WEIGHTS_URL),
        device=_device(),
        tracker_loaded=_predictor is not None,
        sha256=sha,
    )


class WeightsEnsureOut(BaseModel):
    ok: bool
    path: str
    downloaded: bool
    bytes: int
    sha256: str | None = None
    detail: str = ""


@app.post("/weights/ensure")
def weights_ensure() -> WeightsEnsureOut:
    """Download the checkpoint if missing (idempotent, atomic, hashed).

    Refuses to overwrite an existing file — the volume is the cache; delete
    the file explicitly to force a fresh download. SHA256 rides the response
    so the Studio can pin provenance.
    """
    import urllib.request

    p = weights_path()
    if weights_ready():
        return WeightsEnsureOut(
            ok=True,
            path=str(p),
            downloaded=False,
            bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            detail="weights already present (volume cache hit)",
        )
    url = os.getenv("SAM2_WEIGHTS_URL", DEFAULT_WEIGHTS_URL)
    expected_sha = os.getenv("SAM2_WEIGHTS_SHA256", "")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=str(p.parent))
    tmp = Path(tmp_dir) / "weights.pt"
    try:
        with _weights_lock:
            if weights_ready():  # double-check under the lock (two boots race)
                return WeightsEnsureOut(
                    ok=True,
                    path=str(p),
                    downloaded=False,
                    bytes=p.stat().st_size,
                    detail="weights already present (volume cache hit)",
                )
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 — pinned URL, no user input
            size = tmp.stat().st_size
            if size < 1_000_000:
                raise RuntimeError(f"downloaded checkpoint is {size} bytes — not a real model")
            sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if expected_sha and sha != expected_sha:
                raise RuntimeError(
                    f"checkpoint sha256 {sha[:16]}... does not match SAM2_WEIGHTS_SHA256"
                )
            tmp.replace(p)
    except Exception as e:  # noqa: BLE001 — surfaced to the caller, never a 500 wall
        raise HTTPException(status_code=502, detail=f"weight download failed: {e}") from e
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return WeightsEnsureOut(
        ok=True,
        path=str(p),
        downloaded=True,
        bytes=p.stat().st_size,
        sha256=sha,
        detail=f"downloaded from {url}",
    )


class TrackPrompt(BaseModel):
    """One prompt on ONE frame, RAW source pixels (the Studio's canvas)."""

    kind: str = Field(pattern="^(box|point)$")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)
    anchor_frame: int = Field(ge=0)


class TrackRequest(BaseModel):
    video_path: str  # shared-volume path — the service reads the same files
    prompt: TrackPrompt
    frame_start: int = Field(ge=0)
    frame_end: int = Field(gt=0)
    fps: float = Field(gt=0, le=240)
    frame_w: int = Field(gt=0)
    frame_h: int = Field(gt=0)


class TrackHitOut(BaseModel):
    t_s: float
    x: int
    y: int
    w: int
    h: int
    score: float
    interpolated: bool = False


class TrackOut(BaseModel):
    backend: str = "sam2"
    anchor_frame: int
    fps: float
    frame_start: int
    frame_end: int
    frame_w: int
    frame_h: int
    hits: list[TrackHitOut]
    mean_score: float
    n_interpolated: int = 0
    device: str


@app.post("/track")
def track(body: TrackRequest) -> TrackOut:
    """Promptable video segmentation: box/point on the anchor -> per-frame boxes."""
    if body.frame_end < body.frame_start:
        raise HTTPException(status_code=422, detail="frame_end must be >= frame_start")
    if body.prompt.kind == "box" and (body.prompt.w <= 0 or body.prompt.h <= 0):
        raise HTTPException(status_code=422, detail="box prompt needs w and h > 0")
    if not weights_ready():
        raise HTTPException(
            status_code=503,
            detail=f"weights not present at {weights_path()} — POST /weights/ensure first",
        )
    video = Path(body.video_path)
    if not video.exists():
        raise HTTPException(status_code=404, detail=f"video not reachable: {video}")

    predictor = _load_predictor()
    try:
        hits, mean_score = _run_predictor(predictor, body)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return TrackOut(
        anchor_frame=body.prompt.anchor_frame,
        fps=body.fps,
        frame_start=body.frame_start,
        frame_end=body.frame_end,
        frame_w=body.frame_w,
        frame_h=body.frame_h,
        hits=hits,
        mean_score=mean_score,
        n_interpolated=0,  # SAM2 is dense: every frame in the span is real
        device=_device(),
    )


def _load_predictor() -> Any:
    """Lazy one-time predictor load (torch+ultralytics only ever imported here)."""
    global _predictor
    if _predictor is not None:
        return _predictor
    try:
        from ultralytics.models.sam import SAM2VideoPredictor  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"ultralytics/torch unavailable in this container: {e}",
        ) from e
    with _weights_lock:
        if _predictor is None:
            _predictor = SAM2VideoPredictor(model=str(weights_path()))
    return _predictor


def _run_predictor(predictor: Any, body: TrackRequest) -> tuple[list[TrackHitOut], float]:
    """Run the SAM2 video predictor over [frame_start, frame_end].

    Ultralytics' video API streams frames from the file; the prompt lands on
    the anchor frame (bbox or point, source pixels) and every frame's mask
    becomes an axis-aligned bbox. Frames where SAM2 reports the object
    occluded produce no mask — the Studio interpolates those exactly like
    ncc-cpu gaps, so both backends emit the same hit shape.
    """
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"numpy unavailable: {e}") from e

    p = body.prompt
    n_frames = body.frame_end - body.frame_start + 1

    try:
        results = predictor(
            source=str(body.video_path),
            stream=True,
            bboxes=[p.x, p.y, p.x + p.w, p.y + p.h] if p.kind == "box" else None,
            points=[[p.x, p.y]] if p.kind == "point" else None,
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"SAM2 inference failed: {e}") from e

    hits: list[TrackHitOut] = []
    for idx, r in enumerate(results):
        if idx > n_frames - 1:
            break
        t_local = body.frame_start + idx
        xyxy = None
        score = 0.0
        # boxes first (video predictors emit box tensors per frame)...
        r_boxes = getattr(r, "boxes", None)
        if r_boxes is not None and getattr(r_boxes, "xyxy", None) is not None:
            try:
                xyxy = r_boxes.xyxy[0].tolist()
                conf = getattr(r_boxes, "conf", None)
                score = float(conf[0]) if conf is not None and len(conf) else 1.0
            except Exception:  # noqa: BLE001
                xyxy = None
        # ...masks as the fallback (mask -> axis-aligned bbox)
        if xyxy is None:
            mask = getattr(r, "masks", None)
            if mask is not None and mask.data is not None and len(mask.data) > 0:
                m = mask.data[0].cpu().numpy().astype(bool)
                ys, xs = np.where(m)
                if len(xs) > 0:
                    xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if xyxy is not None:
            x0, y0, x1, y1 = xyxy
            box_out = (
                max(0, int(x0)),
                max(0, int(y0)),
                max(4, int(x1 - x0)),
                max(4, int(y1 - y0)),
            )
            hits.append(
                TrackHitOut(
                    t_s=t_local / body.fps,
                    x=box_out[0],
                    y=box_out[1],
                    w=box_out[2],
                    h=box_out[3],
                    score=score,
                )
            )

    if not hits:
        raise RuntimeError(
            f"SAM2 returned no usable masks for frames [{body.frame_start}, {body.frame_end}]"
        )
    mean_score = sum(h.score for h in hits) / len(hits)
    return hits, round(mean_score, 3)
