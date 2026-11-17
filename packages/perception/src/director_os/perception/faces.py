"""Face detection on real footage -> redaction blur regions (PLAN §4.4, M4).

Closes the "privacy redaction is only as good as the human drawing boxes"
gap: a detector sweep over sampled raw frames produces pixel-accurate face
boxes that `cluster_to_regions` consolidates into `BlurRegion` specs the
render pipeline already understands (`compile_timeline.global_blur_regions`
merges them with the studio-drawn canvas boxes).

Detector: OpenCV YuNet (`face_detection_yunet_2023mar.onnx`, ~230 KB) — a
modern single-shot face detector that runs on CPU, no GPU and no service
dependency. The model ships inside the package (`perception/models/`), so
the sweep works on a fresh checkout with zero downloads. (OpenCV 5 removed
the legacy Haar cascades — YuNet is the supported detector.)

Everything here works on RAW pixel coordinates at the source resolution —
the same canvas the studio blur canvas and the renderer assume.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from director_os.core.models import BBox, BlurRegion
from pydantic import BaseModel, Field

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
)

# sampling: bodycam shots sit still — 1 fps is plenty and keeps a 43-min raw
# sweep in the ~2600-frame range (CPU-decode seconds, not minutes)
DEFAULT_SAMPLE_EVERY_S = 1.0
DEFAULT_MAX_SAMPLES = 4000


class FaceDetection(BaseModel):
    """One detector hit on one sampled frame (raw source pixels)."""

    t_s: float = Field(ge=0.0)
    bbox: BBox
    score: float = Field(ge=0.0, le=1.0)


class FaceSweepStats(BaseModel):
    """What the sweep saw — enough for the cost dashboard and the QA story."""

    frames_sampled: int
    frames_decoded_failed: int
    detections: int
    detector: str
    sample_every_s: float


class FaceSweep(BaseModel):
    stats: FaceSweepStats
    detections: list[FaceDetection]


# ----------------------------------------------------------------- detectors
class _DetectorProtocol(Protocol):
    """Structural type any face detector must satisfy (name + raw-pixel boxes)."""

    name: str

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]: ...


class YuNetFaceDetector:
    """cv2.FaceDetectorYN wrapper: fixed input size, raw-pixel output boxes."""

    name = "yunet"

    def __init__(self, model_path: Path | None = None, score_threshold: float = 0.5):
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"YuNet model missing: {path}")
        self._score_threshold = score_threshold
        self._model_path = path
        self._input_w = 0
        self._input_h = 0
        self._det: cv2.FaceDetectorYN | None = None

    def _ensure(self, w: int, h: int) -> None:
        # YuNet wants the input size set once per frame geometry; recreate on change
        if self._det is None or (self._input_w, self._input_h) != (w, h):
            self._det = cv2.FaceDetectorYN.create(
                str(self._model_path),
                "",
                (w, h),
                score_threshold=self._score_threshold,
            )
            self._input_w, self._input_h = w, h

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = frame_bgr.shape[:2]
        self._ensure(w, h)
        assert self._det is not None
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame_bgr)
        out: list[tuple[int, int, int, int, float]] = []
        if faces is None:
            return out
        for f in faces:
            x, y, fw, fh = f[0], f[1], f[2], f[3]
            score = float(f[-1])
            if fw <= 1 or fh <= 1:
                continue
            out.append((max(0, int(x)), max(0, int(y)), int(fw), int(fh), score))
        return out


def make_detector(model_path: Path | None = None) -> _DetectorProtocol:
    """YuNet from the packaged model (or an explicit override path)."""
    return YuNetFaceDetector(Path(model_path) if model_path else DEFAULT_MODEL_PATH)


# -------------------------------------------------------------------- sweep
def detect_faces(
    video_path: Path,
    sample_every_s: float = DEFAULT_SAMPLE_EVERY_S,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    score_threshold: float = 0.5,
    detector: _DetectorProtocol | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FaceSweep:
    """Sample frames at `sample_every_s` and run the detector on each.

    Decoding is capped by `max_samples` (a 43-min source at 1 fps still lands
    under the default). Frame geometry comes from the video itself; boxes are
    in RAW source pixels, directly consumable by the render pipeline.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = detector or make_detector()
    detections: list[FaceDetection] = []
    decoded = 0
    failed = 0
    stride = max(1, int(round(fps * sample_every_s)))
    idx = 0
    while decoded < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            try:
                hits = det.detect(frame)
            except Exception:
                failed += 1
                idx += 1
                continue
            decoded += 1
            t_s = idx / fps if fps else 0.0
            for x, y, w, h, score in hits:
                if score < score_threshold:
                    continue
                detections.append(
                    FaceDetection(
                        t_s=round(t_s, 3), bbox=BBox(x=x, y=y, w=w, h=h), score=round(score, 3)
                    )
                )
            if progress is not None and decoded % 100 == 0:
                progress(decoded, max_samples)
        idx += 1
    cap.release()
    return FaceSweep(
        stats=FaceSweepStats(
            frames_sampled=decoded,
            frames_decoded_failed=failed,
            detections=len(detections),
            detector=det.name,
            sample_every_s=sample_every_s,
        ),
        detections=detections,
    )


# ------------------------------------------------------------- consolidation
def _iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def cluster_to_regions(
    sweep: FaceSweep,
    min_hits: int = 3,
    max_regions: int = 8,
    pad_frac: float = 0.10,
    frame_w: int = 1920,
    frame_h: int = 1080,
    sigma: int = 24,
) -> list[BlurRegion]:
    """Greedy IoU clustering of detections -> padded static `BlurRegion`s.

    A cluster needs `min_hits` observations to survive — one-shot detector
    false positives on e.g. a poster don't earn a permanent blur. The top
    `max_regions` clusters (by hit count) win; boxes are padded by
    `pad_frac` (faces move between sampled frames) and clipped to frame
    bounds. Output is `scope="global"`, `mode="static"`: the renderer already
    merges these with studio-drawn boxes at compile time.
    """
    boxes: list[BBox] = [d.bbox for d in sweep.detections]
    clusters: list[dict] = []  # {"boxes": [BBox], "hits": int}

    for b in boxes:
        placed = False
        for c in clusters:
            rep = c["boxes"][0]
            if _iou(rep, b) > 0.2:
                c["boxes"].append(b)
                c["hits"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"boxes": [b], "hits": 1})

    survivors = [c for c in clusters if c["hits"] >= min_hits]
    survivors.sort(key=lambda c: c["hits"], reverse=True)
    survivors = survivors[:max_regions]

    regions: list[BlurRegion] = []
    for i, c in enumerate(survivors):
        xs = [b.x for b in c["boxes"]]
        ys = [b.y for b in c["boxes"]]
        ws = [b.x + b.w for b in c["boxes"]]
        hs = [b.y + b.h for b in c["boxes"]]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(ws), max(hs)
        pad_w = int((x1 - x0) * pad_frac)
        pad_h = int((y1 - y0) * pad_frac)
        x0 = max(0, x0 - pad_w)
        y0 = max(0, y0 - pad_h)
        x1 = min(frame_w, x1 + pad_w)
        y1 = min(frame_h, y1 + pad_h)
        regions.append(
            BlurRegion(
                scope="global",
                bbox=BBox(x=x0, y=y0, w=max(8, x1 - x0), h=max(8, y1 - y0)),
                mode="static",
                sigma=sigma,
                track_id=f"face_cluster_{i + 1}",
            )
        )
    return regions
