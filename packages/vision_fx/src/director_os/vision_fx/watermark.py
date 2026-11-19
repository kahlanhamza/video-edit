"""Automatic burned-in watermark / channel-logo detection — ANY video.

Owner finding (2026-11-20, Eustis final_720p): the "Police Bodycam Videos"
mark was never blurred at its exact pixels. The static bottom-right band
approach cannot work in general: overlays are INTERMITTENT (only some source
clips carry them) and they MOVE with per-clip crops/zooms. This module finds
them automatically:

  1. ADAPTIVE SAMPLING — stride derived from the probed duration
     (n = clamp(duration/4, 24, 120) frames). No hardcoded lengths.
  2. SHOT CLUSTERING — 8x8 dHash fingerprint, adjacent samples with hamming
     distance <= 10 form a cluster. Intermittent marks live per-cluster, so
     a mark present in only 20% of the video still gets its own cluster.
  3. LOCAL CANDIDATES — per cluster, the TEMPORAL MEDIAN of Sobel edge maps:
     scene edges move shot to shot, a burned-in overlay's edges persist ->
     median edge energy + low per-pixel temporal variance = overlay pixels.
     Morphological close + connected components -> candidate boxes.
  4. VISION-MODEL CONFIRM — one representative frame per cluster to the
     Gemini client (perception.vision.GeminiVisionClient): the model returns
     labeled boxes with kinds; only watermark/channel-brand/promo overlays
     are blurred. Timestamp burns (bodycam clocks), subtitles and content
     text are labeled and LEFT VISIBLE. Without an API key the local
     heuristic candidates stand (still automatic).
  5. STABLE PLAN — one CONSTANT box per (cluster, label): dense per-frame
     boxes inside the cluster span, zero on/off flicker by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from director_os.vision_fx.trajectory import RedactionTrajectory, TrackedBox

if TYPE_CHECKING:
    import numpy as np

WATERMARK_KINDS = {"watermark", "brand", "promo", "channel_logo", "logo"}
DHASH_HAMMING_MAX = 10
MIN_CLUSTER_SAMPLES = 2
MIN_BOX_W_FRAC = 0.03
MAX_BOX_W_FRAC = 0.70
MIN_BOX_H_FRAC = 0.008
MAX_BOX_H_FRAC = 0.22


class WatermarkError(RuntimeError):
    pass


@dataclass
class WatermarkFinding:
    label: str
    kind: str
    box: tuple[int, int, int, int]  # x, y, w, h — RAW SOURCE pixels
    t_start_s: float
    t_end_s: float
    confidence: float
    source: str  # "vision" | "local"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "box": list(self.box),
            "t_start_s": round(self.t_start_s, 3),
            "t_end_s": round(self.t_end_s, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


@dataclass
class WatermarkReport:
    findings: list[WatermarkFinding] = field(default_factory=list)
    n_clusters: int = 0
    n_samples: int = 0
    vision_used: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "findings": [f.as_dict() for f in self.findings],
            "n_clusters": self.n_clusters,
            "n_samples": self.n_samples,
            "vision_used": self.vision_used,
            "notes": self.notes,
        }


def _probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError as e:
        raise WatermarkError(f"cannot probe duration of {path}: {r.stderr[-200:]}") from e


def _dhash(gray) -> bytes:
    import cv2
    import numpy as np

    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).astype(np.uint8)
    return np.packbits(bits).tobytes()


def _hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=False))


def _sample_frames(
    path: Path, duration_s: float, max_samples: int = 120
) -> list[tuple[float, np.ndarray]]:
    """Adaptive even sampling -> [(t_s, frame)] (list, memory-bounded)."""
    import cv2

    n = int(min(max_samples, max(24, math.ceil(duration_s / 4.0))))
    stride_t = duration_s / n
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise WatermarkError(f"cannot open video: {path}")
    out = []
    for k in range(n):
        t = min(duration_s, k * stride_t)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        out.append((t, frame))
    cap.release()
    return out


def _cluster_by_hash(samples: list[tuple[float, np.ndarray]]) -> list[list[int]]:
    """Adjacent-sample clustering: cut where the fingerprint jumps."""
    clusters: list[list[int]] = []
    prev_hash = None
    for i, (_t, frame) in enumerate(samples):
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = _dhash(gray)
        if prev_hash is None or _hamming(h, prev_hash) > DHASH_HAMMING_MAX:
            clusters.append([i])
        else:
            clusters[-1].append(i)
        prev_hash = h
    return clusters


def _edge_map(gray) -> object:
    import cv2
    import numpy as np

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.threshold(mag, 60.0, 255.0, cv2.THRESH_BINARY)[1].astype(np.uint8)


def _median_edge_map(cluster_frames: list[np.ndarray]) -> np.ndarray:
    """Temporal median of per-frame edge maps — static overlay edges survive."""
    import cv2
    import numpy as np

    edges = [np.asarray(_edge_map(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in cluster_frames]
    return np.median(np.stack(edges), axis=0)


def _box_evidence_ratio(box: tuple[int, int, int, int], edge_med: np.ndarray) -> float:
    """Fraction of the box carrying PERSISTENT edges (a burned-in mark's
    strokes). Vision-model claims are only accepted where this is > 0 —
    hallucinated or stale boxes die here."""

    x, y, w, h = box
    H, W = edge_med.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    region = edge_med[y0:y1, x0:x1]
    return float((region > 0).mean())


def _local_candidates(
    cluster_frames: list[np.ndarray], w: int, h: int
) -> list[tuple[int, int, int, int]]:
    """Temporal-median edge + low temporal variance -> static overlay boxes."""
    import cv2
    import numpy as np

    edge_med = _median_edge_map(cluster_frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in cluster_frames]
    gray_var = np.stack(grays).var(axis=0)
    # overlay pixels: persistent edges, stable brightness
    mask = ((edge_med > 0) & (gray_var < 300.0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    mask_u8: np.ndarray = np.ascontiguousarray(dilated, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (MIN_BOX_W_FRAC * w <= bw <= MAX_BOX_W_FRAC * w):
            continue
        if not (MIN_BOX_H_FRAC * h <= bh <= MAX_BOX_H_FRAC * h):
            continue
        if area < 0.25 * bw * bh:
            continue  # sparse scatter, not a solid mark
        boxes.append((int(x), int(y), int(bw), int(bh)))
    return boxes


def _vision_confirm(client, samples, cluster_idx: list[int]) -> list[dict]:
    """Ask the vision model for labeled overlay boxes on 1-2 cluster frames.

    The shared INSIGHT_INSTRUCTION contract returns SceneInsight JSON; the
    watermark boxes ride in `description` as a strict JSON object.
    """
    from director_os.perception.vision import FrameRef

    picks = cluster_idx[:2]
    frames = [
        FrameRef(timestamp_s=round(samples[i][0], 3), image_path=_frame_cache(samples, i))
        for i in picks
    ]
    question = (
        "List every burned-in overlay (channel watermark, brand logo, promo "
        "text) visible in the frame(s). For EACH one reply one object with "
        'kind "redaction_candidate" and description as strict JSON '
        '{"label": str, "box_xywh": [x, y, w, h] in PIXELS of the image, '
        '"overlay_kind": "watermark"|"timestamp"|"subtitle"|"other"}. '
        "Bodycam timestamp/clock burns and burned video subtitles are NOT "
        "watermarks — if a text is a timestamp or subtitle, set overlay_kind "
        "accordingly and it will be kept visible. Answer with the JSON array only."
    )
    try:
        insights = client.analyze(question, frames)
    except Exception:  # noqa: BLE001 — vision is an accelerator, not a gate
        return []
    out: list[dict] = []
    for ins in insights:
        try:
            payload = json.loads(ins.description)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict) or "box_xywh" not in payload:
            continue
        out.append(
            {
                "label": str(payload.get("label", "watermark"))[:60],
                "box": [int(v) for v in payload["box_xywh"][:4]],
                "overlay_kind": str(payload.get("overlay_kind", "watermark")),
                "confidence": float(ins.confidence or 0.0),
                "timestamp_s": float(ins.timestamp_s or 0.0),
            }
        )
    return out


_FRAME_CACHE: dict[int, Path] = {}


def _frame_cache(samples, idx: int) -> Path:
    import cv2

    if idx in _FRAME_CACHE:
        return _FRAME_CACHE[idx]
    import tempfile

    d = Path(tempfile.gettempdir()) / "directoros_wm"
    d.mkdir(parents=True, exist_ok=True)
    t, frame = samples[idx]
    p = d / f"wm_{idx:04d}.jpg"
    cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _FRAME_CACHE[idx] = p
    return p


def detect_watermarks(
    video: str | Path,
    *,
    vision_client=None,
    max_samples: int = 120,
) -> WatermarkReport:
    """Full automatic pass -> findings (any video, any length, no constants)."""
    video = Path(video)
    duration = _probe_duration(video)
    samples = _sample_frames(video, duration, max_samples)
    if len(samples) < 2:
        return WatermarkReport(n_samples=len(samples), notes=["too short to sample"])
    h_px, w_px = samples[0][1].shape[:2]
    clusters = _cluster_by_hash(samples)
    clusters = [c for c in clusters if len(c) >= MIN_CLUSTER_SAMPLES]

    report = WatermarkReport(n_clusters=len(clusters), n_samples=len(samples))
    # merge same-label boxes that repeat across clusters (same mark, same place)
    by_label: dict[tuple[str, tuple[int, int, int, int]], WatermarkFinding] = {}

    def _add(label: str, kind: str, box, t0: float, t1: float, conf: float, src: str):
        x, y, w, h = box
        if not (x >= 0 and y >= 0 and w > 8 and h > 6 and x + w <= w_px + 2 and y + h <= h_px + 2):
            return
        key = (label, (x // 24, y // 24, w // 24, h // 24))
        if key in by_label:
            f = by_label[key]
            f.t_start_s = min(f.t_start_s, t0)
            f.t_end_s = max(f.t_end_s, t1)
            f.confidence = max(f.confidence, conf)
        else:
            by_label[key] = WatermarkFinding(label, kind, (x, y, w, h), t0, t1, conf, src)

    for c in clusters:
        t0, t1 = samples[c[0]][0], samples[c[-1]][0]
        cluster_frames = [samples[i][1] for i in c]
        edge_med = None
        confirmed: list[dict] = []
        if vision_client is not None:
            confirmed = _vision_confirm(vision_client, samples, c)
            report.vision_used = True
        if confirmed:
            # local evidence cross-check INSIDE THIS CLUSTER: a claimed box is
            # accepted only where persistent edges exist there (kills repeats
            # from other clusters and model hallucinations alike)
            edge_med = _median_edge_map(cluster_frames)
            for item in confirmed:
                if item["overlay_kind"] not in WATERMARK_KINDS:
                    continue
                if _box_evidence_ratio(item["box"], edge_med) < 0.05:
                    continue
                _add(
                    item["label"],
                    item["overlay_kind"],
                    item["box"],
                    t0,
                    t1,
                    item["confidence"],
                    "vision",
                )
        else:
            # local heuristic fallback (no key / no vision answer)
            for box in _local_candidates(cluster_frames, w_px, h_px):
                _add("overlay", "watermark", box, t0, t1, 0.4, "local")

    report.findings = sorted(by_label.values(), key=lambda f: f.t_start_s)
    return report


def findings_to_trajectories(
    findings: list[WatermarkFinding], fps: float
) -> list[RedactionTrajectory]:
    """Constant-box dense trajectories — the anti-flicker watermark plan."""
    out: list[RedactionTrajectory] = []
    for i, f in enumerate(findings):
        f0 = max(0, int(math.floor(f.t_start_s * fps)))
        f1 = max(f0 + 1, int(math.ceil(f.t_end_s * fps)))
        x, y, w, h = f.box
        out.append(
            RedactionTrajectory(
                track_id=f"watermark_{_slug(f.label)}_{i}",
                boxes=[
                    TrackedBox(frame_idx=k, t_s=round(k / fps, 3), x=x, y=y, w=w, h=h)
                    for k in range(f0, f1)
                ],
                n_windows=1,
            )
        )
    return out


def _slug(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]
