"""YOLOv8n-face ONNX detector — CPU onnxruntime, no SDK, no service.

Model: `yolov8n-face-lindevs.onnx` (~12 MB, single-class face detector,
ultralytics export, published SHA-256 pinned in scripts/fetch_yolov8n_face.py).
Fetch once into `data/models/` (gitignored) — the backend degrades to YuNet
when the file is absent, so a fresh checkout still sweeps faces offline.

Decode contract (ultralytics ONNX export): output tensor (1, 5, 8400) =
[cx, cy, w, h, score] rows in 640x640 letterboxed input-pixel space; boxes
are mapped back to RAW frame coordinates through the exact letterbox inverse
(scale then pad removal), matching `redact.scale_bbox` semantics in reverse.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL_FILENAME = "yolov8n-face-lindevs.onnx"
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
INPUT_SIZE = 640


def resolve_yolo_model_path(model_path: str | Path | None = None) -> Path | None:
    """Search order: explicit arg (must exist — fail loud otherwise) -> env ->
    repo-root data/models (dev checkout / editable install) -> package models dir."""
    here = Path(__file__).resolve()
    # here = <root>/packages/vision_fx/src/director_os/vision_fx/compute/yolo_onnx.py
    # parents: 0=compute 1=vision_fx 2=director_os 3=src 4=vision_fx(pkg) 5=packages 6=<repo root>
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"explicit yolov8n-face model not found: {p}")
        return p
    env = os.getenv("DIRECTOR_YOLOV8N_FACE_PATH", "").strip()
    candidates: list[Path] = [Path(env)] if env else []
    for up in (6, 5):
        candidates.append(here.parents[up] / "data" / "models" / DEFAULT_MODEL_FILENAME)
    candidates.append(Path.cwd() / "data" / "models" / DEFAULT_MODEL_FILENAME)
    candidates.append(here.parent.parent / "models" / DEFAULT_MODEL_FILENAME)
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_input_size(declared_shape: list, requested: int) -> tuple[int, str]:
    """Reconcile the caller's requested sweep size with the ONNX model's input.

    Returns (effective_size, note). Fixed exports (lindevs releases pin 640x640)
    MUST win: feeding 960 into a 640-fixed model raises INVALID_ARGUMENT — the
    exact silent failure that emptied the Eustis tracked plan (issue #14
    follow-up). Dynamic exports (str/symbolic dims) honor the requested size.
    """
    dims = declared_shape[1:] if len(declared_shape) >= 4 else []
    ints = [d for d in dims if isinstance(d, int)]
    if len(ints) >= 2 and len(ints) == len(dims):
        eff = ints[-1]
        if eff != requested:
            return eff, f"model input fixed at {eff}; requested {requested} honored-as-{eff}"
        return eff, f"model input fixed at {eff}"
    return requested, "dynamic model input; requested size used"


class YoloOnnxFaceDetector:
    """onnxruntime wrapper: (name + raw-pixel boxes), same shape as YuNet's."""

    name = "yolov8n-face-onnx"

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        input_size: int = INPUT_SIZE,
    ):
        resolved = resolve_yolo_model_path(model_path)
        if resolved is None:
            raise FileNotFoundError(
                "yolov8n-face ONNX model not found — run scripts/fetch_yolov8n_face.py "
                "or set DIRECTOR_YOLOV8N_FACE_PATH"
            )
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - exercised only without dep
            raise RuntimeError("onnxruntime is not installed (uv sync should provide it)") from e
        self._model_path = resolved
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._input_size = input_size
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        self._session = ort.InferenceSession(
            str(resolved), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_size, self._input_note = resolve_input_size(
            list(self._session.get_inputs()[0].shape), input_size
        )

    # ------------------------------------------------------------- preprocess
    def _letterbox(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize with aspect preserved + centered pad. Returns blob + inverse params."""
        import cv2

        h, w = frame_bgr.shape[:2]
        s = min(self._input_size / w, self._input_size / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self._input_size - nw) // 2, (self._input_size - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[None], s, pad_x, pad_y

    # --------------------------------------------------------------- decode
    @staticmethod
    def _decode(out: np.ndarray, conf: float) -> list[tuple[float, float, float, float, float]]:
        """(1, D, N) -> [(cx, cy, w, h, score)] above threshold (input pixels).

        Layouts handled: 5 rows = box+score (single-class export); 4+nc rows =
        box + per-class scores (score = max class); 4 rows only = rejected
        (no confidence channel — refusing beats guessing).
        """
        arr = np.asarray(out)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"unexpected YOLO output ndim={arr.ndim}")
        rows, n = arr.shape
        boxes = arr[0:4, :]
        if rows >= 5:
            scores = arr[4:, :].max(axis=0)
        else:
            raise ValueError(f"YOLO output has {rows} rows — no score channel (rows={rows})")
        keep = scores >= conf
        if not keep.any():
            return []
        b = boxes[:, keep].T  # (K, 4)
        s = scores[keep]
        return [
            (float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3]), float(s[i]))
            for i in range(len(s))
        ]

    @staticmethod
    def _nms(dets: list[tuple[float, float, float, float, float]], iou_thr: float):
        """Greedy NMS on (cx, cy, w, h, score) — returns indices kept, best first."""
        order = sorted(range(len(dets)), key=lambda i: dets[i][4], reverse=True)
        kept: list[int] = []
        for i in order:
            ci_x, ci_y, ci_w, ci_h, _ = dets[i]
            ok = True
            for j in kept:
                cj_x, cj_y, cj_w, cj_h, _ = dets[j]
                x1, y1 = (
                    max(ci_x - ci_w / 2, cj_x - cj_w / 2),
                    max(ci_y - ci_h / 2, cj_y - cj_h / 2),
                )
                x2, y2 = (
                    min(ci_x + ci_w / 2, cj_x + cj_w / 2),
                    min(ci_y + ci_h / 2, cj_y + cj_h / 2),
                )
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = ci_w * ci_h + cj_w * cj_h - inter
                if union > 0 and inter / union > iou_thr:
                    ok = False
                    break
            if ok:
                kept.append(i)
        return kept

    # ---------------------------------------------------------------- public
    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """One frame -> [(x, y, w, h, score)] in RAW frame pixels."""

        blob, s, pad_x, pad_y = self._letterbox(frame_bgr)
        out = self._session.run(None, {self._input_name: blob})[0]
        dets = self._decode(out, self._conf)
        dets = [dets[i] for i in self._nms(dets, self._iou)]
        h, w = frame_bgr.shape[:2]
        hits: list[tuple[int, int, int, int, float]] = []
        for cx, cy, bw, bh, score in dets:
            # inverse letterbox: remove pad, divide by scale, clamp to frame
            x0 = (cx - bw / 2 - pad_x) / s
            y0 = (cy - bh / 2 - pad_y) / s
            x1 = (cx + bw / 2 - pad_x) / s
            y1 = (cy + bh / 2 - pad_y) / s
            x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
            x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            hits.append((x0, y0, x1 - x0, y1 - y0, round(float(score), 3)))
        return hits
