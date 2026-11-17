"""LocalOnnxCpuBackend — the $0 default. Zero card, zero cloud, zero SDK.

`COMPUTE_BACKEND=local` (default). Face jobs: yolov8n-face.onnx through
onnxruntime on CPU, sampled every 3rd frame (`sample_every_s=0.1` at 30 fps)
with linear interpolation filling the stride gaps; degrades to packaged YuNet
when the YOLO model file is absent so a fresh checkout still works offline.
`stems`: on-demand Demucs ONLY on music-flagged spans, and only when the
optional `demucs` extra is installed (torch is deliberately not a hard dep —
a CPU-only install would bloat every checkout for a job that runs free on
Kaggle instead). `asr_align`: not available locally — Gladia (perception)
covers M2-M5; WhisperX rides the Kaggle adapter.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from director_os.vision_fx.compute.base import (
    AsrAlignResult,
    ComputeNotAvailable,
    FaceTrack,
    Span,
    StemsResult,
)
from director_os.vision_fx.compute.tracker import track_and_interpolate
from director_os.vision_fx.compute.yolo_onnx import (
    YoloOnnxFaceDetector,
    resolve_yolo_model_path,
)

DEFAULT_SAMPLE_EVERY_S = 0.1  # ~stride 3 at 30 fps
DEFAULT_MAX_SAMPLES = 4000


class _FrameDetector(Protocol):
    """Structural type any face detector must satisfy (name + raw-pixel boxes)."""

    name: str

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]: ...


class LocalOnnxCpuBackend:
    """CPU ONNX adapter — the zero-cost baseline every pipeline can assume."""

    name = "local-onnx-cpu"

    def __init__(
        self,
        yolo_model_path: str | Path | None = None,
        detector: _FrameDetector | None = None,
        stride: int = 3,
        sample_every_s: float | None = None,
    ):
        self._yolo_path = yolo_model_path
        self._detector = detector
        self.stride = max(1, stride)
        # stride=3 wins when the caller didn't pin an explicit cadence
        self._sample_every_s = sample_every_s
        self._yolo_model_file = resolve_yolo_model_path(yolo_model_path)

    # --------------------------------------------------------------- detector
    def _make_detector(self) -> _FrameDetector:
        """YOLO when the model exists, else packaged YuNet, else fail loud."""
        if self._detector is not None:
            return self._detector
        if self._yolo_model_file is not None:
            return YoloOnnxFaceDetector(self._yolo_model_file)
        try:
            # optional dependency (perception ships the 232 KB YuNet model);
            # guarded import keeps vision_fx standalone-installable
            from director_os.perception.faces import make_detector as _yunet_make

            return _yunet_make()
        except Exception as e:  # pragma: no cover - exercised only bare installs
            raise ComputeNotAvailable(
                "LocalOnnxCpuBackend has no detector: run scripts/fetch_yolov8n_face.py "
                "(yolov8n-face.onnx) or install director-os-perception (YuNet fallback)"
            ) from e

    # ------------------------------------------------------------------- jobs
    def detect_faces(
        self,
        video_path: Path | str,
        sample_every_s: float = DEFAULT_SAMPLE_EVERY_S,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        score_threshold: float = 0.35,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[FaceTrack]:
        import cv2

        det: _FrameDetector = self._make_detector()
        if self._sample_every_s is not None and sample_every_s == DEFAULT_SAMPLE_EVERY_S:
            sample_every_s = self._sample_every_s
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ComputeNotAvailable(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # stride=3 at 30 fps via the 0.1 s default cadence; a caller-pinned
        # smaller cadence is honored, never silently coarser than requested
        frame_stride = max(1, int(round(fps * sample_every_s)))

        detections_per_frame: list[list[tuple[int, int, int, int, float]]] = []
        t_per_frame: list[float] = []
        idx, decoded, failed = 0, 0, 0
        while decoded < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                try:
                    hits = det.detect(frame)
                except Exception:
                    failed += 1
                    idx += 1
                    continue
                decoded += 1
                detections_per_frame.append(
                    [(int(x), int(y), int(w), int(h), float(s)) for x, y, w, h, s in hits]
                )
                t_per_frame.append(round(idx / fps, 3) if fps else 0.0)
                if progress is not None and decoded % 100 == 0:
                    progress(decoded, max_samples)
            idx += 1
        cap.release()

        # the tracker consumes the detector's name for stats via capabilities
        self._last_detector_name = getattr(det, "name", "unknown")
        self._last_failed_frames = failed
        tracks = track_and_interpolate(detections_per_frame, t_per_frame)
        return [tr for tr in tracks if tr.min_score >= score_threshold]

    def separate_stems(
        self,
        audio_path: Path | str,
        spans: list[Span],
        out_dir: Path | str | None = None,
    ) -> StemsResult:
        if not spans:
            raise ComputeNotAvailable("separate_stems: no music-flagged spans — nothing to do")
        try:
            import demucs.separate  # noqa: F401
        except ImportError as e:
            raise ComputeNotAvailable(
                "local Demucs needs the optional extra: uv add demucs (torch CPU) — "
                "or flip COMPUTE_BACKEND=kaggle to run it on a free T4 instead"
            ) from e
        out = Path(out_dir) if out_dir else Path(os.getenv("OUTPUT_DIR", "./output")) / "stems"
        out.mkdir(parents=True, exist_ok=True)
        # demucs.separate.main(["--two_stems", "vocals", "-o", str(out), str(audio_path)])
        raise ComputeNotAvailable(
            "demucs CLI path is wired but unverified — prefer COMPUTE_BACKEND=kaggle for stems"
        )

    def transcribe_align(self, audio_path: Path | str, language: str = "") -> AsrAlignResult:
        raise ComputeNotAvailable(
            "no local ASR: use Gladia (perception, M2 path) or COMPUTE_BACKEND=kaggle for WhisperX"
        )

    def capabilities(self) -> dict[str, bool]:
        yolo = (
            self._yolo_model_file is not None
            or resolve_yolo_model_path(self._yolo_path) is not None
        )
        try:
            import demucs  # noqa: F401

            demucs_ok = True
        except ImportError:
            demucs_ok = False
        return {"faces": True, "stems": demucs_ok, "asr_align": False, "yolo_model": yolo}
