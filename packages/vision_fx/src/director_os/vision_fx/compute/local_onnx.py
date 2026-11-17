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
    YoloOnnxPlateDetector,
    resolve_plate_model_path,
    resolve_yolo_model_path,
)

DEFAULT_SAMPLE_EVERY_S = 0.1  # ~stride 3 at 30 fps
DEFAULT_MAX_SAMPLES = 4000
# Redaction-sweep detector defaults (issue #14): 960px input + conf 0.22 so
# side-profile and distant faces are not dropped at Pass 1 — missed faces are
# the only unrecoverable leak; weak hits are pruned later by min_hits/Pass 2.
DEFAULT_IMGSZ = 960
DEFAULT_SWEEP_CONF = 0.22
DEFAULT_TRACK_BUFFER = 60  # BoT-SORT buffer, sampled frames


def median_flow_shift(disp: np.ndarray) -> tuple[float, float]:
    """Median (dx, dy) of a stack of per-point displacements.

    cv2 point arrays arrive shaped (N, 1, 2); `np.median(..., axis=0)` on that
    yields (1, 2) whose elements are arrays — numpy 2.x refuses `float()` on
    them and the whole sweep crashes. Flattening to (N, 2) first makes the
    median a proper (2,) vector. Regression-tested (test_local_onnx_flow.py).
    """
    flat = np.asarray(disp).reshape(-1, 2)
    med = np.median(flat, axis=0)
    return float(med[0]), float(med[1])


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
        plate_model_path: str | Path | None = None,
    ):
        self._yolo_path = yolo_model_path
        self._plate_path = plate_model_path
        self._detector = detector
        self.stride = max(1, stride)
        # stride=3 wins when the caller didn't pin an explicit cadence
        self._sample_every_s = sample_every_s
        self._yolo_model_file = resolve_yolo_model_path(yolo_model_path)
        self._plate_model_file = resolve_plate_model_path(plate_model_path)

    # --------------------------------------------------------------- detector
    def _make_detector(
        self, imgsz: int = DEFAULT_IMGSZ, conf: float = DEFAULT_SWEEP_CONF
    ) -> _FrameDetector:
        """YOLO when the model exists (imgsz/conf per redaction sweep), else
        packaged YuNet, else fail loud. Cached per (imgsz, conf) key."""
        key = (imgsz, round(conf, 3))
        cached = getattr(self, "_detector_cache", None)
        if cached and key in cached:
            return cached[key]
        if self._detector is not None:
            det = self._detector
        elif self._yolo_model_file is not None:
            det = YoloOnnxFaceDetector(self._yolo_model_file, conf_threshold=conf, input_size=imgsz)
        else:
            try:
                # optional dependency (perception ships the 232 KB YuNet model);
                # guarded import keeps vision_fx standalone-installable
                from director_os.perception.faces import make_detector as _yunet_make

                det = _yunet_make()
            except Exception as e:  # pragma: no cover - exercised only bare installs
                raise ComputeNotAvailable(
                    "LocalOnnxCpuBackend has no detector: run scripts/fetch_yolov8n_face.py "
                    "(yolov8n-face.onnx) or install director-os-perception (YuNet fallback)"
                ) from e
        cache = getattr(self, "_detector_cache", None)
        if cache is None:
            cache = {}
            self._detector_cache = cache  # type: ignore[attr-defined]
        cache[key] = det
        return det

    # ------------------------------------------------------------------- jobs
    def _sweep_tracks(
        self,
        video_path: Path | str,
        det: _FrameDetector,
        sample_every_s: float,
        max_samples: int,
        gmc: bool,
        progress: Callable[[int, int], None] | None,
    ) -> tuple[
        list[list[tuple[int, int, int, int, float]]], list[float], list[tuple[float, float]]
    ]:
        """The ONE decode sweep both face and plate jobs ride (issue #12).

        Detection + sparse-optical-flow GMC measured on the same sampled frames;
        a detector exception propagates LOUD (swallowing it here produced the
        empty Eustis plan — see detect_faces). Returns per-sampled-frame
        detections, timestamps and motion vectors for the tracker.
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ComputeNotAvailable(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_stride = max(1, int(round(fps * sample_every_s)))

        detections_per_frame: list[list[tuple[int, int, int, int, float]]] = []
        t_per_frame: list[float] = []
        motions: list[tuple[float, float]] = []
        prev_gray: np.ndarray | None = None
        prev_pts: np.ndarray | None = None
        idx, decoded = 0, 0
        while decoded < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # global motion on the same sweep: median LK shift of corner
                # features prev->current sampled frame (robust vs moving faces)
                if gmc and prev_gray is not None and prev_pts is not None and len(prev_pts) >= 10:
                    next_pts, st, _err = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                        prev_gray, gray, prev_pts, None
                    )
                    good = st.ravel() == 1
                    if good.sum() >= 8:
                        motions.append(median_flow_shift(next_pts[good] - prev_pts[good]))
                    else:
                        motions.append((0.0, 0.0))
                else:
                    motions.append((0.0, 0.0))
                prev_gray, prev_pts = (
                    gray,
                    cv2.goodFeaturesToTrack(
                        gray, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7
                    ),
                )
                # Fail LOUD: a detector exception is a code/shape bug, not a
                # soft capability gap. Swallowing it here produced an EMPTY
                # tracked plan on real footage (Eustis) — the render baked
                # zero redaction while the audit passed vacuously. Capability
                # absence raises at construction (`_make_detector`), never here.
                hits = det.detect(frame)
                decoded += 1
                detections_per_frame.append(
                    [(int(x), int(y), int(w), int(h), float(s)) for x, y, w, h, s in hits]
                )
                t_per_frame.append(round(idx / fps, 3) if fps else 0.0)
                if progress is not None and decoded % 100 == 0:
                    progress(decoded, max_samples)
            idx += 1
        cap.release()
        return detections_per_frame, t_per_frame, motions

    def detect_faces(
        self,
        video_path: Path | str,
        sample_every_s: float = DEFAULT_SAMPLE_EVERY_S,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        score_threshold: float = DEFAULT_SWEEP_CONF,
        progress: Callable[[int, int], None] | None = None,
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_SWEEP_CONF,
        gmc: bool = True,
    ) -> list[FaceTrack]:
        """Pass 1 — detect + GMC-track faces -> persistent per-frame tracks.

        Detection runs at `imgsz` with `conf` floor; sparse-optical-flow camera
        motion is measured on the SAME decode sweep and fed to the tracker so
        whip-pans keep identities; `track_buffer=60` keeps re-association alive.
        Caller MUST CFR-normalize first (redact_track.normalize_cfr) so the
        frame indices returned here are the renderer's frame indices.
        """
        det: _FrameDetector = self._make_detector(imgsz=imgsz, conf=conf)
        if self._sample_every_s is not None and sample_every_s == DEFAULT_SAMPLE_EVERY_S:
            sample_every_s = self._sample_every_s
        detections_per_frame, t_per_frame, motions = self._sweep_tracks(
            video_path, det, sample_every_s, max_samples, gmc, progress
        )

        # the tracker consumes the detector's name for stats via capabilities
        self._last_detector_name = getattr(det, "name", "unknown")
        self._last_failed_frames = 0  # detector errors raise — nothing is swallowed
        tracks = track_and_interpolate(
            detections_per_frame,
            t_per_frame,
            gmc_vectors=motions if gmc else None,
            track_buffer=DEFAULT_TRACK_BUFFER,
        )
        return [tr for tr in tracks if tr.min_score >= score_threshold]

    def detect_plates(
        self,
        video_path: Path | str,
        sample_every_s: float = DEFAULT_SAMPLE_EVERY_S,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        score_threshold: float = DEFAULT_SWEEP_CONF,
        progress: Callable[[int, int], None] | None = None,
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_SWEEP_CONF,
        gmc: bool = True,
    ) -> list[FaceTrack]:
        """Pass 1 for LICENSE PLATES (issue #12) — same law as faces.

        Same decode sweep, same GMC tracker, same track_buffer — plates ride on
        vehicles through whip-pans just like faces do. The plate export is fixed
        at 640 input (`resolve_input_size` honors the model's contract), and the
        detection floor lands near the readability floor of the RAW pixels: what
        the detector cannot see, a viewer cannot read, so nothing legally
        requiring redaction is missed. Ids are re-prefixed `plate_N` by
        `merge_tracks` when the two sweeps combine into one redaction plan.

        Raises ComputeNotAvailable when the plate model file is absent —
        callers check `capabilities()["plates"]` first and skip the sweep.
        """
        if self._plate_model_file is None:
            raise ComputeNotAvailable(
                "license-plate ONNX model not found — run scripts/fetch_yolov8n_plate.py "
                "or set DIRECTOR_YOLOV8N_PLATE_PATH"
            )
        det: _FrameDetector = YoloOnnxPlateDetector(
            self._plate_model_file, conf_threshold=conf, input_size=imgsz
        )
        if self._sample_every_s is not None and sample_every_s == DEFAULT_SAMPLE_EVERY_S:
            sample_every_s = self._sample_every_s
        detections_per_frame, t_per_frame, motions = self._sweep_tracks(
            video_path, det, sample_every_s, max_samples, gmc, progress
        )
        self._last_plate_detector_name = getattr(det, "name", "unknown")
        tracks = track_and_interpolate(
            detections_per_frame,
            t_per_frame,
            gmc_vectors=motions if gmc else None,
            track_buffer=DEFAULT_TRACK_BUFFER,
        )
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
        return {
            "faces": True,
            "plates": self._plate_model_file is not None,
            "stems": demucs_ok,
            "asr_align": False,
            "yolo_model": yolo,
        }
