"""ModalGpuBackend — dormant by design; one .env flip when a card exists.

Modal's free $30/mo tier asks for card verification (verification fees are
why this backend is OFF by default). The adapter is fully wired behind
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`: the same job contracts as the other
adapters, executed as Modal functions on T4/L4/A10G with sub-second cold
starts. Until then the selector never reaches here, and KAGGLE covers the
free-cloud slot.
"""

from __future__ import annotations

import json
from pathlib import Path

from director_os.vision_fx.compute.base import (
    AsrAlignResult,
    ComputeNotAvailable,
    FaceTrack,
    Span,
    StemsResult,
)
from director_os.vision_fx.compute.tracker import track_and_interpolate

FACE_MODAL_SOURCE = """
import json
import cv2, numpy as np, onnxruntime as ort

SIZE = 640

def _letterbox(frame):
    h, w = frame.shape[:2]
    s = min(SIZE / w, SIZE / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    rz = cv2.resize(frame, (nw, nh))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[py:py+nh, px:px+nw] = rz
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob[None], s, px, py

def _nms(dets, thr=0.45):
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    kept = []
    for d in dets:
        ok = True
        for k in kept:
            ax, ay, aw, ah = d[:4]; bx, by, bw, bh = k[:4]
            x1, y1 = max(ax-aw/2, bx-bw/2), max(ay-ah/2, by-bh/2)
            x2, y2 = min(ax+aw/2, bx+bw/2), min(ay+ah/2, by+bh/2)
            inter = max(0, x2-x1) * max(0, y2-y1)
            u = aw*ah + bw*bh - inter
            if u > 0 and inter/u > thr: ok = False; break
        if ok: kept.append(d)
    return kept

def sweep(video_path: str, model_path: str, sample_every_s: float = 0.1,
          max_samples: int = 4000, score_threshold: float = 0.35) -> str:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * sample_every_s)))
    records = []
    idx, decoded = 0, 0
    while decoded < max_samples:
        ok, frame = cap.read()
        if not ok: break
        if idx % stride == 0:
            blob, s, px, py = _letterbox(frame)
            out = sess.run(None, {inp: blob})[0][0]
            dets = []
            for i in range(out.shape[1]):
                cx, cy, w, h = out[0, i], out[1, i], out[2, i], out[3, i]
                sc = float(out[4, i]) if out.shape[0] >= 5 else float(out[4:, i].max())
                if sc < score_threshold: continue
                x0 = (cx - w/2 - px) / s; y0 = (cy - h/2 - py) / s
                x1 = (cx + w/2 - px) / s; y1 = (cy + h/2 - py) / s
                dets.append((x0, y0, x1 - x0, y1 - y0, sc))
            dets = _nms(dets)
            fh, fw = frame.shape[:2]
            for x0, y0, w0, h0, sc in dets:
                x0, y0 = max(0, int(x0)), max(0, int(y0))
                x1, y1 = min(fw, int(x0 + w0)), min(fh, int(y0 + h0))
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    records.append({"t_s": round(idx / fps, 3), "x": x0, "y": y0,
                                    "w": x1 - x0, "h": y1 - y0, "score": round(sc, 3)})
            decoded += 1
        idx += 1
    cap.release()
    return json.dumps({"fps": fps, "detections": records})
"""


class ModalGpuBackend:
    """Serverless GPU adapter (T4/L4/A10G). Inert until MODAL_TOKEN_ID exists."""

    name = "modal-gpu"

    def __init__(self, token_id: str = "", token_secret: str = ""):
        from director_os.core.config import get_settings

        s = get_settings()
        self._token_id = token_id or s.modal_token_id
        self._token_secret = token_secret or s.modal_token_secret
        if not self._token_id or not self._token_secret:
            raise ComputeNotAvailable(
                "ModalGpuBackend requires MODAL_TOKEN_ID + MODAL_TOKEN_SECRET — "
                "kept dormant by design (card verification fees); KAGGLE_KEY covers the free path"
            )

    def _modal(self):
        try:
            import modal
        except ImportError as e:  # pragma: no cover
            raise ComputeNotAvailable("modal SDK not installed (uv add modal)") from e
        return modal

    def _app_and_fn(self, modal):
        app = modal.App("directoros-compute")
        image = modal.Image.debian_slim(python_version="3.12").pip_install(
            "onnxruntime", "numpy", "opencv-python-headless"
        )
        fn = app.function(image=image, gpu="T4", timeout=1800)(self._sweep_remote)
        return app, fn

    @staticmethod
    def _sweep_remote(
        video_url: str,
        model_url: str,
        sample_every_s: float,
        max_samples: int,
        score_threshold: float,
    ) -> str:
        """Runs inside Modal; downloads inputs, returns detections JSON."""
        import subprocess

        subprocess.run(["curl", "-sL", "-o", "/tmp/model.onnx", model_url], check=True)
        subprocess.run(["curl", "-sL", "-o", "/tmp/input_media", video_url], check=True)
        ns: dict = {}
        exec(FACE_MODAL_SOURCE, ns)  # noqa: S102 - pinned source constant above
        return ns["sweep"](
            "/tmp/input_media", "/tmp/model.onnx", sample_every_s, max_samples, score_threshold
        )

    # ------------------------------------------------------------------- jobs
    def detect_faces(
        self,
        video_path: Path | str,
        sample_every_s: float = 0.1,
        max_samples: int = 4000,
        score_threshold: float = 0.35,
        progress: object | None = None,
        video_url: str | None = None,
        model_url: str | None = None,
    ) -> list[FaceTrack]:
        if not video_url:
            raise ComputeNotAvailable(
                "ModalGpuBackend.detect_faces needs a Modal-reachable video_url"
            )
        from director_os.vision_fx.compute.kaggle_batch import DEFAULT_MODEL_URL

        modal = self._modal()
        app, fn = self._app_and_fn(modal)
        with app.run():
            payload = json.loads(
                fn.remote(
                    video_url,
                    model_url or DEFAULT_MODEL_URL,
                    sample_every_s,
                    max_samples,
                    score_threshold,
                )
            )
        frames = sorted({d["t_s"] for d in payload["detections"]})
        fidx = {t: k for k, t in enumerate(frames)}
        grouped: list[list[tuple[int, int, int, int, float]]] = [[] for _ in frames]
        for d in payload["detections"]:
            grouped[fidx[d["t_s"]]].append(
                (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]), float(d["score"]))
            )
        return track_and_interpolate(grouped, frames)

    def separate_stems(
        self,
        audio_path: Path | str,
        spans: list[Span],
        out_dir: Path | str | None = None,
    ) -> StemsResult:
        raise ComputeNotAvailable(
            "ModalGpuBackend.separate_stems: wired on demand (Demucs image) — "
            "use Kaggle adapter for the free path"
        )

    def transcribe_align(self, audio_path: Path | str, language: str = "") -> AsrAlignResult:
        raise ComputeNotAvailable(
            "ModalGpuBackend.transcribe_align: wired on demand (WhisperX image) — "
            "use Kaggle adapter for the free path"
        )

    def capabilities(self) -> dict[str, bool]:
        return {"faces": True, "stems": False, "asr_align": False}
