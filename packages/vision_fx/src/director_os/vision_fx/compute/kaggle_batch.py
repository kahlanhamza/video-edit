"""KaggleBatchGpuBackend — free Tesla T4 cloud GPU, $0, zero card, zero SDK.

Heavy jobs (WhisperX, Demucs, long YOLO sweeps) run as script kernels pushed
through `core.kaggle.KaggleClient` (`KAGGLE_KEY=KGAT_...` bearer auth).
Contract: the kernel downloads its inputs from plain HTTPS URLs (media must
be reachable from the kernel — an R2 presigned URL is the production path),
writes raw detections/results JSON to `/kaggle/working`, and the host
re-uses the SAME tracker/interp logic in `vision_fx.compute.tracker`, so
track semantics never fork between local and cloud.

Verified live 2026-11: push -> running -> complete in ~30s on a Tesla T4
(15.6 GB VRAM), log harvested from /api/v1/kernels/output.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from director_os.core.kaggle import KaggleClient, kaggle_client_from_env
from director_os.vision_fx.compute.base import (
    AsrAlignResult,
    ComputeNotAvailable,
    FaceTrack,
    Span,
    StemsResult,
)
from director_os.vision_fx.compute.tracker import track_and_interpolate

KERNEL_PREFIX = "directoros"

# ------------------------------------------------------------------ templates
FACE_KERNEL_TEMPLATE = """# DirectorOS face sweep kernel — yolov8n-face.onnx on free T4
import json, subprocess, urllib.request
subprocess.run(
    ["pip", "install", "-q", "onnxruntime", "numpy", "opencv-python-headless"],
    check=True,
)
import cv2, numpy as np, onnxruntime as ort

VIDEO_URL = {video_url!r}
MODEL_URL = {model_url!r}
SAMPLE_EVERY_S = {sample_every_s}
OUT = "/kaggle/working/detections.json"

urllib.request.urlretrieve(MODEL_URL, "/kaggle/working/yolov8n-face.onnx")
subprocess.run(["curl", "-sL", "-o", "/kaggle/working/input_media", VIDEO_URL], check=True)

sess = ort.InferenceSession("/kaggle/working/yolov8n-face.onnx", providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
SIZE = 640

cap = cv2.VideoCapture("/kaggle/working/input_media")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
stride = max(1, int(round(fps * SAMPLE_EVERY_S)))

def letterbox(frame):
    h, w = frame.shape[:2]
    s = min(SIZE / w, SIZE / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    rz = cv2.resize(frame, (nw, nh))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[py:py+nh, px:px+nw] = rz
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob[None], s, px, py

def nms(dets, thr=0.45):
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
            if u > 0 and inter/u > thr:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept

records = []
idx, decoded = 0, 0
while decoded < {max_samples}:
    ok, frame = cap.read()
    if not ok: break
    if idx % stride == 0:
        blob, s, px, py = letterbox(frame)
        out = sess.run(None, {{inp_name: blob}})[0][0]
        dets = []
        for i in range(out.shape[1]):
            cx, cy, w, h = out[0, i], out[1, i], out[2, i], out[3, i]
            sc = float(out[4, i]) if out.shape[0] >= 5 else float(out[4:, i].max())
            if sc < {score_threshold}: continue
            x0 = (cx - w/2 - px) / s; y0 = (cy - h/2 - py) / s
            x1 = (cx + w/2 - px) / s; y1 = (cy + h/2 - py) / s
            dets.append((x0, y0, x1 - x0, y1 - y0, sc))
        dets = nms(dets)
        fh, fw = frame.shape[:2]
        for x0, y0, w0, h0, sc in dets:
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(fw, int(x0 + w0)), min(fh, int(y0 + h0))
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                records.append({{"t_s": round(idx / fps, 3), "x": x0, "y": y0,
                                 "w": x1 - x0, "h": y1 - y0, "score": round(sc, 3)}})
        decoded += 1
    idx += 1
cap.release()
json.dump({{"fps": fps, "sample_every_s": SAMPLE_EVERY_S,
            "frames_sampled": decoded, "detections": records}}, open(OUT, "w"))
print("DIRECTOROS_RESULT=" + json.dumps({{"frames_sampled": decoded, "detections": len(records)}}))
"""

DEMUCS_KERNEL_TEMPLATE = """# DirectorOS stems kernel — Demucs two-stems on free T4
import json, subprocess, urllib.request
subprocess.run(["pip", "install", "-q", "demucs"], check=True)
urllib.request.urlretrieve({audio_url!r}, "/kaggle/working/input_audio")
subprocess.run(["python", "-m", "demucs.separate", "--two_stems", "vocals",
                "-n", {model_name!r}, "-o", "/kaggle/working/stems",
                "/kaggle/working/input_audio"], check=True)
print("DIRECTOROS_RESULT=" + json.dumps({{"stems": ["vocals", "no_vocals"]}}))
"""

WHISPERX_KERNEL_TEMPLATE = """# DirectorOS whisperx kernel — transcribe + align on free T4
import json, subprocess, urllib.request
subprocess.run(["pip", "install", "-q", "whisperx"], check=True)
urllib.request.urlretrieve({audio_url!r}, "/kaggle/working/input_audio")
import whisperx
device = "cuda"
model = whisperx.load_model({model_name!r}, device, compute_type="float16")
audio = whisperx.load_audio("/kaggle/working/input_audio")
result = model.transcribe(audio, batch_size=16, language={language!r} or None)
align_model, meta = whisperx.load_align_model(result["language"], device)
result = whisperx.align(result["segments"], align_model, meta, audio, device,
                        return_char_alignments=False)
json.dump({{"language": result["language"], "segments": result["segments"]}},
          open("/kaggle/working/transcript.json", "w"))
print("DIRECTOROS_RESULT=" + json.dumps(
    {{"language": result["language"], "segments": len(result["segments"])}}))
"""

# canonical public mirror used for the sweep kernel (pinned in fetch script)
DEFAULT_MODEL_URL = (
    "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx"
)


class KaggleBatchGpuBackend:
    """Free-T4 batch adapter. Detect/stems/align all resolve through kernels."""

    name = "kaggle-gpu"

    def __init__(
        self,
        client: KaggleClient | None = None,
        out_dir: Path | str | None = None,
        run_timeout_s: float = 30 * 60,
        poll_interval_s: float = 15.0,
    ):
        self._client = client
        self._out_dir = Path(out_dir) if out_dir else None
        self._run_timeout_s = run_timeout_s
        self._poll_interval_s = poll_interval_s

    @property
    def client(self) -> KaggleClient:
        if self._client is None:
            self._client = kaggle_client_from_env()
        return self._client

    def _run(
        self,
        script: str,
        slug: str,
        title: str,
        *,
        internet: bool = True,
        out_dir: Path | None = None,
    ):
        out_dir = out_dir or self._out_dir
        return self.client.run_kernel(
            slug=f"{KERNEL_PREFIX}-{slug}",
            title=title,
            script=script,
            gpu=True,
            internet=internet,
            out_dir=out_dir,
            run_timeout_s=self._run_timeout_s,
            poll_interval_s=self._poll_interval_s,
        )

    # ------------------------------------------------------------------- jobs
    def detect_faces(
        self,
        video_path: Path | str,
        sample_every_s: float = 0.1,
        max_samples: int = 4000,
        score_threshold: float = 0.35,
        progress: Callable[[int, int], None] | None = None,
        video_url: str | None = None,
        out_dir: Path | str | None = None,
    ) -> list[FaceTrack]:
        """Sweep on T4, track + interpolate host-side (one tracker for all).

        `video_url` must be kernel-reachable (R2 presigned / public CDN).
        `video_path` is used only to name the kernel slug deterministically.
        """
        if not video_url:
            raise ComputeNotAvailable(
                "KaggleBatchGpuBackend.detect_faces needs a kernel-reachable video_url "
                "(R2 presigned URL or public CDN) — use LocalOnnxCpuBackend for local files"
            )
        out = Path(out_dir) if out_dir else (self._out_dir or Path("./output/kaggle"))
        run = self._run(
            FACE_KERNEL_TEMPLATE.format(
                video_url=video_url,
                model_url=DEFAULT_MODEL_URL,
                sample_every_s=sample_every_s,
                max_samples=max_samples,
                score_threshold=score_threshold,
            ),
            slug=f"faces-{abs(hash(str(video_path))) % 100000}",
            title="DirectorOS Face Sweep",
            internet=True,
            out_dir=out,
        )
        det_file = (run.out_dir or out) / "detections.json"
        if not det_file.exists():
            raise ComputeNotAvailable("kernel finished but detections.json missing from output")
        payload = json.loads(det_file.read_text())
        # regroup raw detections into per-frame lists ordered by t_s, then run
        # the SAME host-side tracker the local backend uses
        frames = sorted({d["t_s"] for d in payload["detections"]})
        frame_index = {t: k for k, t in enumerate(frames)}
        grouped: list[list[tuple[int, int, int, int, float]]] = [[] for _ in frames]
        for d in payload["detections"]:
            grouped[frame_index[d["t_s"]]].append(
                (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]), float(d["score"]))
            )
        return track_and_interpolate(grouped, frames)

    def separate_stems(
        self,
        audio_path: Path | str,
        spans: list[Span],
        out_dir: Path | str | None = None,
        audio_url: str | None = None,
        model_name: str = "htdemucs",
    ) -> StemsResult:
        if not audio_url:
            raise ComputeNotAvailable("KaggleBatchGpuBackend.separate_stems needs audio_url")
        out = Path(out_dir) if out_dir else (self._out_dir or Path("./output/kaggle"))
        self._run(
            DEMUCS_KERNEL_TEMPLATE.format(audio_url=audio_url, model_name=model_name),
            slug=f"stems-{abs(hash(str(audio_path))) % 100000}",
            title="DirectorOS Stems",
            out_dir=out,
        )
        stems_root = out / "stems"
        vocals = next(stems_root.rglob("vocals.wav"), None) if stems_root.exists() else None
        music = next(stems_root.rglob("no_vocals.wav"), None) if stems_root.exists() else None
        return StemsResult(vocals_path=vocals, music_path=music, spans=spans, backend=self.name)

    def transcribe_align(
        self,
        audio_path: Path | str,
        language: str = "",
        audio_url: str | None = None,
        model_name: str = "large-v2",
        out_dir: Path | str | None = None,
    ) -> AsrAlignResult:
        if not audio_url:
            raise ComputeNotAvailable("KaggleBatchGpuBackend.transcribe_align needs audio_url")
        out = Path(out_dir) if out_dir else (self._out_dir or Path("./output/kaggle"))
        self._run(
            WHISPERX_KERNEL_TEMPLATE.format(
                audio_url=audio_url, language=language, model_name=model_name
            ),
            slug=f"whisperx-{abs(hash(str(audio_path))) % 100000}",
            title="DirectorOS WhisperX",
            out_dir=out,
        )
        tfile = out / "transcript.json"
        if not tfile.exists():
            raise ComputeNotAvailable("kernel finished but transcript.json missing from output")
        payload = json.loads(tfile.read_text())
        words = [
            w
            for seg in payload.get("segments", [])
            for w in (seg.get("words") or [])
            if isinstance(w, dict)
        ]
        return AsrAlignResult(
            language=payload.get("language", language),
            words=words,
            segments=payload.get("segments", []),
            backend=self.name,
        )

    def capabilities(self) -> dict[str, bool]:
        try:
            from director_os.core.config import get_settings

            ok = bool(get_settings().kaggle_key)
        except Exception:
            ok = False
        return {"faces": ok, "stems": ok, "asr_align": ok}
