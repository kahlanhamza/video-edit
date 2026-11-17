"""Modal serverless GPU vision fleet (PLAN §8, M4) — the $0-credits GPU path.

Three serverless jobs riding one app ("directoros-vision"):

- `redact_video`  — T4 GPU: YOLOv8-face + YOLOv8n license-plate, each run
  through Ultralytics ByteTrack for persistent IDs (Person #1, Plate #1...)
  at 60-120 FPS; a 15-minute video costs ~20-30s of a T4.
- Gemini egress for geo-blocked regions lives in deploy/modal/vlm_proxy.py
  (app `directoros-vlm`, function `vlm_call`)
- `ocr_frame`     — CPU: PaddleOCR reads the AXON BODY timestamp overlay off
  one extracted frame (CPU is plenty for one image — no CUDA-matching pain).
- `ensure_weights` — idempotent weight download into a Modal Volume so cold
  starts are fast.

Verified weight sources (do not swap blindly):
  face  : https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt
  plate : https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt
          (YOLOv8n fine-tuned on keremberke/license-plate-object-detection)

One-time setup:  modal token set  (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
Deploy:          modal deploy deploy/modal/redact_gpu.py
Smoke:           modal run deploy/modal/redact_gpu.py::main --url <public-video-url>

The CPU worker calls this via `perception.modal_redact.redact_video_modal`,
which falls back to the local YuNet sweep whenever the fleet is unreachable —
Modal is an accelerator, never a fork of the pipeline.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import modal

app = modal.App("directoros-vision")

weights_volume = modal.Volume.from_name("directoros-weights", create_if_missing=True)

FACE_URL = os.getenv(
    "DOS_FACE_WEIGHT_URL",
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt",
)
PLATE_URL = os.getenv(
    "DOS_PLATE_WEIGHT_URL",
    "https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt",
)

# Modal free-tier accounts need a payment method on file before GPU functions
# are allowed ("Please add a payment method to use T4 GPU functions"). Deploy
# with DOS_MODAL_GPU=0 until then — ultralytics runs CPU-slow-but-correct, and
# vlm_call / ocr_frame are CPU anyway. Redeploy without the flag for the T4.
_GPU_REQUEST = os.getenv("DOS_MODAL_GPU", "1") != "0"
GPU_KWARGS: dict = {"gpu": "T4"} if _GPU_REQUEST else {}

# torch ships CUDA in the default PyPI wheel; ultralytics pulls it transitively.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3",
        "opencv-python-headless",
        "numpy<2.4",
        "requests",
    )
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "paddlepaddle==3.0.0",
        "paddleocr>=3.0",
        "opencv-python-headless",
        "numpy<2.4",
    )
)

_W = Path("/weights")


@app.function(image=gpu_image, volumes={"/weights": weights_volume}, timeout=600)
def ensure_weights() -> dict:
    """Download both YOLO weights into the volume; idempotent per file."""
    import requests

    out = {}
    for name, url in (("face.pt", FACE_URL), ("plate.pt", PLATE_URL)):
        dst = _W / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            out[name] = f"cached ({dst.stat().st_size} bytes)"
        else:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            _W.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(r.content)
            out[name] = f"downloaded ({len(r.content)} bytes)"
    weights_volume.commit()
    return out


def _load_yolo(weights_path: Path):
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _ensure_weights_inline() -> dict:
    """Download weights into THIS container's volume mount.

    A separate ensure_weights.remote() commits to the volume, but the caller's
    mount was snapshotted at container start — Modal volumes only guarantee
    visibility for FUTURE containers, so the download happens inline here.
    """
    import requests

    out = {}
    _W.mkdir(parents=True, exist_ok=True)
    for name, url in (("face.pt", FACE_URL), ("plate.pt", PLATE_URL)):
        dst = _W / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            out[name] = f"cached ({dst.stat().st_size} bytes)"
            continue
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        dst.write_bytes(r.content)
        out[name] = f"downloaded ({len(r.content)} bytes)"
    weights_volume.commit()
    return out


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=1800,
    **GPU_KWARGS,
)
def redact_video(
    video_b64: str | None = None,
    url: str | None = None,
    sample_fps: float = 5.0,
    conf: float = 0.25,
) -> dict:
    """Track faces + plates with persistent ByteTrack IDs; returns raw track JSON.

    Payload shape (consumed by perception.modal_redact.parse_tracks_payload):
    {"video": {w, h, fps, duration_s}, "sample_fps": f,
     "tracks": [{"id": int, "cls": "face|plate",
                 "frames": [{"t": s, "bbox": [x, y, w, h], "score": c}]}]}
    Coordinates are RAW-FOOTAGE pixels — the compiler maps them to render space.
    """
    import cv2

    weights_info = _ensure_weights_inline()
    print(f"[redact_gpu] weights: {weights_info}")

    src = Path("/tmp/input_video")
    if url:
        import requests

        r = requests.get(url, timeout=600)
        r.raise_for_status()
        src.write_bytes(r.content)
    elif video_b64:
        src.write_bytes(base64.b64decode(video_b64))
    else:
        raise ValueError("pass video_b64 or url")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video on GPU worker: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = n_frames / fps if fps else 0.0
    cap.release()

    stride = max(1, int(round(fps / max(0.1, sample_fps))))
    trackers = {"face": _load_yolo(_W / "face.pt"), "plate": _load_yolo(_W / "plate.pt")}

    # deterministic per-class ByteTrack walk with explicit frame timestamps;
    # coordinates land in RAW-FOOTAGE pixel space
    tracks = _track_with_indices(trackers, str(src), fps, stride, conf)

    return {
        "video": {"w": w, "h": h, "fps": round(fps, 3), "duration_s": round(duration_s, 3)},
        "sample_fps": sample_fps,
        "tracks": tracks,
    }


def _track_with_indices(
    trackers: dict, src: str, fps: float, stride: int, conf: float
) -> list[dict]:
    """Deterministic per-class tracking walk with explicit frame timestamps."""
    import cv2

    cap = cv2.VideoCapture(src)
    out: dict[tuple[str, int], dict] = {}
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                t = idx / fps
                for kind, model in trackers.items():
                    res = model.track(
                        frame,
                        tracker="bytetrack.yaml",
                        conf=conf,
                        persist=True,
                        verbose=False,
                    )[0]
                    if res.boxes is None or res.boxes.id is None:
                        continue
                    xywhn = res.boxes.xywh.cpu().numpy()  # center + size (pixels)
                    ids = res.boxes.id.cpu().numpy().astype(int)
                    scores = res.boxes.conf.cpu().numpy()
                    for (cx, cy, bw, bh), tid, sc in zip(xywhn, ids, scores, strict=False):
                        key = (kind, int(tid))
                        entry = out.setdefault(key, {"id": int(tid), "cls": kind, "frames": []})
                        entry["frames"].append(
                            {
                                "t": round(t, 3),
                                "bbox": [
                                    round(float(cx - bw / 2), 1),
                                    round(float(cy - bh / 2), 1),
                                    round(float(bw), 1),
                                    round(float(bh), 1),
                                ],
                                "score": round(float(sc), 3),
                            }
                        )
        idx += 1
    cap.release()
    return list(out.values())


@app.function(image=cpu_image, timeout=300)
def ocr_frame(image_b64: str, mime: str = "image/png") -> dict:
    """PaddleOCR over ONE frame; returns {'texts': [...], 'boxes': [...]}."""

    data = base64.b64decode(image_b64)
    arr = cv2_imdecode(data)
    texts: list[str] = []
    boxes: list[list[int]] = []
    try:
        ocr = _paddle_once()
        result = ocr.ocr(arr)
        for page in result or []:
            for line in page or []:
                if len(line) >= 2 and line[1] and len(line[1]) >= 1:
                    texts.append(str(line[1][0]))
                    boxes.append(line[0])
    except Exception as e:  # paddle import/usage drift — fail loud but structured
        return {"texts": [], "boxes": [], "error": str(e)[:300]}
    return {"texts": texts, "boxes": boxes}


_PADDLE = None


def _paddle_once():
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR

        _PADDLE = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    return _PADDLE


def cv2_imdecode(data: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decode failed on GPU worker")
    return img


@app.local_entrypoint()
def main(
    url: str | None = None,
    video_path: str | None = None,
    sample_fps: float = 5.0,
    conf: float = 0.25,
    ocr_image: str | None = None,
):
    if ocr_image:
        payload = base64.b64encode(Path(ocr_image).read_bytes()).decode()
        print(json.dumps(ocr_frame.remote(payload), indent=2)[:4000])
        return
    payload = redact_video.remote(
        video_b64=base64.b64encode(Path(video_path).read_bytes()).decode() if video_path else None,
        url=url,
        sample_fps=sample_fps,
        conf=conf,
    )
    n_tracks = len(payload.get("tracks", []))
    print(json.dumps({"video": payload["video"], "tracks": n_tracks}, indent=2))
    Path("/tmp/dos_redact_payload.json").write_text(json.dumps(payload))
    print("full payload -> /tmp/dos_redact_payload.json")
