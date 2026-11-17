"""ComputeBackend port — heavy GPU/compute jobs behind one clean interface.

The port-and-adapter pattern used by `LLMBackend` (agent_runtime) and
`TtsBackend` (audio_engine), applied to vision/audio compute. Three adapters,
chosen by `COMPUTE_BACKEND` (default `local`):

- `LocalOnnxCpuBackend`   DEFAULT — $0, zero card. yolov8n-face.onnx via
  onnxruntime (stride=3 sampling + linear interpolation for skipped frames),
  YuNet fallback; on-demand Demucs on music-flagged spans when the optional
  `demucs` extra is installed.
- `KaggleBatchGpuBackend` OPTIONAL free cloud GPU — $0, zero card. Pushes
  script kernels through `core.kaggle.KaggleClient` (Tesla T4) for heavy
  WhisperX / Demucs / YOLO runs; raw detections are tracked + interpolated
  host-side so the tracker logic lives in exactly one place.
- `ModalGpuBackend`       OPTIONAL, dormant — wired behind `MODAL_TOKEN_ID`;
  the day a card exists, one .env flip unlocks sub-second serverless
  L4/A10G bursts without touching pipeline code.

Job kinds: `faces` (detect + track), `stems` (vocal/music separation),
`asr_align` (transcribe + word alignment). Every adapter returns the same
models (`FaceTrack`, `StemsResult`, ...) in RAW source pixel space, the same
canvas the renderer and studio already speak.
"""

from director_os.vision_fx.compute.base import (
    AsrAlignResult,
    ComputeBackend,
    ComputeNotAvailable,
    FaceTrack,
    Span,
    StemsResult,
    TrackHit,
)
from director_os.vision_fx.compute.selector import (
    available_backends,
    get_compute_backend,
)

__all__ = [
    "AsrAlignResult",
    "ComputeBackend",
    "ComputeNotAvailable",
    "FaceTrack",
    "Span",
    "StemsResult",
    "TrackHit",
    "available_backends",
    "get_compute_backend",
]
